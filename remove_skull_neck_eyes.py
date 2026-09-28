import os
import glob
import csv
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
from scipy import ndimage

# ---------- CONFIGURE ----------
# IMPORTANT: run this on the RAW nifti files (original HU values),
# not on the already-clipped/normalized output.
INPUT_DIR = r"D:\Desktop\no_lvo\test"
OUTPUT_DIR = r"D:\Desktop\no_lvo\brain_extracted"
# --------------------------------

TARGET_VOXEL = (1.0, 1.0, 1.0)

BONE_THRESHOLD = 300         # HU above this = bone
OPEN_ITERATIONS = 3          # erosion/dilation depth used to break thin gaps (orbits, sinuses,
                              # foramen magnum) that otherwise leak the brain mask into face/scalp/neck
NECK_AREA_RATIO = 0.25       # fallback: fraction of peak head area that marks the neck transition (was 0.4)
HEAD_SEARCH_FRACTION = 1.0   # search the FULL volume for the head's peak area - now that
                              # skull removal no longer leaks into the neck/shoulders, mask1
                              # is already bounded near the skull base, so this restriction
                              # is unnecessary and was cutting into real brain
NECK_DROP_RATIO = 0.65       # require a bigger local area drop before cutting (was too easily
                              # triggered near the top of the head, which naturally narrows there)
EYE_ANTERIOR_FRACTION = 0.20 # fraction of head kept as "anterior" search zone (was 0.35, too aggressive)
EYE_Z_FRACTION = (0.05, 0.25)  # inferior-third search zone, narrowed (was 0.05-0.40)
VESSEL_HU_MIN = 120          # HU above this = likely contrast-enhanced vessel / dense tissue

# ---- validation thresholds: extractions outside these get REJECTED (not saved) ----
MIN_BRAIN_VOLUME_ML = 400     # a real brain/intracranial cavity won't be smaller than this
MAX_BRAIN_VOLUME_ML = 2500    # or bigger than this
MIN_HEIGHT_MM = 60            # minimum top-to-bottom extent - catches "thin sliver" failures
MIN_FILL_RATIO = 0.15         # mask voxels / bounding-box voxels - catches thin shells/rims
REJECTED_LOG_PATH_NAME = "rejected_subjects.csv"


# ---------------------------------------------------------------------
# STEP 0: ORIENTATION STANDARDIZATION (RAS)
# ---------------------------------------------------------------------
def standardize_orientation(img):
    """Reorient the image to the closest canonical RAS+ orientation.
    This runs FIRST, before any resampling or extraction, so every
    downstream step (including neck-crop and eye-removal, which assume
    R/A/S axis directions) works on a consistently oriented volume."""
    return nib.as_closest_canonical(img)


# ---------------------------------------------------------------------
# STEP 0.5: RESAMPLING (1x1x1 mm)
# ---------------------------------------------------------------------
def resample_isotropic(img, voxel_sizes=TARGET_VOXEL):
    """Resample the RAS-oriented image to isotropic voxel spacing."""
    return nibproc.resample_to_output(img, voxel_sizes=voxel_sizes, order=3)


def load_and_prepare(nifti_path):
    """Load a raw NIfTI file and run the two prep steps (RAS orientation,
    then resampling) before any brain-extraction step touches it."""
    img = nib.load(nifti_path)

    img = standardize_orientation(img)   # STEP 0
    img = resample_isotropic(img)        # STEP 0.5

    data = np.squeeze(img.get_fdata().astype(np.float32))
    return img, data


# ---------------------------------------------------------------------
# STEP 1: SKULL REMOVAL
# ---------------------------------------------------------------------
def remove_skull(data, bone_threshold=BONE_THRESHOLD, open_iterations=OPEN_ITERATIONS):
    """Threshold-based skull stripping.

    The naive version (largest connected non-bone component) tends to leak:
    small gaps in the skull (orbits, sinuses, ear canals, foramen magnum) -
    often widened by resampling blur - let the intracranial cavity connect
    to the face/scalp/neck soft tissue. Since that combined blob is bigger
    than the brain alone, "largest component" ends up picking a thin outer
    shell (scalp) instead of the solid brain interior.

    Fix: morphological opening. Erode first to break those thin bridges,
    find the largest surviving piece (now truly isolated to the brain),
    then dilate back to restore its real size - constrained to the original
    candidate mask so it can't regrow back across the same gap.
    """
    air_val = float(np.min(data))

    body_mask = data > -300  # exclude surrounding air
    body_mask = ndimage.binary_fill_holes(body_mask)

    bone_mask = data > bone_threshold
    bone_mask = ndimage.binary_dilation(bone_mask, iterations=1)

    candidate = body_mask & (~bone_mask)

    eroded = ndimage.binary_erosion(candidate, iterations=open_iterations)
    labeled, num = ndimage.label(eroded)
    if num == 0:
        return data.copy(), np.zeros_like(data, dtype=bool)

    sizes = ndimage.sum(eroded, labeled, range(1, num + 1))
    largest_label = int(np.argmax(sizes)) + 1
    seed = labeled == largest_label

    brain_mask = ndimage.binary_dilation(seed, iterations=open_iterations)
    brain_mask = brain_mask & candidate  # don't regrow past the original tissue/bone boundary
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    brain_mask = ndimage.binary_closing(brain_mask, iterations=2)

    output = np.full_like(data, air_val)
    output[brain_mask] = data[brain_mask]
    return output, brain_mask


# ---------------------------------------------------------------------
# STEP 2: NECK REMOVAL
# ---------------------------------------------------------------------
def remove_neck(data, mask,
                 area_ratio_threshold=NECK_AREA_RATIO,
                 head_search_fraction=HEAD_SEARCH_FRACTION,
                 drop_ratio=NECK_DROP_RATIO):
    """Crop the neck. CTA scans often extend down into the neck/shoulders,
    which can have a LARGER cross-sectional area than the head itself - so
    we only search for the head's peak area within the TOP portion of the
    volume (head_search_fraction), then walk down from there and stop at
    the first slice where the area drops sharply (skull-base / neck
    transition), instead of relying on a global peak that can be thrown
    off by the shoulders."""
    z_dim = data.shape[2]
    areas = np.array([mask[:, :, z].sum() for z in range(z_dim)], dtype=float)
    if areas.max() == 0:
        return data.copy(), mask.copy()

    search_start = int(z_dim * (1 - head_search_fraction))
    z_peak = search_start + int(np.argmax(areas[search_start:]))

    # light smoothing so single noisy slices don't trigger a false cut
    kernel = np.ones(5) / 5
    smoothed = np.convolve(areas, kernel, mode="same")

    z_cut = 0
    found_sharp_drop = False
    for z in range(z_peak, 0, -1):
        if smoothed[z] <= 1e-6:
            continue
        local_drop = (smoothed[z] - smoothed[z - 1]) / smoothed[z]
        if local_drop > drop_ratio:
            z_cut = z
            found_sharp_drop = True
            break

    if not found_sharp_drop:
        # fallback to the simpler global-ratio method
        threshold_area = areas[z_peak] * area_ratio_threshold
        for z in range(z_peak, -1, -1):
            if areas[z] < threshold_area:
                z_cut = z
                break

    print(f"    [neck debug] z_dim={z_dim}, search_start={search_start}, "
          f"z_peak={z_peak} (area={areas[z_peak]:.0f}), "
          f"z_cut={z_cut} (sharp_drop={found_sharp_drop})")

    new_mask = mask.copy()
    new_mask[:, :, :z_cut] = False
    output = data.copy()
    output[:, :, :z_cut] = np.min(data)
    return output, new_mask


# ---------------------------------------------------------------------
# STEP 3: EYE REMOVAL
# ---------------------------------------------------------------------
def remove_eyes(data, mask,
                 anterior_fraction=EYE_ANTERIOR_FRACTION,
                 z_fraction=EYE_Z_FRACTION):
    """Coarse anatomical ROI removal: zeroes out the anterior, inferior-third
    region of the head where the orbits sit. Approximate only."""
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return data.copy(), mask.copy()

    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    y_cutoff = int(y_min + (1 - anterior_fraction) * (y_max - y_min))
    z_low = int(z_min + z_fraction[0] * (z_max - z_min))
    z_high = int(z_min + z_fraction[1] * (z_max - z_min))

    eye_region = np.zeros_like(mask)
    eye_region[:, y_cutoff:, z_low:z_high] = True

    new_mask = mask & (~eye_region)
    output = data.copy()
    output[mask & eye_region] = np.min(data)
    return output, new_mask


# ---------------------------------------------------------------------
# STEP 4: SOFT TISSUE REMOVAL
# ---------------------------------------------------------------------
def remove_soft_tissue(data, mask, vessel_hu_min=VESSEL_HU_MIN):
    """Isolate higher-HU vascular/dense structures, discarding lower-HU
    soft tissue (parenchyma, fat, etc.)."""
    vessel_mask = mask & (data > vessel_hu_min)
    output = np.full_like(data, np.min(data))
    output[vessel_mask] = data[vessel_mask]
    return output, vessel_mask


# ---------------------------------------------------------------------
# VALIDATION: decide whether the extraction is good enough to save
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# CLEANUP: keep only the single largest connected piece
# ---------------------------------------------------------------------
def keep_largest_component(data, mask, open_iterations=4):
    """After skull/neck/eye removal, small leftover fragments (skull-base
    remnants, orbit pieces, etc.) can still remain connected to the main
    brain through a thin 3D bridge. Use proper morphological opening
    (erosion then dilation, WITHOUT intersecting back with the original
    mask) to permanently break those bridges, then keep only the largest
    resulting piece. Intersecting back with the original mask (an earlier
    version of this function did that) lets the dilation regrow straight
    back through the same bridge into the fragment, undoing the cleanup."""
    opened = ndimage.binary_opening(mask, iterations=open_iterations)
    labeled, num = ndimage.label(opened)
    if num == 0:
        return data.copy(), mask.copy()

    sizes = ndimage.sum(opened, labeled, range(1, num + 1))
    largest_label = int(np.argmax(sizes)) + 1
    cleaned_mask = labeled == largest_label

    sorted_sizes = sorted(sizes, reverse=True)
    top5 = [int(s) for s in sorted_sizes[:5]]
    print(f"    [fragment debug] found {num} separate piece(s) after opening "
          f"(open_iterations={open_iterations}); top sizes: {top5}; "
          f"kept largest ({int(sizes[largest_label - 1])} voxels)")

    output = np.full_like(data, np.min(data))
    output[cleaned_mask] = data[cleaned_mask]
    return output, cleaned_mask


def validate_extraction(mask,
                         min_volume_ml=MIN_BRAIN_VOLUME_ML,
                         max_volume_ml=MAX_BRAIN_VOLUME_ML,
                         min_height_mm=MIN_HEIGHT_MM,
                         min_fill_ratio=MIN_FILL_RATIO):
    """Sanity-check the extracted brain mask. Returns (passed, reasons).
    Catches the two failure modes seen in practice:
      - "thin sliver" (neck-removal cut away almost everything)
      - "thin shell" (skull-removal leaked into scalp/face, so the mask
        is a hollow rim rather than a solid filled brain)
    Voxels are 1x1x1mm after resampling, so voxel count == volume in mm3.
    """
    reasons = []
    voxel_count = int(mask.sum())
    volume_ml = voxel_count / 1000.0

    if voxel_count == 0:
        return False, ["mask is completely empty"]

    if volume_ml < min_volume_ml:
        reasons.append(f"volume too small ({volume_ml:.0f} mL < {min_volume_ml} mL) - likely a sliver")
    if volume_ml > max_volume_ml:
        reasons.append(f"volume too large ({volume_ml:.0f} mL > {max_volume_ml} mL) - likely leaked into face/neck")

    coords = np.array(np.where(mask))
    z_extent = int(coords[2].max() - coords[2].min() + 1)
    x_extent = int(coords[0].max() - coords[0].min() + 1)
    y_extent = int(coords[1].max() - coords[1].min() + 1)

    if z_extent < min_height_mm:
        reasons.append(f"height too small ({z_extent} mm < {min_height_mm} mm) - likely a thin sliver")

    bbox_volume = x_extent * y_extent * z_extent
    fill_ratio = voxel_count / bbox_volume if bbox_volume > 0 else 0
    if fill_ratio < min_fill_ratio:
        reasons.append(f"low fill ratio ({fill_ratio:.2f} < {min_fill_ratio}) - likely a thin shell, not a solid brain")

    passed = len(reasons) == 0
    return passed, reasons


def process_file(nifti_path, output_dir, rejected_rows):
    filename = os.path.basename(nifti_path)
    subject_name = filename.replace(".nii.gz", "").replace(".nii", "")
    print(f"Processing {filename} ...")

    img, data = load_and_prepare(nifti_path)
    print(f"  [0/4] RAS orientation + resampling done, shape={data.shape}")

    data1, mask1 = remove_skull(data)
    print(f"  [1/4] skull removed  (brain voxels: {mask1.sum()})")

    data2, mask2 = remove_neck(data1, mask1)
    print(f"  [2/4] neck removed   (remaining voxels: {mask2.sum()})")

    data3, mask3 = remove_eyes(data2, mask2)
    print(f"  [3/4] eyes removed   (remaining voxels: {mask3.sum()})")

    data3, mask3 = keep_largest_component(data3, mask3, open_iterations=8)
    print(f"  [3.5/4] disconnected fragments removed (remaining voxels: {mask3.sum()})")

    passed, reasons = validate_extraction(mask3)
    if not passed:
        print(f"  ✗ REJECTED - not saving. Reasons: {'; '.join(reasons)}")
        rejected_rows.append({
            "subject": subject_name,
            "reasons": "; ".join(reasons),
            "voxel_count": int(mask3.sum()),
        })
        return

    data4, mask4 = remove_soft_tissue(data3, mask3)
    print(f"  [4/4] soft tissue removed (vessel/dense voxels: {mask4.sum()})")

    os.makedirs(output_dir, exist_ok=True)

    # Full brain volume with skull/neck/eyes removed (HU values preserved) -
    # feed THIS into your clipping/normalization/enhancement pipeline.
    out_img = nib.Nifti1Image(data3.astype(np.float32), img.affine, img.header)
    out_path = os.path.join(output_dir, f"{subject_name}_extracted.nii.gz")
    nib.save(out_img, out_path)

    # Vessel-only mask, saved separately for reference/QC
    mask_img = nib.Nifti1Image(mask4.astype(np.uint8), img.affine, img.header)
    mask_path = os.path.join(output_dir, f"{subject_name}_vessel_mask.nii.gz")
    nib.save(mask_img, mask_path)

    print(f"  ✓ PASSED - saved {out_path}")
    print(f"  -> saved {mask_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nifti_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.nii"))
        + glob.glob(os.path.join(INPUT_DIR, "*.nii.gz"))
    )

    if not nifti_files:
        print(f"No NIfTI files found in {INPUT_DIR}")
        return

    rejected_rows = []
    for f in nifti_files:
        process_file(f, OUTPUT_DIR, rejected_rows)
        print()

    if rejected_rows:
        report_path = os.path.join(OUTPUT_DIR, REJECTED_LOG_PATH_NAME)
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "reasons", "voxel_count"])
            writer.writeheader()
            writer.writerows(rejected_rows)
        print(f"{len(rejected_rows)} subject(s) REJECTED and not saved - see {report_path}")

    passed_count = len(nifti_files) - len(rejected_rows)
    print(f"Done. {passed_count}/{len(nifti_files)} subjects saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()