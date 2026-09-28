#!/usr/bin/env python3
"""
CTA-Safe Data Augmentation - augment_brain_extracted1.py
============================================================================
Generates 5 augmented versions of each preprocessed CTA volume, plus a
contact-sheet QC image (original + all 5 augmentations) per subject.

WHY THESE AUGMENTATIONS AND NOT OTHERS:
  - NO LEFT-RIGHT FLIPPING. This is deliberate and non-negotiable for
    stroke/LVO data: flipping reverses which hemisphere appears affected,
    silently corrupting labels/laterality for anything downstream (LVO
    classification, hemisphere-symmetry features, clot burden score,
    etc.). Only rotation / scaling / translation / intensity-based
    transforms are used - none of them change which side is which.
  - Percentile-based intensity normalization (1st-99th percentile per
    image, not a fixed global range) is used instead of min/max
    normalization, because CTA volumes can have a handful of very bright
    voxels (contrast-filled vessels, bone, artifacts) that would otherwise
    dominate a min/max-based rescale and wash out the rest of the image.
  - Contrast jitter is centered on the image MEDIAN (of foreground/nonzero
    voxels), not the mean, for the same reason - the mean is more easily
    skewed by a small number of very bright or very dark outlier voxels
    than the median is.

"IMAGES SHOULD NOT VANISH":
  Every augmented volume is checked against the original: if the
  augmented volume's nonzero voxel count drops below
  --min_nonzero_fraction (default 0.85) of the original's nonzero count -
  which can happen if a rotation/scale/shift pushes brain tissue out of
  the field of view, or if intensity clipping is too aggressive - the
  augmentation is automatically RETRIED with progressively gentler
  parameters (smaller angle/scale/shift), and if it still fails after a
  few attempts, falls back to an intensity-only augmentation (which can
  never empty the volume, since it doesn't move any voxels). A volume is
  never written to disk if it fails this check, so "vanished" images
  should not be possible.

Input : preprocessed_extracted/<subject>.nii(.gz)
Output: augmented1/<subject>/<subject>_aug1.nii.gz ... <subject>_aug5.nii.gz
        augmented1/<subject>/<subject>_augmentation_qc.png   (original + all 5, side by side)

Usage:
    python augment_brain_extracted1.py
    python augment_brain_extracted1.py --original_dir preprocessed_extracted --output_dir augmented1
    python augment_brain_extracted1.py --seed 7 --min_nonzero_fraction 0.90
"""

import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import rotate, zoom, shift as nd_shift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MIN_NONZERO_FRACTION = 0.85
DEFAULT_MAX_RETRIES = 3
NUM_AUGMENTATIONS = 5


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_volume(path):
    img = nib.load(path)
    return np.squeeze(np.asarray(img.get_fdata(dtype=np.float32))), img.affine, img.header


def get_si_axis(affine):
    """Finds which array axis corresponds to Superior-Inferior, so the QC
    contact sheet can show an axial (top-down) mid-slice - the most
    recognizable view for a quick visual sanity check."""
    codes = aff2axcodes(affine)
    for i, c in enumerate(codes):
        if c in ("S", "I"):
            return i
    return 2  # fallback: assume last axis


def find_input_files(input_dir):
    files = []
    for fname in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, fname)
        if os.path.isfile(path) and (fname.lower().endswith(".nii.gz") or fname.lower().endswith(".nii")):
            stem = fname[:-7] if fname.lower().endswith(".nii.gz") else fname[:-4]
            files.append((stem, path))
    return files


# --------------------------------------------------------------------------
# Safe, non-flip augmentation primitives
# --------------------------------------------------------------------------
def percentile_normalize(volume, low_pct=1, high_pct=99):
    """Clips to the [low_pct, high_pct] percentile range (robust to a
    handful of very bright/dark outlier voxels - contrast-filled vessels,
    bone, noise spikes) then rescales linearly back to the original
    min/max range, stretching contrast without being outlier-sensitive."""
    nonzero = volume[volume != 0]
    if nonzero.size == 0:
        return volume.copy()
    p_lo, p_hi = np.percentile(nonzero, [low_pct, high_pct])
    if p_hi <= p_lo:
        return volume.copy()
    clipped = np.clip(volume, p_lo, p_hi)
    orig_min, orig_max = float(volume.min()), float(volume.max())
    rescaled = (clipped - p_lo) / (p_hi - p_lo) * (orig_max - orig_min) + orig_min
    return rescaled.astype(np.float32)


def median_contrast_jitter(volume, factor):
    """Adjusts contrast around the foreground MEDIAN (not mean - more
    robust to outlier bright/dark voxels): new = median + factor*(old-median).
    factor > 1 increases contrast, < 1 decreases it. Background (zero)
    voxels are left untouched so the brain/skull boundary doesn't shift."""
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    med = float(np.median(volume[fg_mask]))
    out = volume.copy()
    out[fg_mask] = med + factor * (volume[fg_mask] - med)
    return out.astype(np.float32)


def small_rotation(volume, angle_deg, axes):
    """Small in-plane rotation. reshape=False keeps the array shape fixed;
    mode='nearest' avoids introducing hard black borders (which could
    trip the 'vanished image' check and also don't reflect anything a
    real CTA acquisition would look like)."""
    return rotate(volume, angle=angle_deg, axes=axes, reshape=False,
                  order=1, mode="nearest").astype(np.float32)


def small_scale(volume, factor):
    """Small zoom in/out, then center-crop or pad back to the original
    shape so every augmented volume stays the same size as the input."""
    zoomed = zoom(volume, zoom=factor, order=1)
    out = np.zeros_like(volume)
    src_shape = np.array(zoomed.shape)
    dst_shape = np.array(volume.shape)

    src_start = np.maximum((src_shape - dst_shape) // 2, 0)
    dst_start = np.maximum((dst_shape - src_shape) // 2, 0)
    copy_shape = np.minimum(src_shape, dst_shape)

    src_slices = tuple(slice(s, s + c) for s, c in zip(src_start, copy_shape))
    dst_slices = tuple(slice(s, s + c) for s, c in zip(dst_start, copy_shape))
    out[dst_slices] = zoomed[src_slices]
    return out.astype(np.float32)


def small_translation(volume, shift_vox):
    return nd_shift(volume, shift=shift_vox, order=1, mode="nearest").astype(np.float32)


def add_mild_noise(volume, sigma_fraction, rng):
    """Gaussian noise scaled to a small fraction of the foreground std,
    applied only to foreground voxels so background stays clean."""
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    fg_std = float(volume[fg_mask].std())
    sigma = sigma_fraction * fg_std
    out = volume.copy()
    noise = rng.normal(0, sigma, size=volume.shape).astype(np.float32)
    out[fg_mask] = volume[fg_mask] + noise[fg_mask]
    return out


# --------------------------------------------------------------------------
# "Not vanished" validation
# --------------------------------------------------------------------------
def nonzero_fraction_ok(aug_volume, orig_nonzero_count, min_fraction):
    if orig_nonzero_count == 0:
        return True  # nothing to preserve
    aug_nonzero = int(np.count_nonzero(aug_volume))
    return aug_nonzero >= min_fraction * orig_nonzero_count


# --------------------------------------------------------------------------
# The 5 augmentation "recipes"
# --------------------------------------------------------------------------
def augmentation_1(volume, rng, in_plane_axes, strength=1.0):
    angle = rng.uniform(-7, 7) * strength
    out = small_rotation(volume, angle, in_plane_axes)
    out = percentile_normalize(out)
    return out, {"type": "rotation+percentile_norm", "angle_deg": round(angle, 2)}


def augmentation_2(volume, rng, in_plane_axes, strength=1.0):
    factor = 1.0 + rng.uniform(-0.03, 0.03) * strength
    out = small_scale(volume, factor)
    contrast_factor = 1.0 + rng.uniform(-0.15, 0.15) * strength
    out = median_contrast_jitter(out, contrast_factor)
    return out, {"type": "scale+median_contrast", "scale_factor": round(factor, 3),
                 "contrast_factor": round(contrast_factor, 3)}


def augmentation_3(volume, rng, in_plane_axes, strength=1.0):
    angle = rng.uniform(-5, 5) * strength
    factor = 1.0 + rng.uniform(-0.02, 0.02) * strength
    out = small_rotation(volume, angle, in_plane_axes)
    out = small_scale(out, factor)
    out = percentile_normalize(out)
    return out, {"type": "rotation+scale+percentile_norm", "angle_deg": round(angle, 2),
                 "scale_factor": round(factor, 3)}


def augmentation_4(volume, rng, in_plane_axes, strength=1.0):
    out = percentile_normalize(volume, low_pct=2 * min(strength, 1.0) + 0.5, high_pct=99.5)
    gamma = 1.0 + rng.uniform(-0.10, 0.10) * strength
    fg_mask = out != 0
    out2 = out.copy()
    if np.any(fg_mask):
        vmin, vmax = float(out[fg_mask].min()), float(out[fg_mask].max())
        if vmax > vmin:
            norm01 = (out[fg_mask] - vmin) / (vmax - vmin)
            out2[fg_mask] = (np.power(norm01, gamma) * (vmax - vmin) + vmin)
    return out2.astype(np.float32), {"type": "percentile_norm+gamma", "gamma": round(gamma, 3)}


def augmentation_5(volume, rng, in_plane_axes, strength=1.0):
    shift_vox = [rng.uniform(-2, 2) * strength if ax != None else 0 for ax in range(volume.ndim)]
    out = small_translation(volume, shift_vox)
    contrast_factor = 1.0 + rng.uniform(-0.10, 0.10) * strength
    out = median_contrast_jitter(out, contrast_factor)
    out = add_mild_noise(out, sigma_fraction=0.02 * strength, rng=rng)
    return out, {"type": "translation+median_contrast+mild_noise",
                 "shift_vox": [round(s, 2) for s in shift_vox],
                 "contrast_factor": round(contrast_factor, 3)}


AUGMENTATION_RECIPES = [augmentation_1, augmentation_2, augmentation_3, augmentation_4, augmentation_5]


def generate_one_augmentation(volume, recipe_fn, rng, in_plane_axes, orig_nonzero_count,
                               min_nonzero_fraction, max_retries):
    """Runs recipe_fn, checking the 'not vanished' condition each time and
    retrying with a smaller `strength` if it fails. Falls back to a pure
    intensity-only transform (guaranteed not to lose voxels) as a last
    resort."""
    strength = 1.0
    for attempt in range(max_retries):
        out, meta = recipe_fn(volume, rng, in_plane_axes, strength=strength)
        if nonzero_fraction_ok(out, orig_nonzero_count, min_nonzero_fraction):
            meta["attempt"] = attempt + 1
            meta["fallback_used"] = False
            return out, meta
        strength *= 0.5  # try a gentler version of the same recipe

    # last resort: intensity-only augmentation can't move voxels out of frame
    contrast_factor = 1.0 + rng.uniform(-0.10, 0.10)
    out = median_contrast_jitter(volume, contrast_factor)
    out = percentile_normalize(out)
    return out, {"type": "fallback_intensity_only", "contrast_factor": round(contrast_factor, 3),
                 "attempt": max_retries, "fallback_used": True}


# --------------------------------------------------------------------------
# QC contact sheet
# --------------------------------------------------------------------------
def save_contact_sheet(subject_id, original, augmented_list, meta_list, si_axis, out_path):
    mid_slice = original.shape[si_axis] // 2
    sl = [slice(None)] * 3
    sl[si_axis] = mid_slice

    images = [original] + augmented_list
    titles = ["Original"] + [f"Aug {i+1}\n{m['type']}" for i, m in enumerate(meta_list)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    vmin, vmax = np.percentile(original[original != 0], [1, 99]) if np.any(original != 0) else (0, 1)

    for ax, img, title in zip(axes, images, titles):
        img_slice = np.rot90(img[tuple(sl)])
        ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"{subject_id} - original vs 5 augmentations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-subject pipeline
# --------------------------------------------------------------------------
def process_subject(subject_id, path, output_dir, rng, min_nonzero_fraction, max_retries):
    print(f"\n[{subject_id}] loading ...")
    volume, affine, header = load_volume(path)
    si_axis = get_si_axis(affine)
    in_plane_axes = tuple(a for a in range(3) if a != si_axis)  # rotate within the axial plane
    orig_nonzero_count = int(np.count_nonzero(volume))

    subj_out = os.path.join(output_dir, subject_id)
    os.makedirs(subj_out, exist_ok=True)

    augmented_list = []
    meta_list = []
    for i, recipe_fn in enumerate(AUGMENTATION_RECIPES, start=1):
        out, meta = generate_one_augmentation(volume, recipe_fn, rng, in_plane_axes,
                                               orig_nonzero_count, min_nonzero_fraction, max_retries)
        aug_path = os.path.join(subj_out, f"{subject_id}_aug{i}.nii.gz")
        nib.save(nib.Nifti1Image(out, affine, header), aug_path)
        fallback_note = " [FALLBACK USED]" if meta.get("fallback_used") else ""
        print(f"    aug{i}: {meta['type']}{fallback_note} -> saved {aug_path}")
        augmented_list.append(out)
        meta_list.append(meta)

    qc_path = os.path.join(subj_out, f"{subject_id}_augmentation_qc.png")
    save_contact_sheet(subject_id, volume, augmented_list, meta_list, si_axis, qc_path)
    print(f"    QC contact sheet -> {qc_path}")


def main():
    parser = argparse.ArgumentParser(description="CTA-safe data augmentation (no flipping) with a "
                                                   "not-vanished check and a QC contact sheet per subject.")
    parser.add_argument("--original_dir", default="preprocessed_extracted")
    parser.add_argument("--output_dir", default="augmented1")
    parser.add_argument("--min_nonzero_fraction", type=float, default=DEFAULT_MIN_NONZERO_FRACTION,
                         help="An augmented volume is rejected/retried if its nonzero voxel count drops "
                              "below this fraction of the original's. Default: %(default)s")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                         help="Retries (with progressively gentler parameters) before falling back to an "
                              "intensity-only augmentation. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducible augmentations. Default: not fixed (random each run).")
    args = parser.parse_args()

    if not os.path.isdir(args.original_dir):
        print(f"'{args.original_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    input_files = find_input_files(args.original_dir)
    if not input_files:
        print(f"No .nii/.nii.gz files found in '{args.original_dir}'.")
        return

    print(f"Found {len(input_files)} subject(s). Output -> {args.output_dir}")
    for subject_id, path in input_files:
        process_subject(subject_id, path, args.output_dir, rng, args.min_nonzero_fraction, args.max_retries)

    print(f"\nAll done. {len(input_files)} subject(s) x {NUM_AUGMENTATIONS} augmentations written to: {args.output_dir}")


if __name__ == "__main__":
    main()