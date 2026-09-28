import os
import glob
import csv
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIGURE ----------
RAW_DIR = r"D:\Desktop\no_lvo\test"
EXTRACTED_DIR = r"D:\Desktop\no_lvo\brain_extracted"
QC_DIR = r"D:\Desktop\no_lvo\qc_brain_extraction"
# --------------------------------

REPORT_PATH = os.path.join(QC_DIR, "brain_extraction_qc.csv")
TARGET_VOXEL = (1.0, 1.0, 1.0)

BONE_THRESHOLD = 300
EXPECTED_VOLUME_ML = (900, 1900)     # generous range for intracranial cavity content
SYMMETRY_TOLERANCE = 0.20            # allowed left/right volume imbalance (fraction)
BONE_LEAK_TOLERANCE = 0.02           # max allowed fraction of remaining bone voxels


def prepare_raw(nifti_path):
    """Same orientation + resampling used in the extraction step, so the
    raw volume lines up voxel-for-voxel with the extracted one."""
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)
    img = nibproc.resample_to_output(img, voxel_sizes=TARGET_VOXEL, order=3)
    return np.squeeze(img.get_fdata().astype(np.float32))


def compute_mask(extracted_data):
    background = np.min(extracted_data)
    return extracted_data > (background + 1e-3)


def check_volume(mask, voxel_volume_mm3=1.0):
    volume_ml = mask.sum() * voxel_volume_mm3 / 1000.0
    ok = EXPECTED_VOLUME_ML[0] <= volume_ml <= EXPECTED_VOLUME_ML[1]
    return volume_ml, ok


def check_symmetry(mask):
    """Left/right voxel-count balance across the R axis (axis 0 in RAS)."""
    mid = mask.shape[0] // 2
    left = mask[:mid, :, :].sum()
    right = mask[mid:, :, :].sum()
    if max(left, right) == 0:
        return 0.0, False
    ratio = min(left, right) / max(left, right)
    ok = ratio >= (1 - SYMMETRY_TOLERANCE)
    return ratio, ok


def check_bone_leak(extracted_data, mask, threshold=BONE_THRESHOLD):
    if mask.sum() == 0:
        return 1.0, False
    leak_fraction = float((extracted_data[mask] > threshold).mean())
    ok = leak_fraction <= BONE_LEAK_TOLERANCE
    return leak_fraction, ok


def get_best_slice_indices(mask):
    """Pick, for each axis, the slice index with the MOST foreground
    (mask) pixels - not the median coordinate. Median can land in empty
    space once the mask has an irregular shape (e.g. after neck/eye
    cropping), showing a misleading near-empty slice even when the mask
    has plenty of content elsewhere."""
    x, y, z = mask.shape
    if mask.sum() == 0:
        return x // 2, y // 2, z // 2

    x_counts = mask.sum(axis=(1, 2))
    y_counts = mask.sum(axis=(0, 2))
    z_counts = mask.sum(axis=(0, 1))

    best_x = int(np.argmax(x_counts))
    best_y = int(np.argmax(y_counts))
    best_z = int(np.argmax(z_counts))
    return best_x, best_y, best_z


def save_comparison_snapshot(raw_data, extracted_data, mask, subject_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mid_x, mid_y, mid_z = get_best_slice_indices(mask)

    planes = [
        ("Sagittal", raw_data[mid_x, :, :], extracted_data[mid_x, :, :], mask[mid_x, :, :]),
        ("Coronal", raw_data[:, mid_y, :], extracted_data[:, mid_y, :], mask[:, mid_y, :]),
        ("Axial", raw_data[:, :, mid_z], extracted_data[:, :, mid_z], mask[:, :, mid_z]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (label, raw_slice, ext_slice, mask_slice) in enumerate(planes):
        axes[0, col].imshow(np.rot90(raw_slice), cmap="gray", vmin=-100, vmax=700)
        axes[0, col].set_title(f"{label} - original")
        axes[0, col].axis("off")

        axes[1, col].imshow(np.rot90(ext_slice), cmap="gray", vmin=-100, vmax=700)
        if mask_slice.any():
            axes[1, col].contour(np.rot90(mask_slice), colors="red", linewidths=0.8)
        axes[1, col].set_title(f"{label} - extracted (mask outline)")
        axes[1, col].axis("off")

    fig.suptitle(subject_name)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{subject_name}_qc.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def qc_subject(raw_path, extracted_path, out_dir, report_rows):
    filename = os.path.basename(extracted_path)
    subject_name = filename.replace("_extracted.nii.gz", "").replace("_extracted.nii", "")
    print(f"Checking {subject_name} ...")

    raw_data = prepare_raw(raw_path)
    ext_img = nib.load(extracted_path)
    extracted_data = np.squeeze(ext_img.get_fdata().astype(np.float32))
    mask = compute_mask(extracted_data)

    volume_ml, volume_ok = check_volume(mask)
    symmetry_ratio, symmetry_ok = check_symmetry(mask)
    bone_leak, bone_ok = check_bone_leak(extracted_data, mask)

    overall_pass = volume_ok and symmetry_ok and bone_ok

    print(f"  brain volume: {volume_ml:.1f} mL {'OK' if volume_ok else 'CHECK (out of expected range)'}")
    print(f"  L/R symmetry ratio: {symmetry_ratio:.2f} {'OK' if symmetry_ok else 'CHECK (asymmetric)'}")
    print(f"  residual bone fraction: {bone_leak:.4f} {'OK' if bone_ok else 'CHECK (bone remnants)'}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'REVIEW NEEDED'}")

    snap_path = save_comparison_snapshot(raw_data, extracted_data, mask, subject_name, out_dir)
    print(f"  snapshot saved -> {snap_path}")

    report_rows.append({
        "subject": subject_name,
        "brain_volume_ml": round(volume_ml, 1),
        "volume_ok": volume_ok,
        "symmetry_ratio": round(symmetry_ratio, 3),
        "symmetry_ok": symmetry_ok,
        "bone_leak_fraction": round(bone_leak, 4),
        "bone_ok": bone_ok,
        "overall_pass": overall_pass,
        "snapshot": snap_path,
    })


def main():
    os.makedirs(QC_DIR, exist_ok=True)
    extracted_files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, "*_extracted.nii*")))

    if not extracted_files:
        print(f"No extracted files found in {EXTRACTED_DIR}")
        return

    report_rows = []
    for extracted_path in extracted_files:
        filename = os.path.basename(extracted_path)
        subject_name = filename.replace("_extracted.nii.gz", "").replace("_extracted.nii", "")

        matches = glob.glob(os.path.join(RAW_DIR, f"{subject_name}.nii*"))
        if not matches:
            print(f"Skipping {subject_name}: raw file not found in {RAW_DIR}")
            continue

        qc_subject(matches[0], extracted_path, QC_DIR, report_rows)
        print()

    if not report_rows:
        print("No subjects were checked.")
        return

    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    passed = sum(r["overall_pass"] for r in report_rows)
    print(f"\n{passed}/{len(report_rows)} subjects passed all automated checks.")
    print(f"Full report: {REPORT_PATH}")
    print(f"Snapshots for visual check: {QC_DIR}")


if __name__ == "__main__":
    main()