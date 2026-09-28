# Here we will verify whether the preprocessing steps are correctly applied to the input data. This includes checking for missing values, ensuring that categorical variables are properly encoded, and confirming that numerical features are scaled appropriately. We will also validate that the data types of each feature match the expected types defined in our schema. 
import os
import glob
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

# ---------- CONFIGURE ----------
PREPROCESSED_DIR = r"D:\Desktop\no_lvo\preprocessed_extracted"
QC_OUTPUT_DIR = r"D:\Desktop\no_lvo\qc_check"
# --------------------------------

EXPECTED_VOXEL = (1.0, 1.0, 1.0)
VOXEL_TOLERANCE = 0.05          # mm
EXPECTED_RANGE = (0.0, 1.05)    # allow tiny overshoot from smoothing/CLAHE
REPORT_PATH = os.path.join(QC_OUTPUT_DIR, "qc_summary.csv")


def check_orientation(img):
    ornt = nib.aff2axcodes(img.affine)
    is_ras = ornt == ("R", "A", "S")
    return "".join(ornt), is_ras


def check_voxel_size(img):
    voxel_sizes = img.header.get_zooms()[:3]
    ok = all(abs(v - t) <= VOXEL_TOLERANCE for v, t in zip(voxel_sizes, EXPECTED_VOXEL))
    return voxel_sizes, ok


def check_intensity_range(data):
    dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
    ok = (dmin >= EXPECTED_RANGE[0] - 0.05) and (dmax <= EXPECTED_RANGE[1])
    return dmin, dmax, ok


def check_nan_inf(data):
    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())
    return has_nan, has_inf


def save_snapshot(data, subject_name, out_dir):
    """Save mid-slice snapshots in all three planes for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    x, y, z = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(np.rot90(data[x // 2, :, :]), cmap="gray")
    axes[0].set_title("Sagittal (mid)")
    axes[1].imshow(np.rot90(data[:, y // 2, :]), cmap="gray")
    axes[1].set_title("Coronal (mid)")
    axes[2].imshow(np.rot90(data[:, :, z // 2]), cmap="gray")
    axes[2].set_title("Axial (mid)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(subject_name)
    out_path = os.path.join(out_dir, f"{subject_name}_snapshot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def qc_file(nifti_path, snapshot_dir, report_rows):
    filename = os.path.basename(nifti_path)
    subject_name = filename.replace(".nii.gz", "").replace(".nii", "")
    print(f"Checking {filename} ...")

    img = nib.load(nifti_path)
    data = np.squeeze(img.get_fdata())

    orient_str, orient_ok = check_orientation(img)
    voxel_sizes, voxel_ok = check_voxel_size(img)
    dmin, dmax, range_ok = check_intensity_range(data)
    has_nan, has_inf = check_nan_inf(data)

    all_ok = orient_ok and voxel_ok and range_ok and not has_nan and not has_inf

    print(f"  orientation: {orient_str} {'OK' if orient_ok else 'FAIL (expected RAS)'}")
    print(f"  voxel size: {tuple(round(v, 3) for v in voxel_sizes)} "
          f"{'OK' if voxel_ok else 'FAIL (expected ~1x1x1mm)'}")
    print(f"  intensity range: [{dmin:.3f}, {dmax:.3f}] "
          f"{'OK' if range_ok else 'FAIL (expected ~[0,1])'}")
    print(f"  NaN present: {has_nan} | Inf present: {has_inf}")
    print(f"  shape: {data.shape}")
    print(f"  OVERALL: {'PASS' if all_ok else 'CHECK NEEDED'}")

    snap_path = save_snapshot(data, subject_name, snapshot_dir)
    print(f"  snapshot saved -> {snap_path}")

    report_rows.append({
        "subject": subject_name,
        "shape": str(data.shape),
        "orientation": orient_str,
        "orientation_ok": orient_ok,
        "voxel_size": str(tuple(round(v, 3) for v in voxel_sizes)),
        "voxel_ok": voxel_ok,
        "intensity_min": round(dmin, 4),
        "intensity_max": round(dmax, 4),
        "range_ok": range_ok,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "overall_pass": all_ok,
        "snapshot": snap_path,
    })


def main():
    os.makedirs(QC_OUTPUT_DIR, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(PREPROCESSED_DIR, "*.nii"))
        + glob.glob(os.path.join(PREPROCESSED_DIR, "*.nii.gz"))
    )

    if not files:
        print(f"No preprocessed files found in {PREPROCESSED_DIR}")
        return

    report_rows = []
    for f in files:
        qc_file(f, QC_OUTPUT_DIR, report_rows)
        print()

    fieldnames = list(report_rows[0].keys())
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    passed = sum(r["overall_pass"] for r in report_rows)
    print(f"\n{passed}/{len(report_rows)} files passed all automated checks.")
    print(f"Full report: {REPORT_PATH}")
    print(f"Snapshots for visual check: {QC_OUTPUT_DIR}")


if __name__ == "__main__":
    main()