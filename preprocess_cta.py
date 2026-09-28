import os
import glob
import csv
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import exposure

# ---------- CONFIGURE ----------
INPUT_DIR = r"D:\Desktop\no_lvo\brain_extracted"
OUTPUT_DIR = r"D:\Desktop\no_lvo\preprocessed_extracted"
# --------------------------------

REPORT_PATH = os.path.join(OUTPUT_DIR, "motion_artifact_report.csv")

HU_MIN, HU_MAX = -100, 700
MIN_FOREGROUND_PIXELS = 50   # slices with fewer foreground pixels are skipped in motion check


def clip_intensity(data, hu_min=HU_MIN, hu_max=HU_MAX):
    return np.clip(data, hu_min, hu_max)


def normalize_intensity(data, hu_min=HU_MIN, hu_max=HU_MAX):
    return (data - hu_min) / (hu_max - hu_min)


def reduce_noise(data, sigma=0.7):
    return ndimage.gaussian_filter(data, sigma=sigma)


def enhance_contrast(data):
    enhanced = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[2]):
        slice_2d = data[:, :, i]
        if slice_2d.max() > slice_2d.min():
            enhanced[:, :, i] = exposure.equalize_adapthist(slice_2d, clip_limit=0.01)
        else:
            enhanced[:, :, i] = slice_2d
    return enhanced


def detect_motion_artifacts(data, background_mask, blur_threshold=100.0):
    """
    Same Laplacian-variance heuristic as before, but now computed only over
    foreground (brain) pixels per slice. Slices with too little foreground
    (e.g. near the very top/bottom of the extracted brain) are skipped
    instead of being wrongly flagged due to large flat background regions.
    """
    results = []
    flagged_count = 0
    evaluated_count = 0

    for i in range(data.shape[2]):
        slice_2d = data[:, :, i]
        fg_mask = background_mask[:, :, i]

        if fg_mask.sum() < MIN_FOREGROUND_PIXELS:
            results.append((i, None, False, "skipped_low_foreground"))
            continue

        lap = ndimage.laplace(slice_2d)
        lap_var = float(lap[fg_mask].var())
        flagged = lap_var < blur_threshold
        evaluated_count += 1
        if flagged:
            flagged_count += 1
        results.append((i, lap_var, flagged, "evaluated"))

    overall_flag = evaluated_count > 0 and flagged_count > 0.15 * evaluated_count
    return results, overall_flag


def process_file(nifti_path, output_dir, report_rows):
    filename = os.path.basename(nifti_path)
    subject_name = filename.replace("_extracted.nii.gz", "").replace("_extracted.nii", "")
    print(f"Processing {filename} ...")

    img = nib.load(nifti_path)  # already RAS + 1x1x1mm from the extraction step
    data = np.squeeze(img.get_fdata().astype(np.float32))

    background_val = float(np.min(data))
    foreground_mask = data > (background_val + 1e-3)

    data = clip_intensity(data)
    print("  [1/5] intensity clipped to [-100, 700] HU")

    data = normalize_intensity(data)
    print("  [2/5] intensity normalized to [0, 1]")

    data = reduce_noise(data)
    print("  [3/5] noise reduced")

    data = enhance_contrast(data)
    print("  [4/5] contrast enhanced")

    slice_results, overall_flag = detect_motion_artifacts(data, foreground_mask)
    flagged_slices = [str(i) for i, _, flag, status in slice_results if flag]
    print(f"  [5/5] motion artifact check done "
          f"({len(flagged_slices)} slices flagged)")

    report_rows.append({
        "subject": subject_name,
        "num_slices": data.shape[2],
        "flagged_slices_count": len(flagged_slices),
        "flagged_slice_indices": ";".join(flagged_slices),
        "motion_artifact_suspected": overall_flag,
    })

    out_img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    out_path = os.path.join(output_dir, f"{subject_name}_preprocessed.nii.gz")
    nib.save(out_img, out_path)
    print(f"  -> saved {out_path}")
    if overall_flag:
        print(f"  !! motion artifact suspected in {subject_name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # only pick up the extracted brain volumes, not the vessel mask files
    nifti_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_extracted.nii*")))

    if not nifti_files:
        print(f"No extracted NIfTI files found in {INPUT_DIR}")
        return

    report_rows = []
    for nifti_path in nifti_files:
        process_file(nifti_path, OUTPUT_DIR, report_rows)
        print()

    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "num_slices", "flagged_slices_count",
            "flagged_slice_indices", "motion_artifact_suspected"
        ])
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Done. Preprocessed files saved to {OUTPUT_DIR}")
    print(f"Motion artifact report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()