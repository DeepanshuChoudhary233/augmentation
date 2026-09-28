#!/usr/bin/env python3
"""
Data Augmentation for brain_extracted volumes
==================================================
Generates augmented copies of each skull-stripped brain volume for
deep-learning training, applying:
  - Random Scaling (0.9-1.1x)   - zooms the volume then crops/pads back
                                   to the original array shape (standard
                                   DL-augmentation convention: changes
                                   apparent object size in image space,
                                   NOT a physically-accurate rescan - the
                                   affine/voxel spacing is left unchanged)
  - Gaussian Noise               - adds random noise (in HU) to the brain
                                   foreground only, background left untouched
  - Gamma Adjustment             - nonlinear intensity remapping (contrast
                                   curve change) within the foreground's
                                   own [0,1]-normalized range
  - Brightness                   - random HU offset applied to foreground
  - Contrast                     - scales foreground intensities around
                                   their own mean

Each output volume gets ALL FIVE augmentations applied together with
random parameters (a standard composed-augmentation approach), rather
than five separate single-effect files.

Input : brain_extracted/<subject>_extracted.nii.gz   (raw HU-range volumes;
                                                        vessel_mask files are skipped)
Output: augmented/<subject>_aug1.nii.gz, _aug2.nii.gz, ... (configurable count)
        augmented/qc/<subject>_aug_qc.png   (original vs augmented comparison)

Usage:
    python augment_brain_extracted.py
    python augment_brain_extracted.py --input_dir brain_extracted --output_dir augmented --num_augmentations 3
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- augmentation parameter ranges ----
SCALE_RANGE = (0.9, 1.1)
NOISE_STD_RANGE_HU = (5.0, 20.0)
GAMMA_RANGE = (0.85, 1.15)
BRIGHTNESS_OFFSET_RANGE_HU = (-30.0, 30.0)
CONTRAST_FACTOR_RANGE = (0.8, 1.2)


def load_volume(path):
    img = nib.load(path)
    return np.squeeze(img.get_fdata(dtype=np.float32)), img.affine, img.header


def save_volume(data, affine, header, path):
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine, header), path)
    print(f"    saved -> {path}")


# --------------------------------------------------------------------------
# 1. Random Scaling
# --------------------------------------------------------------------------
def random_scale(data, background_val, scale_range=SCALE_RANGE):
    factor = np.random.uniform(*scale_range)
    zoomed = zoom(data, zoom=factor, order=1, cval=background_val)

    out = np.full_like(data, background_val)
    orig_shape = data.shape
    zoom_shape = zoomed.shape

    # center-crop or pad the zoomed volume back to the original shape
    slices_src, slices_dst = [], []
    for i in range(3):
        if zoom_shape[i] >= orig_shape[i]:
            start = (zoom_shape[i] - orig_shape[i]) // 2
            slices_src.append(slice(start, start + orig_shape[i]))
            slices_dst.append(slice(0, orig_shape[i]))
        else:
            start = (orig_shape[i] - zoom_shape[i]) // 2
            slices_src.append(slice(0, zoom_shape[i]))
            slices_dst.append(slice(start, start + zoom_shape[i]))

    out[tuple(slices_dst)] = zoomed[tuple(slices_src)]
    return out, factor


# --------------------------------------------------------------------------
# 2. Gaussian Noise
# --------------------------------------------------------------------------
def add_gaussian_noise(data, foreground_mask, std_range=NOISE_STD_RANGE_HU):
    std = np.random.uniform(*std_range)
    noise = np.random.normal(0, std, size=data.shape).astype(np.float32)
    out = data.copy()
    out[foreground_mask] += noise[foreground_mask]
    return out, std


# --------------------------------------------------------------------------
# 3. Gamma Adjustment
# --------------------------------------------------------------------------
def apply_gamma(data, foreground_mask, gamma_range=GAMMA_RANGE, hu_ref_min=-100.0, hu_ref_max=700.0):
    """Uses a FIXED HU reference window (-100 to 700, matching the same
    clip range used elsewhere in this pipeline) instead of this image's
    own min/max. A few very bright vessel/calcification voxels (up to
    ~700 HU) would otherwise dominate the normalization, squashing normal
    brain tissue (which sits around 0-80 HU) down to a tiny fraction near
    zero and making it look almost entirely black after gamma."""
    gamma = np.random.uniform(*gamma_range)
    out = data.copy()
    fg = data[foreground_mask]
    if fg.size == 0:
        return out, gamma
    clipped = np.clip(fg, hu_ref_min, hu_ref_max)
    normalized = (clipped - hu_ref_min) / (hu_ref_max - hu_ref_min)
    gamma_applied = np.power(normalized, gamma)
    out[foreground_mask] = gamma_applied * (hu_ref_max - hu_ref_min) + hu_ref_min
    return out, gamma


# --------------------------------------------------------------------------
# 4. Brightness
# --------------------------------------------------------------------------
def apply_brightness(data, foreground_mask, offset_range=BRIGHTNESS_OFFSET_RANGE_HU):
    offset = np.random.uniform(*offset_range)
    out = data.copy()
    out[foreground_mask] += offset
    return out, offset


# --------------------------------------------------------------------------
# 5. Contrast
# --------------------------------------------------------------------------
def apply_contrast(data, foreground_mask, factor_range=CONTRAST_FACTOR_RANGE):
    factor = np.random.uniform(*factor_range)
    out = data.copy()
    fg = data[foreground_mask]
    if fg.size == 0:
        return out, factor
    mean_val = fg.mean()
    out[foreground_mask] = mean_val + (fg - mean_val) * factor
    return out, factor


# --------------------------------------------------------------------------
# Composed augmentation
# --------------------------------------------------------------------------
def augment_volume(data, background_val):
    foreground_mask = data > (background_val + 1e-3)

    data, scale_factor = random_scale(data, background_val)
    foreground_mask = data > (background_val + 1e-3)  # recompute after scaling/crop-pad

    data, noise_std = add_gaussian_noise(data, foreground_mask)
    data, gamma = apply_gamma(data, foreground_mask)
    data, brightness_offset = apply_brightness(data, foreground_mask)
    data, contrast_factor = apply_contrast(data, foreground_mask)

    params = {
        "scale_factor": round(scale_factor, 3),
        "noise_std_hu": round(noise_std, 2),
        "gamma": round(gamma, 3),
        "brightness_offset_hu": round(brightness_offset, 2),
        "contrast_factor": round(contrast_factor, 3),
    }
    return data, params


# --------------------------------------------------------------------------
# QC snapshot
# --------------------------------------------------------------------------
def save_qc_snapshot(original, augmented_list, subject_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mid_z = original.shape[2] // 2

    n_cols = 1 + len(augmented_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(np.rot90(original[:, :, mid_z]), cmap="gray", vmin=-100, vmax=700)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (aug_data, params) in enumerate(augmented_list):
        axes[i + 1].imshow(np.rot90(aug_data[:, :, mid_z]), cmap="gray", vmin=-100, vmax=700)
        title = (f"Aug {i+1}\nscale={params['scale_factor']}, gamma={params['gamma']}\n"
                 f"bright={params['brightness_offset_hu']:.0f}, contrast={params['contrast_factor']}")
        axes[i + 1].set_title(title, fontsize=8)
        axes[i + 1].axis("off")

    fig.suptitle(subject_id)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{subject_id}_aug_qc.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Per-subject pipeline
# --------------------------------------------------------------------------
def process_subject(nifti_path, output_dir, qc_dir, num_augmentations):
    filename = os.path.basename(nifti_path)
    subject_id = filename.replace("_extracted.nii.gz", "").replace("_extracted.nii", "")
    print(f"Processing {filename} ...")

    data, affine, header = load_volume(nifti_path)
    background_val = float(np.min(data))

    augmented_list = []
    for i in range(num_augmentations):
        aug_data, params = augment_volume(data.copy(), background_val)
        out_path = os.path.join(output_dir, f"{subject_id}_aug{i + 1}.nii.gz")
        save_volume(aug_data, affine, header, out_path)
        print(f"    aug{i + 1} params: {params}")
        augmented_list.append((aug_data, params))

    snap_path = save_qc_snapshot(data, augmented_list, subject_id, qc_dir)
    print(f"    QC snapshot -> {snap_path}")


def main():
    parser = argparse.ArgumentParser(description="Augment brain_extracted volumes for DL training")
    parser.add_argument("--input_dir", default="brain_extracted")
    parser.add_argument("--output_dir", default="augmented")
    parser.add_argument("--num_augmentations", type=int, default=3,
                         help="how many augmented copies to generate per subject")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    qc_dir = os.path.join(args.output_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    # only pick up the extracted brain volumes, not vessel mask files
    nifti_files = sorted(glob.glob(os.path.join(args.input_dir, "*_extracted.nii*")))

    if not nifti_files:
        print(f"No '*_extracted.nii*' files found in '{args.input_dir}'.")
        return

    print(f"Found {len(nifti_files)} subject(s). Generating {args.num_augmentations} "
          f"augmented copy/copies each.")
    for nifti_path in nifti_files:
        process_subject(nifti_path, args.output_dir, qc_dir, args.num_augmentations)
        print()

    print(f"Done. Augmented volumes saved to {args.output_dir}")
    print(f"QC snapshots saved to {qc_dir}")


if __name__ == "__main__":
    main()