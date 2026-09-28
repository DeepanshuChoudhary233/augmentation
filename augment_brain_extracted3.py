#!/usr/bin/env python3
"""
CTA-Safe Data Augmentation v3 - augment_brain_extracted3.py
============================================================================
Generates 15 augmented versions of each preprocessed CTA volume (a wider
variety of individual and combined transforms than augment_brain_extracted1.py),
plus a contact-sheet QC image (original + all 15 augmentations) per subject.

SAME NON-NEGOTIABLE RULES AS augment_brain_extracted1.py:
  - NO LEFT-RIGHT FLIPPING. Flipping reverses which hemisphere appears
    affected, silently corrupting laterality for LVO/hemisphere-symmetry/
    clot-burden analysis downstream. Only rotation / scaling / translation
    / intensity-based transforms are used.
  - Percentile-based (not min/max) intensity normalization, and
    MEDIAN-based (not mean-based) contrast adjustment - both more robust
    to the handful of very bright/dark outlier voxels a CTA volume often
    has (contrast-filled vessels, bone, artifacts).

TWO SAFETY CHECKS EVERY AUGMENTATION MUST PASS (both were requested
explicitly - "images should not vanish" AND "vessels should not be
drowned into brain tissue intensity"):

  1. NOT VANISHED: augmented nonzero voxel count must stay above
     --min_nonzero_fraction (default 0.85) of the original's nonzero
     count. Catches rotations/scales/shifts that push brain tissue out
     of the field of view.

  2. VESSELS NOT DROWNED IN TISSUE: we approximate "vessel signal" as the
     95th-percentile intensity of foreground voxels (CTA-contrast-filled
     vessels read as the brightest structures in the image) and "tissue
     signal" as the MEDIAN foreground intensity. The gap between them
     (vessel_level - tissue_level) must stay above
     --min_vessel_contrast_fraction (default 0.60) of that same gap in
     the ORIGINAL image. If an augmentation (heavy contrast reduction,
     aggressive gamma, etc.) collapses that gap, vessels become
     indistinguishable from surrounding tissue - exactly what we don't
     want for anything downstream that depends on vessel visibility.

  If either check fails, the augmentation is retried with progressively
  gentler parameters (--max_retries attempts). If it still fails, we fall
  back to a very mild, guaranteed-safe intensity-only variant (small
  percentile normalization only) for that slot, so nothing unusable is
  ever written to disk.

Input : preprocessed_extracted/<subject>.nii(.gz)
Output: augmented3/<subject>/<subject>_aug1.nii.gz ... <subject>_aug15.nii.gz
        augmented3/<subject>/<subject>_augmentation_qc.png   (original + all 15, grid)

Usage:
    python augment_brain_extracted3.py
    python augment_brain_extracted3.py --original_dir preprocessed_extracted --output_dir augmented3
    python augment_brain_extracted3.py --seed 7 --min_vessel_contrast_fraction 0.65
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
DEFAULT_MIN_VESSEL_CONTRAST_FRACTION = 0.60
DEFAULT_MAX_RETRIES = 3
NUM_AUGMENTATIONS = 15


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_volume(path):
    img = nib.load(path)
    return np.squeeze(np.asarray(img.get_fdata(dtype=np.float32))), img.affine, img.header


def get_si_axis(affine):
    codes = aff2axcodes(affine)
    for i, c in enumerate(codes):
        if c in ("S", "I"):
            return i
    return 2


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
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    med = float(np.median(volume[fg_mask]))
    out = volume.copy()
    out[fg_mask] = med + factor * (volume[fg_mask] - med)
    return out.astype(np.float32)


def gamma_adjust(volume, gamma):
    fg_mask = volume != 0
    out = volume.copy()
    if not np.any(fg_mask):
        return out
    vmin, vmax = float(volume[fg_mask].min()), float(volume[fg_mask].max())
    if vmax <= vmin:
        return out
    norm01 = (volume[fg_mask] - vmin) / (vmax - vmin)
    out[fg_mask] = np.power(norm01, gamma) * (vmax - vmin) + vmin
    return out.astype(np.float32)


def small_rotation(volume, angle_deg, axes):
    return rotate(volume, angle=angle_deg, axes=axes, reshape=False,
                  order=1, mode="nearest").astype(np.float32)


def small_scale(volume, factor):
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
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    fg_std = float(volume[fg_mask].std())
    sigma = sigma_fraction * fg_std
    out = volume.copy()
    noise = rng.normal(0, sigma, size=volume.shape).astype(np.float32)
    out[fg_mask] = volume[fg_mask] + noise[fg_mask]
    return out


def brightness_shift(volume, delta_fraction):
    """Shifts foreground intensity up/down by a fraction of the foreground
    std - a plain brightness change, distinct from contrast (spread) or
    gamma (nonlinear curve) adjustments."""
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    fg_std = float(volume[fg_mask].std())
    out = volume.copy()
    out[fg_mask] = volume[fg_mask] + delta_fraction * fg_std
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------------
def nonzero_fraction_ok(aug_volume, orig_nonzero_count, min_fraction):
    if orig_nonzero_count == 0:
        return True
    aug_nonzero = int(np.count_nonzero(aug_volume))
    return aug_nonzero >= min_fraction * orig_nonzero_count


def vessel_contrast_gap(volume):
    """Approximates the 'vessel signal vs tissue signal' gap: 95th
    percentile foreground intensity (bright, contrast-filled vessels)
    minus the MEDIAN foreground intensity (typical brain tissue)."""
    fg = volume[volume != 0]
    if fg.size == 0:
        return 0.0
    vessel_level = float(np.percentile(fg, 95))
    tissue_level = float(np.median(fg))
    return vessel_level - tissue_level


def vessel_contrast_ok(aug_volume, orig_contrast_gap, min_fraction):
    if orig_contrast_gap <= 0:
        return True  # nothing to preserve (degenerate/flat original)
    aug_gap = vessel_contrast_gap(aug_volume)
    return aug_gap >= min_fraction * orig_contrast_gap


# --------------------------------------------------------------------------
# 15 augmentation recipes - a broader variety than augment_brain_extracted1.py
# --------------------------------------------------------------------------
def recipe_percentile_mild(volume, rng, axes, strength=1.0):
    out = percentile_normalize(volume, low_pct=1, high_pct=99)
    return out, {"type": "percentile_norm_mild(1-99pct)"}


def recipe_percentile_strong(volume, rng, axes, strength=1.0):
    lo = 2.0 * strength
    hi = 100 - 2.0 * strength
    out = percentile_normalize(volume, low_pct=lo, high_pct=hi)
    return out, {"type": "percentile_norm_strong", "range": f"{lo:.1f}-{hi:.1f}pct"}


def recipe_contrast_boost(volume, rng, axes, strength=1.0):
    factor = 1.0 + 0.30 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_boost", "factor": round(factor, 3)}


def recipe_contrast_reduce(volume, rng, axes, strength=1.0):
    factor = 1.0 - 0.12 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_reduce", "factor": round(factor, 3)}


def recipe_gamma_brighten(volume, rng, axes, strength=1.0):
    gamma = 1.0 - 0.15 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_brighten", "gamma": round(gamma, 3)}


def recipe_gamma_darken(volume, rng, axes, strength=1.0):
    gamma = 1.0 + 0.15 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_darken", "gamma": round(gamma, 3)}


def recipe_rotation_pos(volume, rng, axes, strength=1.0):
    angle = rng.uniform(3, 7) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_positive", "angle_deg": round(angle, 2)}


def recipe_rotation_neg(volume, rng, axes, strength=1.0):
    angle = -rng.uniform(3, 7) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_negative", "angle_deg": round(angle, 2)}


def recipe_scale_in(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(0.02, 0.05) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_in", "factor": round(factor, 3)}


def recipe_scale_out(volume, rng, axes, strength=1.0):
    factor = 1.0 - rng.uniform(0.02, 0.05) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_out", "factor": round(factor, 3)}


def recipe_translation(volume, rng, axes, strength=1.0):
    shift_vox = [rng.uniform(-2, 2) * strength for _ in range(volume.ndim)]
    out = small_translation(volume, shift_vox)
    return out, {"type": "translation", "shift_vox": [round(s, 2) for s in shift_vox]}


def recipe_mild_noise(volume, rng, axes, strength=1.0):
    out = add_mild_noise(volume, sigma_fraction=0.025 * strength, rng=rng)
    return out, {"type": "mild_noise"}


def recipe_brightness(volume, rng, axes, strength=1.0):
    delta = rng.choice([-1, 1]) * rng.uniform(0.10, 0.20) * strength
    out = brightness_shift(volume, delta)
    return out, {"type": "brightness_shift", "delta_fraction": round(float(delta), 3)}


def recipe_rotation_contrast_combo(volume, rng, axes, strength=1.0):
    angle = rng.uniform(-5, 5) * strength
    factor = 1.0 + rng.uniform(-0.10, 0.15) * strength
    out = small_rotation(volume, angle, axes)
    out = median_contrast_jitter(out, factor)
    return out, {"type": "rotation+contrast_combo", "angle_deg": round(angle, 2), "factor": round(factor, 3)}


def recipe_scale_gamma_combo(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(-0.03, 0.03) * strength
    gamma = 1.0 + rng.uniform(-0.10, 0.10) * strength
    out = small_scale(volume, factor)
    out = gamma_adjust(out, gamma)
    return out, {"type": "scale+gamma_combo", "scale_factor": round(factor, 3), "gamma": round(gamma, 3)}


AUGMENTATION_RECIPES = [
    recipe_percentile_mild,
    recipe_percentile_strong,
    recipe_contrast_boost,
    recipe_contrast_reduce,
    recipe_gamma_brighten,
    recipe_gamma_darken,
    recipe_rotation_pos,
    recipe_rotation_neg,
    recipe_scale_in,
    recipe_scale_out,
    recipe_translation,
    recipe_mild_noise,
    recipe_brightness,
    recipe_rotation_contrast_combo,
    recipe_scale_gamma_combo,
]
assert len(AUGMENTATION_RECIPES) == NUM_AUGMENTATIONS


def generate_one_augmentation(volume, recipe_fn, rng, in_plane_axes, orig_nonzero_count,
                               orig_contrast_gap, min_nonzero_fraction, min_vessel_contrast_fraction,
                               max_retries):
    """Runs recipe_fn, checking BOTH safety conditions each time (not
    vanished, vessels not drowned in tissue) and retrying with a smaller
    `strength` if either fails. Falls back to a very mild, guaranteed-safe
    percentile normalization as a last resort."""
    strength = 1.0
    for attempt in range(max_retries):
        out, meta = recipe_fn(volume, rng, in_plane_axes, strength=strength)
        vanished_ok = nonzero_fraction_ok(out, orig_nonzero_count, min_nonzero_fraction)
        contrast_ok = vessel_contrast_ok(out, orig_contrast_gap, min_vessel_contrast_fraction)
        if vanished_ok and contrast_ok:
            meta["attempt"] = attempt + 1
            meta["fallback_used"] = False
            return out, meta
        strength *= 0.5

    out = percentile_normalize(volume, low_pct=1, high_pct=99)
    return out, {"type": "fallback_mild_percentile_norm", "attempt": max_retries, "fallback_used": True}


# --------------------------------------------------------------------------
# QC contact sheet (4 x 4 grid: original + 15 augmentations)
# --------------------------------------------------------------------------
def save_contact_sheet(subject_id, original, augmented_list, meta_list, si_axis, out_path):
    mid_slice = original.shape[si_axis] // 2
    sl = [slice(None)] * 3
    sl[si_axis] = mid_slice

    images = [original] + augmented_list
    titles = ["Original"] + [f"Aug {i+1}: {m['type']}" for i, m in enumerate(meta_list)]

    n = len(images)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    vmin, vmax = np.percentile(original[original != 0], [1, 99]) if np.any(original != 0) else (0, 1)

    for ax, img, title in zip(axes, images, titles):
        img_slice = np.rot90(img[tuple(sl)])
        ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"{subject_id} - original vs {NUM_AUGMENTATIONS} augmentations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Per-subject pipeline
# --------------------------------------------------------------------------
def process_subject(subject_id, path, output_dir, rng, min_nonzero_fraction,
                     min_vessel_contrast_fraction, max_retries):
    print(f"\n[{subject_id}] loading ...")
    volume, affine, header = load_volume(path)
    si_axis = get_si_axis(affine)
    in_plane_axes = tuple(a for a in range(3) if a != si_axis)
    orig_nonzero_count = int(np.count_nonzero(volume))
    orig_contrast_gap = vessel_contrast_gap(volume)
    print(f"    original vessel-vs-tissue contrast gap = {orig_contrast_gap:.2f}")

    subj_out = os.path.join(output_dir, subject_id)
    os.makedirs(subj_out, exist_ok=True)

    augmented_list = []
    meta_list = []
    for i, recipe_fn in enumerate(AUGMENTATION_RECIPES, start=1):
        out, meta = generate_one_augmentation(
            volume, recipe_fn, rng, in_plane_axes, orig_nonzero_count, orig_contrast_gap,
            min_nonzero_fraction, min_vessel_contrast_fraction, max_retries
        )
        aug_path = os.path.join(subj_out, f"{subject_id}_aug{i}.nii.gz")
        nib.save(nib.Nifti1Image(out, affine, header), aug_path)
        fallback_note = " [FALLBACK USED]" if meta.get("fallback_used") else ""
        print(f"    aug{i:>2d}: {meta['type']:<35s}{fallback_note} -> saved {aug_path}")
        augmented_list.append(out)
        meta_list.append(meta)

    qc_path = os.path.join(subj_out, f"{subject_id}_augmentation_qc.png")
    save_contact_sheet(subject_id, volume, augmented_list, meta_list, si_axis, qc_path)
    print(f"    QC contact sheet -> {qc_path}")


def main():
    parser = argparse.ArgumentParser(description="CTA-safe data augmentation v3 (15 variants, no flipping, "
                                                   "with not-vanished + vessel-not-drowned-in-tissue checks).")
    parser.add_argument("--original_dir", default="preprocessed_extracted")
    parser.add_argument("--output_dir", default="augmented3")
    parser.add_argument("--min_nonzero_fraction", type=float, default=DEFAULT_MIN_NONZERO_FRACTION,
                         help="Reject/retry if nonzero voxel count drops below this fraction of original. "
                              "Default: %(default)s")
    parser.add_argument("--min_vessel_contrast_fraction", type=float, default=DEFAULT_MIN_VESSEL_CONTRAST_FRACTION,
                         help="Reject/retry if the vessel-vs-tissue intensity gap (95th percentile minus "
                              "median, foreground voxels) drops below this fraction of the original's gap. "
                              "Default: %(default)s")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                         help="Retries (with progressively gentler parameters) before falling back to a "
                              "guaranteed-safe mild percentile-normalization variant. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for reproducible augmentations. Default: not fixed.")
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

    print(f"Found {len(input_files)} subject(s). {NUM_AUGMENTATIONS} augmentations each. Output -> {args.output_dir}")
    for subject_id, path in input_files:
        process_subject(subject_id, path, args.output_dir, rng,
                         args.min_nonzero_fraction, args.min_vessel_contrast_fraction, args.max_retries)

    print(f"\nAll done. {len(input_files)} subject(s) x {NUM_AUGMENTATIONS} augmentations written to: {args.output_dir}")


if __name__ == "__main__":
    main()