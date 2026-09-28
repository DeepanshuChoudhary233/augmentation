#!/usr/bin/env python3
"""
CTA-Safe Data Augmentation v4 - augment_brain_extracted4.py
============================================================================
Generates 35 augmented versions of each preprocessed CTA volume - a wider
and more varied set of recipes than augment_brain_extracted3.py's 15,
including three NEW primitives not used in v3 (Gaussian blur, unsharp-mask
sharpening, and a smooth multiplicative bias-field - all common CT/MRI
augmentations), plus a contact-sheet QC image (original + all 35) per
subject.

SAME NON-NEGOTIABLE RULES AS v1/v3:
  - NO LEFT-RIGHT FLIPPING. Flipping reverses which hemisphere appears
    affected, silently corrupting laterality for LVO/hemisphere-symmetry/
    clot-burden analysis downstream. Only rotation / scaling / translation
    / intensity-based transforms are used.
  - Percentile-based (not min/max) intensity normalization, and
    MEDIAN-based (not mean-based) contrast adjustment.

SAME TWO SAFETY CHECKS AS v3 (see that script's docstring for full
rationale) - every augmentation must pass both, with gentler-parameter
retries and a guaranteed-safe fallback if it still fails:
  1. NOT VANISHED - nonzero voxel count stays above --min_nonzero_fraction
     of the original.
  2. VESSELS NOT DROWNED IN TISSUE - the 95th-percentile-minus-median
     foreground intensity gap stays above --min_vessel_contrast_fraction
     of the original's gap.

NEW PRIMITIVES IN v4 (not in v1/v3):
  - Gaussian blur: mild smoothing, simulates scanner/reconstruction
    softness or motion blur. Applied then re-masked to the original
    foreground so the background stays exactly zero.
  - Unsharp-mask sharpening: volume + amount*(volume - blurred(volume)),
    simulates a sharper reconstruction kernel.
  - Bias field: a smooth, low-frequency random multiplicative field
    (built from a small random grid upsampled to the volume's shape),
    simulating the gradual intensity drift/shading real CT/MRI scanners
    can introduce across the field of view.

Input : preprocessed_extracted/<subject>.nii(.gz)
Output: augmented4/<subject>/<subject>_aug1.nii.gz ... <subject>_aug35.nii.gz
        augmented4/<subject>/<subject>_augmentation_qc.png   (original + all 35, grid)

Usage:
    python augment_brain_extracted4.py
    python augment_brain_extracted4.py --original_dir preprocessed_extracted --output_dir augmented4
    python augment_brain_extracted4.py --seed 7 --min_vessel_contrast_fraction 0.65
"""

import os
import argparse
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import rotate, zoom, shift as nd_shift, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MIN_NONZERO_FRACTION = 0.85
DEFAULT_MIN_VESSEL_CONTRAST_FRACTION = 0.60
DEFAULT_MAX_RETRIES = 3
NUM_AUGMENTATIONS = 35


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
# Safe, non-flip augmentation primitives (shared with v1/v3)
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
    fg_mask = volume != 0
    if not np.any(fg_mask):
        return volume.copy()
    fg_std = float(volume[fg_mask].std())
    out = volume.copy()
    out[fg_mask] = volume[fg_mask] + delta_fraction * fg_std
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# NEW primitives for v4: blur, sharpen, bias field
# --------------------------------------------------------------------------
def gaussian_blur(volume, sigma):
    """Mild smoothing (scanner/reconstruction softness or motion blur).
    Blurred then re-masked to the ORIGINAL foreground so the background
    stays exactly zero (blur would otherwise smear signal into it)."""
    fg_mask = volume != 0
    blurred = gaussian_filter(volume, sigma=sigma)
    out = np.where(fg_mask, blurred, 0.0)
    return out.astype(np.float32)


def unsharp_sharpen(volume, sigma, amount):
    """Unsharp-mask sharpening: out = volume + amount*(volume - blurred),
    simulating a sharper reconstruction kernel. Also re-masked to the
    original foreground."""
    fg_mask = volume != 0
    blurred = gaussian_filter(volume, sigma=sigma)
    sharpened = volume + amount * (volume - blurred)
    out = np.where(fg_mask, sharpened, 0.0)
    return out.astype(np.float32)


def bias_field(volume, rng, grid_size=4, strength=0.15):
    """Simulates the smooth, low-frequency intensity drift/shading real
    CT/MRI scanners can introduce: builds a coarse random grid (values
    around 1.0 +/- strength), upsamples it smoothly to the volume's full
    shape via zoom (order=3, cubic - gives a smooth field, not blocky),
    and multiplies it into the foreground only."""
    fg_mask = volume != 0
    coarse = rng.uniform(1.0 - strength, 1.0 + strength, size=(grid_size,) * volume.ndim).astype(np.float32)
    zoom_factors = [s / g for s, g in zip(volume.shape, coarse.shape)]
    field = zoom(coarse, zoom_factors, order=3)
    field = field[tuple(slice(0, s) for s in volume.shape)]  # guard against off-by-one from zoom rounding
    out = volume.copy()
    out[fg_mask] = volume[fg_mask] * field[fg_mask]
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Safety checks (identical to v3)
# --------------------------------------------------------------------------
def nonzero_fraction_ok(aug_volume, orig_nonzero_count, min_fraction):
    if orig_nonzero_count == 0:
        return True
    aug_nonzero = int(np.count_nonzero(aug_volume))
    return aug_nonzero >= min_fraction * orig_nonzero_count


def vessel_contrast_gap(volume):
    fg = volume[volume != 0]
    if fg.size == 0:
        return 0.0
    vessel_level = float(np.percentile(fg, 95))
    tissue_level = float(np.median(fg))
    return vessel_level - tissue_level


def vessel_contrast_ok(aug_volume, orig_contrast_gap, min_fraction):
    if orig_contrast_gap <= 0:
        return True
    aug_gap = vessel_contrast_gap(aug_volume)
    return aug_gap >= min_fraction * orig_contrast_gap


# --------------------------------------------------------------------------
# 35 augmentation recipes
# --------------------------------------------------------------------------
def recipe_percentile_mild(volume, rng, axes, strength=1.0):
    out = percentile_normalize(volume, low_pct=0.5, high_pct=99.5)
    return out, {"type": "percentile_norm_mild(0.5-99.5pct)"}


def recipe_percentile_moderate(volume, rng, axes, strength=1.0):
    out = percentile_normalize(volume, low_pct=2, high_pct=98)
    return out, {"type": "percentile_norm_moderate(2-98pct)"}


def recipe_percentile_strong(volume, rng, axes, strength=1.0):
    lo = 3.0 * strength
    hi = 100 - 3.0 * strength
    out = percentile_normalize(volume, low_pct=lo, high_pct=hi)
    return out, {"type": "percentile_norm_strong", "range": f"{lo:.1f}-{hi:.1f}pct"}


def recipe_contrast_boost_mild(volume, rng, axes, strength=1.0):
    factor = 1.0 + 0.15 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_boost_mild", "factor": round(factor, 3)}


def recipe_contrast_boost_strong(volume, rng, axes, strength=1.0):
    factor = 1.0 + 0.35 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_boost_strong", "factor": round(factor, 3)}


def recipe_contrast_reduce_mild(volume, rng, axes, strength=1.0):
    factor = 1.0 - 0.08 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_reduce_mild", "factor": round(factor, 3)}


def recipe_contrast_reduce_strong(volume, rng, axes, strength=1.0):
    factor = 1.0 - 0.15 * strength
    out = median_contrast_jitter(volume, factor)
    return out, {"type": "median_contrast_reduce_strong", "factor": round(factor, 3)}


def recipe_gamma_brighten_mild(volume, rng, axes, strength=1.0):
    gamma = 1.0 - 0.08 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_brighten_mild", "gamma": round(gamma, 3)}


def recipe_gamma_brighten_strong(volume, rng, axes, strength=1.0):
    gamma = 1.0 - 0.18 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_brighten_strong", "gamma": round(gamma, 3)}


def recipe_gamma_darken_mild(volume, rng, axes, strength=1.0):
    gamma = 1.0 + 0.08 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_darken_mild", "gamma": round(gamma, 3)}


def recipe_gamma_darken_strong(volume, rng, axes, strength=1.0):
    gamma = 1.0 + 0.18 * strength
    out = gamma_adjust(volume, gamma)
    return out, {"type": "gamma_darken_strong", "gamma": round(gamma, 3)}


def recipe_rotation_pos_small(volume, rng, axes, strength=1.0):
    angle = rng.uniform(2, 4) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_positive_small", "angle_deg": round(angle, 2)}


def recipe_rotation_pos_large(volume, rng, axes, strength=1.0):
    angle = rng.uniform(5, 8) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_positive_large", "angle_deg": round(angle, 2)}


def recipe_rotation_neg_small(volume, rng, axes, strength=1.0):
    angle = -rng.uniform(2, 4) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_negative_small", "angle_deg": round(angle, 2)}


def recipe_rotation_neg_large(volume, rng, axes, strength=1.0):
    angle = -rng.uniform(5, 8) * strength
    out = small_rotation(volume, angle, axes)
    return out, {"type": "rotation_negative_large", "angle_deg": round(angle, 2)}


def recipe_scale_in_small(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(0.01, 0.03) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_in_small", "factor": round(factor, 3)}


def recipe_scale_in_large(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(0.04, 0.06) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_in_large", "factor": round(factor, 3)}


def recipe_scale_out_small(volume, rng, axes, strength=1.0):
    factor = 1.0 - rng.uniform(0.01, 0.03) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_out_small", "factor": round(factor, 3)}


def recipe_scale_out_large(volume, rng, axes, strength=1.0):
    factor = 1.0 - rng.uniform(0.04, 0.06) * strength
    out = small_scale(volume, factor)
    return out, {"type": "scale_zoom_out_large", "factor": round(factor, 3)}


def recipe_translation_small(volume, rng, axes, strength=1.0):
    shift_vox = [rng.uniform(-1.5, 1.5) * strength for _ in range(volume.ndim)]
    out = small_translation(volume, shift_vox)
    return out, {"type": "translation_small", "shift_vox": [round(s, 2) for s in shift_vox]}


def recipe_translation_large(volume, rng, axes, strength=1.0):
    shift_vox = [rng.uniform(-3, 3) * strength for _ in range(volume.ndim)]
    out = small_translation(volume, shift_vox)
    return out, {"type": "translation_large", "shift_vox": [round(s, 2) for s in shift_vox]}


def recipe_mild_noise(volume, rng, axes, strength=1.0):
    out = add_mild_noise(volume, sigma_fraction=0.02 * strength, rng=rng)
    return out, {"type": "mild_noise"}


def recipe_stronger_noise(volume, rng, axes, strength=1.0):
    out = add_mild_noise(volume, sigma_fraction=0.045 * strength, rng=rng)
    return out, {"type": "stronger_noise"}


def recipe_brightness_up(volume, rng, axes, strength=1.0):
    delta = rng.uniform(0.10, 0.18) * strength
    out = brightness_shift(volume, delta)
    return out, {"type": "brightness_up", "delta_fraction": round(float(delta), 3)}


def recipe_brightness_down(volume, rng, axes, strength=1.0):
    delta = -rng.uniform(0.10, 0.18) * strength
    out = brightness_shift(volume, delta)
    return out, {"type": "brightness_down", "delta_fraction": round(float(delta), 3)}


def recipe_blur_mild(volume, rng, axes, strength=1.0):
    sigma = 0.5 * strength
    out = gaussian_blur(volume, sigma)
    return out, {"type": "gaussian_blur_mild", "sigma": round(sigma, 3)}


def recipe_blur_strong(volume, rng, axes, strength=1.0):
    sigma = 1.0 * strength
    out = gaussian_blur(volume, sigma)
    return out, {"type": "gaussian_blur_strong", "sigma": round(sigma, 3)}


def recipe_sharpen(volume, rng, axes, strength=1.0):
    sigma, amount = 1.0, 0.6 * strength
    out = unsharp_sharpen(volume, sigma, amount)
    return out, {"type": "unsharp_sharpen", "sigma": sigma, "amount": round(amount, 3)}


def recipe_bias_field(volume, rng, axes, strength=1.0):
    out = bias_field(volume, rng, grid_size=4, strength=0.15 * strength)
    return out, {"type": "bias_field", "strength": round(0.15 * strength, 3)}


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


def recipe_translation_contrast_combo(volume, rng, axes, strength=1.0):
    shift_vox = [rng.uniform(-2, 2) * strength for _ in range(volume.ndim)]
    factor = 1.0 + rng.uniform(-0.08, 0.12) * strength
    out = small_translation(volume, shift_vox)
    out = median_contrast_jitter(out, factor)
    return out, {"type": "translation+contrast_combo", "shift_vox": [round(s, 2) for s in shift_vox],
                 "factor": round(factor, 3)}


def recipe_rotation_scale_combo(volume, rng, axes, strength=1.0):
    angle = rng.uniform(-4, 4) * strength
    factor = 1.0 + rng.uniform(-0.03, 0.03) * strength
    out = small_rotation(volume, angle, axes)
    out = small_scale(out, factor)
    return out, {"type": "rotation+scale_combo", "angle_deg": round(angle, 2), "scale_factor": round(factor, 3)}


def recipe_rotation_translation_combo(volume, rng, axes, strength=1.0):
    angle = rng.uniform(-4, 4) * strength
    shift_vox = [rng.uniform(-2, 2) * strength for _ in range(volume.ndim)]
    out = small_rotation(volume, angle, axes)
    out = small_translation(out, shift_vox)
    return out, {"type": "rotation+translation_combo", "angle_deg": round(angle, 2),
                 "shift_vox": [round(s, 2) for s in shift_vox]}


def recipe_scale_noise_combo(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(-0.03, 0.03) * strength
    out = small_scale(volume, factor)
    out = add_mild_noise(out, sigma_fraction=0.02 * strength, rng=rng)
    return out, {"type": "scale+noise_combo", "scale_factor": round(factor, 3)}


def recipe_gamma_noise_combo(volume, rng, axes, strength=1.0):
    gamma = 1.0 + rng.uniform(-0.10, 0.10) * strength
    out = gamma_adjust(volume, gamma)
    out = add_mild_noise(out, sigma_fraction=0.02 * strength, rng=rng)
    return out, {"type": "gamma+noise_combo", "gamma": round(gamma, 3)}


def recipe_contrast_brightness_combo(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(-0.10, 0.15) * strength
    delta = rng.uniform(-0.12, 0.12) * strength
    out = median_contrast_jitter(volume, factor)
    out = brightness_shift(out, delta)
    return out, {"type": "contrast+brightness_combo", "factor": round(factor, 3),
                 "delta_fraction": round(float(delta), 3)}


def recipe_triple_rotation_gamma_contrast(volume, rng, axes, strength=1.0):
    angle = rng.uniform(-4, 4) * strength
    gamma = 1.0 + rng.uniform(-0.08, 0.08) * strength
    factor = 1.0 + rng.uniform(-0.08, 0.10) * strength
    out = small_rotation(volume, angle, axes)
    out = gamma_adjust(out, gamma)
    out = median_contrast_jitter(out, factor)
    return out, {"type": "triple_rotation+gamma+contrast", "angle_deg": round(angle, 2),
                 "gamma": round(gamma, 3), "factor": round(factor, 3)}


def recipe_triple_scale_rotation_translation(volume, rng, axes, strength=1.0):
    factor = 1.0 + rng.uniform(-0.02, 0.02) * strength
    angle = rng.uniform(-3, 3) * strength
    shift_vox = [rng.uniform(-1.5, 1.5) * strength for _ in range(volume.ndim)]
    out = small_scale(volume, factor)
    out = small_rotation(out, angle, axes)
    out = small_translation(out, shift_vox)
    return out, {"type": "triple_scale+rotation+translation", "scale_factor": round(factor, 3),
                 "angle_deg": round(angle, 2), "shift_vox": [round(s, 2) for s in shift_vox]}


def recipe_blur_contrast_combo(volume, rng, axes, strength=1.0):
    sigma = 0.6 * strength
    factor = 1.0 + rng.uniform(0.05, 0.15) * strength
    out = gaussian_blur(volume, sigma)
    out = median_contrast_jitter(out, factor)
    return out, {"type": "blur+contrast_combo", "sigma": round(sigma, 3), "factor": round(factor, 3)}


def recipe_sharpen_noise_combo(volume, rng, axes, strength=1.0):
    sigma, amount = 1.0, 0.5 * strength
    out = unsharp_sharpen(volume, sigma, amount)
    out = add_mild_noise(out, sigma_fraction=0.015 * strength, rng=rng)
    return out, {"type": "sharpen+noise_combo", "amount": round(amount, 3)}


def recipe_bias_field_contrast_combo(volume, rng, axes, strength=1.0):
    out = bias_field(volume, rng, grid_size=4, strength=0.12 * strength)
    factor = 1.0 + rng.uniform(-0.08, 0.08) * strength
    out = median_contrast_jitter(out, factor)
    return out, {"type": "bias_field+contrast_combo", "factor": round(factor, 3)}


def recipe_bias_field_rotation_combo(volume, rng, axes, strength=1.0):
    angle = rng.uniform(-3, 3) * strength
    out = small_rotation(volume, angle, axes)
    out = bias_field(out, rng, grid_size=4, strength=0.12 * strength)
    return out, {"type": "bias_field+rotation_combo", "angle_deg": round(angle, 2)}


AUGMENTATION_RECIPES = [
    recipe_percentile_mild,
    recipe_percentile_moderate,
    recipe_percentile_strong,
    recipe_contrast_boost_mild,
    recipe_contrast_boost_strong,
    recipe_contrast_reduce_mild,
    recipe_contrast_reduce_strong,
    recipe_gamma_brighten_mild,
    recipe_gamma_brighten_strong,
    recipe_gamma_darken_mild,
    recipe_gamma_darken_strong,
    recipe_rotation_pos_small,
    recipe_rotation_pos_large,
    recipe_rotation_neg_small,
    recipe_rotation_neg_large,
    recipe_scale_in_small,
    recipe_scale_in_large,
    recipe_scale_out_small,
    recipe_scale_out_large,
    recipe_translation_small,
    recipe_translation_large,
    recipe_mild_noise,
    recipe_stronger_noise,
    recipe_brightness_up,
    recipe_brightness_down,
    recipe_blur_mild,
    recipe_blur_strong,
    recipe_sharpen,
    recipe_bias_field,
    recipe_rotation_contrast_combo,
    recipe_scale_gamma_combo,
    recipe_triple_rotation_gamma_contrast,
    recipe_triple_scale_rotation_translation,
    recipe_blur_contrast_combo,
    recipe_bias_field_rotation_combo,
]
assert len(AUGMENTATION_RECIPES) == NUM_AUGMENTATIONS, \
    f"expected {NUM_AUGMENTATIONS} recipes, have {len(AUGMENTATION_RECIPES)}"


def generate_one_augmentation(volume, recipe_fn, rng, in_plane_axes, orig_nonzero_count,
                               orig_contrast_gap, min_nonzero_fraction, min_vessel_contrast_fraction,
                               max_retries):
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
# QC contact sheet (6 x 6 grid: original + 35 augmentations = 36 slots)
# --------------------------------------------------------------------------
def save_contact_sheet(subject_id, original, augmented_list, meta_list, si_axis, out_path):
    mid_slice = original.shape[si_axis] // 2
    sl = [slice(None)] * 3
    sl[si_axis] = mid_slice

    images = [original] + augmented_list
    titles = ["Original"] + [f"Aug {i+1}: {m['type']}" for i, m in enumerate(meta_list)]

    n = len(images)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.array(axes).flatten()

    vmin, vmax = np.percentile(original[original != 0], [1, 99]) if np.any(original != 0) else (0, 1)

    for ax, img, title in zip(axes, images, titles):
        img_slice = np.rot90(img[tuple(sl)])
        ax.imshow(img_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=6.5)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"{subject_id} - original vs {NUM_AUGMENTATIONS} augmentations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
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
    parser = argparse.ArgumentParser(description="CTA-safe data augmentation v4 (35 variants, no flipping, "
                                                   "with not-vanished + vessel-not-drowned-in-tissue checks).")
    parser.add_argument("--original_dir", default="preprocessed_extracted")
    parser.add_argument("--output_dir", default="augmented4")
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