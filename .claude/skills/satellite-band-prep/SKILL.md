---
name: satellite-band-prep
description: Prepare hyperspectral Rrs for fitting at PACE, MODIS, SeaWiFS, or SBG band centers — interpolate the spectrum to the satellite wavelength grid and generate the appropriate noise variance. Use when the user asks to simulate a satellite spectrum, downsample hyperspectral data to satellite bands, or build the varRrs noise vector for a fit.
---

# Prep an Rrs spectrum for a specific satellite

## When to use this skill

- "Convert this hyperspectral Rrs to PACE bands"
- "Simulate a MODIS spectrum from L23"
- "What's the noise vector for SeaWiFS at these wavelengths?"
- Building inputs for a fit when the data and the model live on different wavelength grids

## Two ingredients

1. **Wavelengths** — the satellite's band centers (or hyperspectral grid).
2. **Variance** — per-band noise variance `varRrs(λ)`, used as the Gaussian likelihood width.

Wavelengths come from `ocpy.satellites.{pace,modis,seawifs,sbg}`. Noise comes from `bing.noise.scale_noise`.

## Wavelength grids per satellite

```python
from ocpy.satellites import pace as sat_pace
from ocpy.satellites import modis as sat_modis
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import sbg as sat_sbg

# Hyperspectral PACE (~5 nm grid). Tunable range.
pace_wave    = sat_pace.wave(wv_min=400., wv_max=700.)

# Multispectral MODIS Aqua
modis_wave   = sat_modis.modis_wave  # ~412, 443, 469, 488, 531, 547, 555, ...

# Historical SeaWiFS
seawifs_wave = sat_seawifs.seawifs_wave  # ~412, 443, 490, 510, 555, 670

# Future SBG (hyperspectral)
sbg_wave     = sat_sbg.gen_noise_vector(np.linspace(380, 1000, 100))  # see below
```

## Interpolate Rrs to a satellite grid

`bing.preproc.convert_to_satwave` does linear interpolation with `'extrapolate'` outside the input range (so be careful at the edges):

```python
from bing.preproc import convert_to_satwave

# in_wave: hyperspectral input grid (e.g., L23's 5-nm Hydrolight grid)
# in_Rrs:  Rrs on in_wave
# model_wave: target satellite grid

model_Rrs = convert_to_satwave(in_wave, in_Rrs, model_wave)
```

For multispectral targets (MODIS, SeaWiFS) you may prefer a band-averaging convolution (with the sensor's RSR). `convert_to_satwave` does **not** do that — it's linear interpolation at the band centers. If accuracy matters near sharp features (chl fluorescence peak), use the sensor RSR; for synthetic L23 work the linear interp is what the rest of the pipeline assumes.

## Noise variance per satellite

`bing.noise.scale_noise(scl_noise, model_Rrs, model_wave)` returns `varRrs` (σ²):

```python
from bing.noise import scale_noise

# Use the satellite's published per-band uncertainty
varRrs = scale_noise('PACE',       model_Rrs, model_wave)
varRrs = scale_noise('MODIS_Aqua', model_Rrs, model_wave)
varRrs = scale_noise('SeaWiFS',    model_Rrs, model_wave)
varRrs = scale_noise('SBG',        model_Rrs, model_wave)

# Or a flat relative noise (e.g., 2% of Rrs)
varRrs = scale_noise(0.02, model_Rrs, model_wave)
```

Internally:
- `'PACE'` and `'SBG'` call `gen_noise_vector(model_wave)` from the respective satellite module.
- `'MODIS_Aqua'` / `'SeaWiFS'` use `calc_errors()` with per-band σ at fixed band centers — they only work when `model_wave == modis_wave` or `seawifs_wave`.
- A float `scl_noise` means `σ = scl_noise · Rrs` → `varRrs = (scl_noise · Rrs)²`.

## Adding synthetic noise to a clean spectrum

If you want noisy observations (e.g., to build a Monte Carlo):

```python
from bing.noise import add_noise

# Per-wavelength absolute σ
sig = np.sqrt(varRrs)
noisy_Rrs = add_noise(model_Rrs, abs_sig=sig)

# Or single percentage applied uniformly
noisy_Rrs = add_noise(model_Rrs, perc=2)  # 2%

# Correlated noise (4-wavelength tridiagonal-ish covariance)
noisy_Rrs = add_noise(np.atleast_2d(model_Rrs), abs_sig=sig, correlate=True)
```

The noise draw is **truncated to ±3σ** to avoid extreme outliers.

## End-to-end satellite-prep example

```python
import numpy as np
from ocpy.satellites import pace as sat_pace
from bing.preproc import convert_to_satwave
from bing.noise   import scale_noise, add_noise

# Hyperspectral input (e.g., from L23 or a Hydrolight sim)
in_wave  = np.arange(400, 701, 5.0)
in_Rrs   = ...   # shape (nwave,) sr^-1

# Target grid: PACE hyperspectral 400-700 nm
model_wave = sat_pace.wave(wv_min=400., wv_max=700.)

# Interpolate
model_Rrs = convert_to_satwave(in_wave, in_Rrs, model_wave)

# Per-band variance
model_varRrs = scale_noise('PACE', model_Rrs, model_wave)

# Optional: corrupt with random noise to simulate a real observation
obs_Rrs = add_noise(model_Rrs, abs_sig=np.sqrt(model_varRrs))

# Now feed (obs_Rrs, model_varRrs, model_wave) to the fit (see run-bing-fit).
```

## When the model wavelength grid must match

The same `model_wave` is used to:
- Initialize the bio-optical models: `model_utils.init(p.model_names, model_wave)`
- Compute Raman excitation wavelengths (`models[*].wave_ex` is offset from `model_wave`)
- Generate the satellite noise vector
- Sample wavelength-dependent G₁(λ), G₂(λ) for variable Gordon

If you use different grids in any of these places, the fit silently extrapolates and biases creep in.

## UV cutoff convention

`p.wv_min` / `p.wv_max` clip both the data and the model grid. Common choices:

| Setting | Behavior |
|---|---|
| `wv_min=None` | Use the full grid (down to 400 nm typically) |
| `wv_min=350` | Include UV (only PACE/SBG have this) |
| `wv_min=400` | Standard visible-only fit |
| `wv_max=700` | Standard cutoff before strong water absorption |
| `wv_max=750` | Include the chlorophyll fluorescence peak (685 nm) |

The L23 prep wrapper (`l23.prep_one_l23`) applies these cuts via `gd_wave` masking before interpolation.

## Common pitfalls

- **Linear interpolation across the 685 nm fluorescence peak** flattens it. Use a sensor-RSR convolution or fit fluorescence explicitly ([inelastic-rrs](../inelastic-rrs/SKILL.md)).
- **`scale_noise('MODIS_Aqua', ..., model_wave)`** assumes `model_wave` is exactly `sat_modis.modis_wave`. If you've interpolated to a different grid first, this lookup silently misaligns.
- **Extrapolating Rrs below 380 nm or above 800 nm** with `convert_to_satwave` produces unphysical numbers — `fill_value='extrapolate'` just runs the line off the end. Clip to a safe range first.
- **Forgetting `add_noise=False`** in synthetic experiments while still using `scl_noise='PACE'` → `varRrs` is set as if you had noise, but `Rrs` is the clean spectrum, so χ²ν will be artificially small.
- **Mixing `model_wave` from PACE with `gen_noise_vector` on a different grid** → silent broadcasting against the wrong axis.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — consumer of the prepped data
- [fit-l23-spectrum](../fit-l23-spectrum/SKILL.md) — full pipeline that does this internally
- [inelastic-rrs](../inelastic-rrs/SKILL.md) — what to do near 685 nm
