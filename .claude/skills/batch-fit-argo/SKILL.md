---
name: batch-fit-argo
description: Run parallel MCMC batch fits over matched Argo BGC float profiles using inference.fit_batch, with checkpointing and standard output NPZ layout. Use when the user asks to fit all Argo profiles, run the biomass pipeline, or process matched satellite-float data in parallel.
---

# Parallel batch fitting for Argo BGC matchups

## When to use this skill

- "Fit all Argo profiles in the matchup table"
- "Run the biomass pipeline"
- "Process the matched satellite/float dataset"
- Anywhere you have N spectra you want fitted in parallel with `inference.fit_batch`

## Inputs

A pandas DataFrame (or CSV) of matched profiles with one row per profile. The biomass paper uses something like `matched_argo_bgc_profiles_bbp.csv` in [papers/biomass/Analysis/](../../../papers/biomass/Analysis/). Required columns vary, but the per-row pattern is always:

| Column | Meaning |
|---|---|
| `wave_*` or per-wavelength columns | Hyperspectral PACE Rrs |
| `Chl` (optional) | Surface chlorophyll for Bricaud models |
| `cruise`, `profile`, `lat`, `lon`, `time` | Metadata for filename / provenance |

## Pattern (one process, parallel inside `fit_batch`)

```python
import os
import numpy as np
import pandas as pd

from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf
from bing.rt import defs as rt_defs
from bing.fitting.l23 import save_chains, chain_filename

# ---------- 1. Configuration ----------
p = standard.expb_pow(
    satellite='PACE',
    nsteps=20000,
    nburn=2000,
    scl_noise='PACE',
    include_Raman=False,
    variable_Gordon=False,
)
rt_dict = rt_defs.rt_dict_from_p(p)

matched = pd.read_csv('papers/biomass/Analysis/matched_argo_bgc_profiles_bbp.csv')
out_dir = 'papers/biomass/Analysis/Fits/'
os.makedirs(out_dir, exist_ok=True)

# ---------- 2. Build a wavelength-aligned input array ----------
# Adjust depending on your CSV schema:
wave = ...                                    # shape (nwave,)
Rrs    = matched[[c for c in matched.columns if c.startswith('Rrs_')]].values
sigRrs = matched[[c for c in matched.columns if c.startswith('sigRrs_')]].values
Chls   = matched['Chl'].values                # shape (n_spectra,)
Ys     = matched.get('Y', pd.Series(np.full(len(matched), 1.0))).values

# ---------- 3. Models share the same grid for the whole batch ----------
models = model_utils.init(p.model_names, wave)
bing_priors.set_standard_priors(models, p)

# ---------- 4. Skip already-fit profiles (resumable) ----------
to_fit = []
for i, row in matched.iterrows():
    outfile = os.path.join(out_dir,
                           f"Argo_{row.cruise}_{int(row.profile):03d}.npz")
    if os.path.exists(outfile):
        continue
    to_fit.append(i)
print(f"Fitting {len(to_fit)} of {len(matched)} profiles")

# ---------- 5. Build items + initial guesses ----------
pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)
pdict['Chl'] = np.zeros(len(matched))
pdict['Y']   = np.zeros(len(matched))
pdict['Chl'][to_fit] = Chls[to_fit]
pdict['Y'][to_fit]   = Ys[to_fit]

items = []
for i in to_fit:
    # Crude initial guess; refine if you keep getting -inf
    p0_a = models[0].init_guess(Rrs[i] * 5.0)
    p0_b = models[1].init_guess(Rrs[i] * 0.1)
    p0 = np.concatenate([np.atleast_1d(p0_a), np.atleast_1d(p0_b)])
    # log10 the parameters whose prior is log_*
    j = 0
    for ss in [0, 1]:
        for prior in models[ss].priors.priors:
            if prior.flavor.startswith('log'):
                p0[j] = np.log10(max(p0[j], 1e-6))
            j += 1
    items.append((Rrs[i], sigRrs[i]**2, p0, i))

# ---------- 6. Parallel fit ----------
all_samples, all_idx = bing_inf.fit_batch(
    models, pdict, items, rt_dict, n_cores=10,
)

# ---------- 7. Save per-profile NPZ ----------
for sample, idx in zip(all_samples, all_idx):
    row = matched.iloc[idx]
    outfile = os.path.join(out_dir,
                           f"Argo_{row.cruise}_{int(row.profile):03d}.npz")
    save_chains(
        sample, idx, outfile,
        extras=dict(wave=wave, obs_Rrs=Rrs[idx], varRrs=sigRrs[idx]**2,
                    Chl=Chls[idx], Y=Ys[idx],
                    cruise=str(row.cruise),
                    profile=int(row.profile),
                    lat=float(row.lat), lon=float(row.lon),
                    time=str(row.time)),
    )
```

## Why this pattern works

`inference.fit_batch` returns:

- `all_samples`: shape `(n_spectra, nsteps, nwalkers, nparam)` — float32 to save memory.
- `all_idx`: the indices, preserved across the parallel pool so you can map back to your DataFrame.

It uses `ProcessPoolExecutor`, so models are pickled to each worker. Models with `uses_Chl=True` rely on `pdict['Chl'][idx]` — set every index your `items` references, even with placeholder values for the rest.

## Memory and runtime considerations

- A single PACE ExpBricaud+Pow fit at 20k steps / 16 walkers / 5 params ~= 25 MB per `chains` (float32). For 1000 profiles, plan for ~25 GB held in `all_samples` if you don't stream to disk inside the loop.
- For very large batches, fit in smaller chunks (see `l23.batch_fit`'s chunked-write pattern) and call `save_chains` between chunks instead of accumulating.

```python
chunk = 50
for i0 in range(0, len(items), chunk):
    sub = items[i0:i0+chunk]
    samples, idxs = bing_inf.fit_batch(models, pdict, sub, rt_dict, n_cores=10)
    for s, j in zip(samples, idxs):
        # save immediately, drop reference
        save_chains(s, j, ...)
```

## Aggregation into a results CSV

After the batch, follow the L23 convention (in [bing/fitting/l23.py](../../../bing/fitting/l23.py)) to derive per-profile summary stats:

```python
import numpy as np
from glob import glob
from bing.evaluate import thin_burn_chains

rows = []
for npz_path in sorted(glob(os.path.join(out_dir, 'Argo_*.npz'))):
    d = np.load(npz_path, allow_pickle=True)
    flat = thin_burn_chains(d['chains'])
    rows.append(dict(
        cruise   = str(d['cruise']),
        profile  = int(d['profile']),
        lat      = float(d['lat']),
        lon      = float(d['lon']),
        # Example: median log10(Adg)
        log10_Adg_med = float(np.median(flat[:, 0])),
        log10_Adg_p14 = float(np.percentile(flat[:, 0], 14)),
        log10_Adg_p86 = float(np.percentile(flat[:, 0], 86)),
    ))
pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
```

## Common pitfalls

- **`pdict['Chl']` / `pdict['Y']` shape mismatch** with the highest `idx` you submit → `IndexError` inside the worker (hard to debug because it's in another process). Always size them to `max(idx)+1` and fill the indices you actually fit.
- **Reusing one `models` list across the pool** is fine for read-only state (priors, wavelengths), but BING models are mutated in-place by `init_other_bits` (sets `aph_star`, `_shape`). The worker that runs `fit_one` calls `init_other_bits` first, so each worker re-sets state per spectrum — don't try to "pre-cache" across spectra.
- **Forgetting `n_cores`** → defaults to 1, defeats the point.
- **HDF5 backend (`pdict['save_file']`) inside parallel workers** writes to the same file from multiple processes → corruption. Leave `save_file=None` for batch jobs; use NPZ via `save_chains` instead.
- **Restart story**: filenames must be deterministic. If you use `chain_filename(p, idx=row['profile'])`, two profiles with the same `idx` in different cruises will collide — include cruise in the filename.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — the single-spectrum equivalent
- [fit-l23-spectrum](../fit-l23-spectrum/SKILL.md) — uses the same `fit_batch` pattern (`l23.batch_fit`)
- [satellite-band-prep](../satellite-band-prep/SKILL.md) — for PACE noise / band centers

## Reference data

- [papers/biomass/Analysis/](../../../papers/biomass/Analysis/) — matched datasets and fit outputs
- [bing/fitting/inference.py](../../../bing/fitting/inference.py) — `fit_batch`
- [bing/fitting/l23.py](../../../bing/fitting/l23.py) — `batch_fit`, `save_chains`, `chain_filename`
