---
name: add-paper-analysis
description: Lay out a new papers/<topic>/ directory for a BING analysis or paper, following the project conventions used in papers/biomass, papers/bing_2.0, and papers/phytoplankton. Use when the user asks to start a new analysis, create a new paper directory, or organize new research code outside the bing/ package.
---

# Lay out a new papers/<topic>/ analysis

## When to use this skill

- "Start a new analysis directory"
- "Create papers/<topic>/ for the <X> study"
- "Where should this new figure/analysis script live?"
- Any time scientific code needs a home outside the reusable `bing/` package

## Conventions in this repo

Look at the existing examples before creating a new one:

- [papers/biomass/](../../../papers/biomass/) — PACE-Argo BGC matchups (currently active; has untracked subdirs `Argo_Constrained/`, `Frouin/`, `Low_bbp/`)
- [papers/bing_2.0/](../../../papers/bing_2.0/) — Large-scale L23 benchmarking; defines the `chain_filename` / `process_one` / `process_all` pattern
- [papers/phytoplankton/](../../../papers/phytoplankton/) — Model comparison study
- [papers/Solar/](../../../papers/Solar/) — Solar-induced fluorescence / inelastic studies

The shared shape is:

```
papers/<topic>/
├── README.md           # 1-paragraph intro: what, why, status
├── Analysis/           # All scripts, notebooks, and intermediate outputs
│   ├── Fits/           # NPZ outputs from MCMC (per spectrum)
│   ├── *.py            # Pipeline + helper scripts (loaders, batch runners)
│   ├── *.ipynb         # Exploratory notebooks
│   └── *.csv           # Aggregated results, matchup tables
├── Figures/            # Final figures for the manuscript (PDF + PNG)
│   ├── fig_*.py        # Script that builds each figure
│   └── fig_*.{pdf,png} # Output (gitignored if large)
└── Data/               # Static inputs not pulled from external sources
                        # (often a symlink; keep out of git when possible)
```

Notebooks and figure scripts go in `Analysis/` and `Figures/` respectively. Anything that's reusable across papers moves into the `bing/` package.

## Scaffold

```bash
TOPIC=YourTopic
mkdir -p papers/$TOPIC/Analysis/Fits
mkdir -p papers/$TOPIC/Figures
mkdir -p papers/$TOPIC/Data
```

Create `papers/$TOPIC/README.md`:

```markdown
# <Topic> Analysis

**Status**: in progress / submitted / published
**Lead**: <name>
**Target venue**: <journal/conference>

## Question
One sentence: what scientific question this analysis answers.

## Data
- Input: <link to dataset, or path under Data/>
- Pipeline outputs: Analysis/Fits/

## Reproducing

```bash
# From the repo root, with `pip install -e .` already done
python papers/<Topic>/Analysis/run_fits.py
python papers/<Topic>/Analysis/aggregate.py
python papers/<Topic>/Figures/fig_summary.py
```

## Notes
- Anything project-specific that wouldn't fit elsewhere
- Decisions made (e.g., which model combo, which satellite, UV cutoff)
```

## Pipeline script skeleton

A `run_fits.py` script that mirrors [bing/fitting/l23.py::batch_fit](../../../bing/fitting/l23.py):

```python
"""papers/<Topic>/Analysis/run_fits.py — batch MCMC pipeline."""
import os
import numpy as np
import pandas as pd

from bing.parameters import standard
from bing.models import utils as model_utils
from bing.priors import priors as bing_priors
from bing.fitting import inference as bing_inf
from bing.fitting.l23 import save_chains
from bing.rt import defs as rt_defs

OUT_DIR = os.path.join(os.path.dirname(__file__), 'Fits')
os.makedirs(OUT_DIR, exist_ok=True)

def main(n_cores=10, debug=False):
    p = standard.expb_pow(
        satellite='PACE',
        nsteps=20000, nburn=2000,
        scl_noise='PACE',
    )
    rt_dict = rt_defs.rt_dict_from_p(p)

    # ---------- 1. Load matchups ----------
    matched = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                       'matched_<topic>.csv'))
    if debug:
        matched = matched.head(5)

    # ---------- 2. Build wavelength + Rrs arrays ----------
    wave = ...  # (nwave,)
    Rrs    = ...
    varRrs = ...

    # ---------- 3. Init models, priors, pdict ----------
    models = model_utils.init(p.model_names, wave)
    bing_priors.set_standard_priors(models, p)
    pdict = bing_inf.init_mcmc(models, nsteps=p.nsteps, nburn=p.nburn)
    pdict['Chl'] = matched.get('Chl', np.zeros(len(matched))).values
    pdict['Y']   = matched.get('Y',   np.ones(len(matched))).values

    # ---------- 4. Build items (with checkpoint skipping) ----------
    items = []
    for i, row in matched.iterrows():
        outfile = os.path.join(OUT_DIR, f"{row.id}.npz")
        if os.path.exists(outfile):
            continue
        p0 = ...   # see run-bing-fit skill for the log10 dance
        items.append((Rrs[i], varRrs[i], p0, i))

    # ---------- 5. Parallel fit ----------
    samples, idxs = bing_inf.fit_batch(models, pdict, items, rt_dict,
                                       n_cores=n_cores)

    # ---------- 6. Save NPZs ----------
    for s, idx in zip(samples, idxs):
        save_chains(s, idx,
                    os.path.join(OUT_DIR, f"{matched.iloc[idx].id}.npz"),
                    extras=dict(wave=wave,
                                obs_Rrs=Rrs[idx], varRrs=varRrs[idx],
                                Chl=pdict['Chl'][idx], Y=pdict['Y'][idx]))

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_cores', type=int, default=10)
    ap.add_argument('--debug', action='store_true')
    main(**vars(ap.parse_args()))
```

## Aggregation script skeleton

```python
"""papers/<Topic>/Analysis/aggregate.py — combine per-fit NPZ → CSV."""
from glob import glob
import numpy as np
import pandas as pd
from bing.evaluate import thin_burn_chains

FIT_DIR = os.path.join(os.path.dirname(__file__), 'Fits')

rows = []
for path in sorted(glob(os.path.join(FIT_DIR, '*.npz'))):
    d = np.load(path, allow_pickle=True)
    flat = thin_burn_chains(d['chains'])
    rows.append(dict(
        id     = os.path.basename(path)[:-4],
        Chl    = float(d['Chl']),
        # Add per-parameter summaries here
        param0_med = float(np.median(flat[:, 0])),
        param0_p14 = float(np.percentile(flat[:, 0], 14)),
        param0_p86 = float(np.percentile(flat[:, 0], 86)),
    ))
pd.DataFrame(rows).to_csv(
    os.path.join(FIT_DIR, '..', 'results_summary.csv'),
    index=False)
```

## Figure script skeleton

```python
"""papers/<Topic>/Figures/fig_summary.py — main paper figure."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, '..', 'Analysis', 'results_summary.csv')

def main():
    df = pd.read_csv(RESULTS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df.Chl, 10**df.param0_med, s=10)
    ax.set_xlabel('Chl [mg m$^{-3}$]')
    ax.set_ylabel(r'$A_{dg}$ [m$^{-1}$]')
    ax.set_xscale('log'); ax.set_yscale('log')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_summary.{ext}'),
                    dpi=300, bbox_inches='tight')
    print('Saved fig_summary.{pdf,png}')

if __name__ == '__main__':
    main()
```

## What stays in `bing/`, what stays in `papers/`

| Belongs in `bing/` (reusable) | Belongs in `papers/<topic>/` (one-off) |
|---|---|
| A new bio-optical model | Loading code for one specific CSV |
| A general plotting function | Layout for one specific figure |
| A new prior class | Per-paper parameter choice (e.g., UV cutoff at 410) |
| Validation utilities used by ≥2 papers | The matchup table for this paper |
| A new RT correction term | The script that runs the batch for this paper |

When in doubt: write it under `papers/`, and migrate to `bing/` when a second paper would benefit. Migration is a focused PR — don't bundle it with the analysis itself.

## Gitignore for large data

Per-paper outputs can blow up the repo size. Suggested `.gitignore` entries (add to repo-level `.gitignore`):

```
papers/*/Analysis/Fits/*.npz
papers/*/Figures/*.pdf
papers/*/Figures/*.png
papers/*/Data/
```

Keep small CSV summaries tracked; keep heavy NPZ + figure outputs untracked. The full output is regenerable from `run_fits.py`.

## Common pitfalls

- **Putting reusable code in `papers/` and importing it from another paper** → first time you do this is a smell; lift to `bing/` instead.
- **`from anly_utils_XX import ...`** scattered across `papers/` — these are paper-local helpers and won't import elsewhere. Keep relative imports paper-local.
- **Hard-coded absolute paths** (`/home/xavier/...`) — use `os.path.dirname(__file__)` to anchor.
- **Notebooks tracked with output cells** → bloat. Strip outputs before committing (or use `nbstripout`).
- **No README** → six months later you (or a collaborator) won't remember which script does what.

## Related skills

- [run-bing-fit](../run-bing-fit/SKILL.md) — fitting code your pipeline calls
- [batch-fit-argo](../batch-fit-argo/SKILL.md) — pattern your `run_fits.py` mirrors
- [plot-bing-fit](../plot-bing-fit/SKILL.md) — figure helpers
- [fit-l23-spectrum](../fit-l23-spectrum/SKILL.md) — reference implementation in [bing/fitting/l23.py](../../../bing/fitting/l23.py)
