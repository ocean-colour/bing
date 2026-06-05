# Fit the full L23 dataset (again!)

## Goals

Here are our goals:

- Examine biases in the bbp fitting based on the L23 dataset

## Data

We may use the simulated spectra provided by Loisel et al. (2023) for this.  Those data are located in $OS_COLOR/Loisel2023 and can be loaded with the ocpy.hydrolight.loisel23.load_ds() method.

## Skills

Consider using the skills in .claude/skills/

## Code

### Writing

Here are guidelines for writing code:

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place any new code in the existing Analysis/py directory
- Use lines of code that are less than 80 characters wide
- Include doc strings for all functions and methods, including parameters and return values

## Development

1. Following the existing code in the modules in Analysis/py, fit the full L23 dataset with BING.  Do this:

- Generate a new module named bbp_fit_l23.py in Analysis/py
- Model the module after the fit_with_argo.py module
- Fit the full L23 dataset with BING
- Begin with fitting to the elastic spectra of L23 using the standard Gordon coefficients
- Output the fits in $OS_COLOR/Biomass/L23_Fits
- Generate a figure for each fit. Include the correct anw and bbp as black lines
- Do not add noise to the spectra
- Start with fitting only 3 in a debug=True mode but enable the full dataset 

2. I wish to parse the outputs of the fitting into a pandas dataframe and write to disk as a csv file.  The dataframe should have the following columns:

- idx of the L23 data
- Fitted bbp and bbp_sig
- Fitted beta and beta_sig
- Fitted aph and aph_sig
- Fitted adg and adg_sig
- Fitted Sdg and Sdg_sig
- True bbp from the L23 data
- True aph from the L23 data
- True adg from the L23 data

You can read the outputs from the fits in the $OS_COLOR/Biomass/L23_Fits folder.  The fitted values are provided in the stats array.  Generate a new method in bbp_fit_l23.py to parse the outputs and write to the dataframe. 

3.  I have modified the code to generate the spectra in a different way.  The fitting method is now erroring.  Please debug it.  Also, holding all of the chains in memory is not feasible when running on the full dataset.  Please:

- Modify the fit_all_l23() method to fit the spectra in batches of 500
- You will need to save/plot the fits in these batches
- Run a test on the 3 DEBUG_IDX spectra
- Update the Logs


## Modifications

1. Please make these modifications to the bbp_fit_l23.py module:

- Include the true value of bbp in the PNG output
- Do add noise to the spectra.  Standard PACE noise.
- Have the default be not to include the corner plot in the figure 
- Include a 0 line in the Rrs panel
- Run the test
- Update the Logs

2. Please make these modifications to the bbp_fit_l23.py module:

- Include the true value of Aph and Adg in the PNG output
- Remove the existing outputs and rerun the test
- Update the Logs

3. Can you also parallelize the saving and plotting?

- Update the Logs

## Prompts

1. Read this doc.  Proceed with the first item under Development
2. Read this doc.  Proceed with the 1st item under Modifications
3. Read this doc.  Proceed with the 2nd item under Modifications
4. Read this doc.  Proceed with the 2nd item under Development
5. Read this doc.  Proceed with the 3rd item under Development
6. Read this doc.  Proceed with the 3rd item under Modifications

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-04 (Created bbp_fit_l23.py for the L23 dataset)

Added `Analysis/py/bbp_fit_l23.py`, modeled after `fit_with_argo.py` but
targeting the Loisel et al. (2023) synthetic dataset, to examine bbp (and
anw) retrieval biases against truth.  The module reuses
`bing.fitting.l23` (`load_one_l23`, `prep_one_l23`, `fit_one`) and
`bing_inf.fit_batch` for parallel MCMC, and the local `fitting.plot_fit`
for the diagnostic figures.

Baseline configuration (`get_l23_params`): ExpBricaud + power-law, the
*elastic* Gordon forward model with the *standard* (constant) Gordon
coefficients (`variable_Gordon=False`), no Raman/fluorescence, on the
PACE wavelength grid.  Fits are written to `$OS_COLOR/Biomass/L23_Fits`
as `L23_<idx>.npz/.json/.png` via `bing.io.save_fit`, with one figure per
fit overlaying the true anw and bbp as black lines.  `fit_all_l23(debug=True)`
fits only three spectra (idx 170/180/200); `debug=False` enables the full
3320.  Learned that the elastic L23 spectrum with standard Gordon
coefficients is exactly what `prep_one_l23` produces when the
`variable_Gordon`/`include_*` flags are off, so no custom forward-model
code was needed.

### 2026-06-04 (Modifications: PACE noise, true-bbp value, no corner, zero line)

Applied the Modifications list to `bbp_fit_l23.py` (and small additive
options to `fitting.plot_fit`):

- **PACE noise**: `get_l23_params` now sets `add_noise=True` (with the
  default `scl_noise='PACE'`), so the synthetic spectra carry standard
  PACE noise.
- **True bbp value in the PNG**: `plot_l23_fit` annotates the true
  bbp amplitude at the 600 nm power-law pivot (the quantity `Bnw`
  targets), in both linear and log10, via a new optional `bb_anno`
  argument to `plot_fit`.
- **Corner plot off by default**: added a `show_corner` flag to
  `plot_fit` (default True to preserve other callers) and to
  `plot_l23_fit` (default False); when off, no `tmpc.png` is written and
  the bottom-right panel is left blank.
- **Zero line in the Rrs panel**: `plot_fit` now draws a dotted gray
  reference line at Rrs=0.

Ran the test (`python bbp_fit_l23.py 1`): all three debug spectra fit and
saved cleanly (e.g. idx 170 reduced χ² ≈ 1.09 with PACE noise), confirming
the figure content.  The retrieved bbp band sits a touch low relative to
truth at idx 170 (fit log Bnw = -3.61 vs true -3.49) — a first hint of the
bbp bias this study is meant to quantify across the full dataset.

### 2026-06-04 (Modifications #2: true Aph / Adg in the PNG)

Added the true Aph and Adg amplitudes to the figure, in the same
ExpBricaud parameterization as the fitted values so they compare directly:
Adg = a_dg(400 nm) and Aph = a_ph(440 nm), both reported in log10.
`plot_l23_fit` interpolates `odict['adg']`/`odict['aph']` to 400/440 nm
and passes the text via a new optional `anw_anno` argument on
`fitting.plot_fit` (mirroring `bb_anno`).  The annotation is placed at the
lower-right of the anw panel so it clears the legend and the descending
a_nw curve.

Removed the prior `L23_*` outputs and reran `python bbp_fit_l23.py 1`.  The
three debug fits regenerated cleanly (idx 170 reduced χ² ≈ 0.76 on this
noise draw).  At idx 170 the retrieval recovers Adg well (fit log Adg =
-1.55 vs true -1.70) but underestimates Aph (fit log Aph = -2.91 vs true
-2.14) — i.e. it pushes phytoplankton absorption low while keeping the
dissolved/detrital term close, a partitioning degeneracy worth watching
alongside the bbp bias when the full dataset is run.

### 2026-06-05 (Development #2: parse fits to a CSV dataframe)

Added `parse_fits()` (plus the helpers `_med_lo_hi` and
`load_one_l23_truth`) to `bbp_fit_l23.py`, and a `flg==3` entry point.
It globs the `L23_*.json` sidecars in `$OS_COLOR/Biomass/L23_Fits`
(reading only the lightweight JSON `stats` block, so the ~13 MB NPZ chains
never load), pulls each fitted parameter's median and 14/86 percentiles,
and pairs them with the L23 truth from `l23.load_one_l23`.

The CSV (`L23_fit_summary.csv`) has exactly the requested columns: `idx`,
`bbp`/`bbp_sig`, `beta`/`beta_sig`, `aph`/`aph_sig`, `adg`/`adg_sig`,
`Sdg`/`Sdg_sig`, `true_bbp`, `true_aph`, `true_adg`.  Amplitudes (Bnw,
Aph, Adg) are fit in log10 and converted to linear so fitted/true columns
compare directly; their sigma is half the linear 14–86 interval.  Slopes
(beta, Sdg) are linear and reported as-is.  Reference wavelengths match
the model pivots: bbp at 600 nm, aph at 440 nm, adg at 400 nm.

The full run had already completed on disk, so the parser produced 2996
rows in one pass.  First look at the population-level biases (log10
fit − truth):

- **bbp: essentially unbiased** — median fit/true = 1.001, mean log10 bias
  −0.002, scatter 0.044.  The elastic + standard-Gordon retrieval recovers
  backscattering very well across all water types.
- **aph: underestimated** — median fit/true = 0.80, mean log10 bias −0.33
  with large scatter (0.57).
- **adg: slightly overestimated** — median fit/true = 1.16, mean log10
  bias +0.065 (scatter 0.124).

The aph-low / adg-high pattern confirms the dissolved-vs-phytoplankton
partitioning degeneracy seen in the single-spectrum figure, while bbp —
the quantity of primary interest for the biomass study — is robust.

### 2026-06-05 (Development #3: debug new spectra path + batch the fitting)

The spectra are now generated with
`lowest_bbp.generate_pace_spectrum(idx, use_elastic=True)` instead of
`l23.prep_one_l23`, and `fit_all_l23` was erroring with
`IndexError: list index out of range` from inside `fit_batch`.

**Root cause(s).** `bing_inf.fit_one` looks up `pdict['Chl'][idx]` where
`idx` is the *L23 spectrum index* (e.g. 170, 3003), but the code had set
`pdict['Chl']` to a length-3 positional list — so any real L23 index blew
past the end.  Fixed by indexing Chl by L23 idx: `pdict['Chl']` is now a
`np.zeros(N_L23)` array filled at `pdict['Chl'][idx]`.  A second latent bug:
the initial guess `p0` was used directly from `init_guess` (linear), but
the log-prior amplitudes (Adg, Aph, Bnw) must be in log10 — so walkers
started outside the priors and `Chl = 10**p0[2]/0.05582` was nonsense.
Restored the log10 conversion (mirroring `l23.prep_one_l23`) in the new
`_prep_spectrum` helper.

**Batching.** `fit_all_l23` now processes the indices in chunks of
`BATCH_SIZE` (500): each chunk is prepped, fit via `fit_batch`,
saved+plotted, then its chains are `del`-eted before the next chunk — so
the full dataset's chains never co-reside in memory.  Models, priors and
the MCMC config are built once and reused across batches; only per-spectrum
data and the per-idx Chl change.  Also added `plt.close(fig)` after save in
`fitting.plot_fit` so 3320 figures don't accumulate.

**Consistency fix.** Since generation switched to the *elastic* L23
dataset `load_ds(1, 0)`, `parse_fits` now loads truth from `(1, 0)` too
(it had used `(4, 0)`), so the CSV's `true_*` columns match the fitted
spectra.

**Test.** `python bbp_fit_l23.py 1` fit the three DEBUG_IDX spectra
(170/180/3003, all very-low-bbp cases) in a single batch and saved
NPZ/JSON/PNG cleanly (e.g. idx 3003 reduced χ² ≈ 0.88).  At these low-bbp
spectra the bbp retrieval reads slightly high (idx 3003: fit log Bnw =
-3.71 vs true -4.01), worth tracking when the full elastic dataset is run.

### 2026-06-05 (Modifications #3: parallelize save + plot)

The per-spectrum save/plot loop inside each batch was serial; since
`bing_io.save_fit` reconstructs IOPs from the chains (CPU-heavy) and the
figure rendering is slow, that step dominated wall-clock once the MCMC was
batched.  Pulled it into a top-level worker `_save_plot_one(args)` and
fanned it out with a `ProcessPoolExecutor(max_workers=n_cores)` + `tqdm`,
mirroring the `fit_batch` pattern.

**Gotcha — unpicklable namedtuple.** The first attempt passed the BING
parameter tuple `p` straight into the worker and hit
`PicklingError: Can't pickle ... BING20_tuple`: `p_ntuple.gen` builds the
namedtuple class dynamically, so it isn't importable by name and can't
cross the process boundary (this is exactly why `fit_batch` never passes
`p`).  Fixed by sending `p._asdict()` (a plain dict) and rebuilding it in
the worker with `p_ntuple.gen(**p_dict)`.  Everything else in the packed
tuple (models, chains, p0, Rrs, varRrs, odict) was already picklable —
`fit_batch` relies on the same.

**Test.** Reran `fit_all_l23(debug=True, n_cores=3, clobber=True)`: the
three fits saved + plotted in parallel, and `bing.io.load_fit` round-trips
each output (NPZ + JSON) without error.