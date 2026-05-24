 # Saving results from BING fits to files

## Goals

Capture all of the main outputs of a BING fit and the inputs required to reproduce the fit.

## Code

Here are guidelines for the code: 

- Use Python
- When possible use existing methods from the repository
- Add inline comments to explain the effort
- Use methods, not classes
- Place import statements at the top of the file.
- Use numpy.savez() to save arrays 
- Use JSON files for inputs and simple stats
- Add functions to the bing/io.py module to save and load the fits.

## Items to save

- The wavelengths of the fitted spectra
- The observed remote sensing reflectance
- The variance of the observed remote sensing reflectance
- All of the parameters in the parameter tuple used for the fit
- The names of the models used for the fit
- A version number for BING
- The MCMC chains
- The priors 
- The initial guess for the fit
- The reconstructed remote sensing reflectance from the chains using evaluate.reconstruct_from_chains()
- The stats from the chains using evaluate.calc_stats() with the variable names

## Testing

If you need to run python, use the "ocean14" environment in conda.

## Docs

Examine the files in the docs/ directory and update the docs to reflect the new changes.  In particular:

## Planning

### Output layout

For each BING fit we will produce **two files** that share a common
basename `<base>`:

- `<base>.npz` — large numerical arrays (saved with `numpy.savez`):
  - `wave`      — wavelengths of the fitted spectrum
  - `Rrs`       — observed remote sensing reflectance
  - `varRrs`    — variance of observed Rrs
  - `chains`    — MCMC chains, shape `(nsteps, nwalkers, nparam)`
  - `p0`        — initial guess used to seed MCMC (post-LM if a least-
    squares pass was used)
  - `p0_init`   — *optional* pre-LM seed.  Only written when the caller
    supplies it.
  - Reconstructed quantities from `evaluate.reconstruct_from_chains()`:
    `a`, `bb`, `a_lo`, `a_hi`, `bb_lo`, `bb_hi`, `Rrs_recon`,
    `sigRrs_recon`

- `<base>.json` — small, human-readable metadata (saved with `json.dump`):
  - `bing_version`     — value of `bing.__version__`
  - `model_names`      — list `[a_model_name, bb_model_name]`
  - `pnames`           — concatenated parameter names from both models
  - `params`           — full contents of the parameter named-tuple as a
    dict (so the fit can be reproduced); numpy arrays / namedtuples are
    converted to plain lists.  `rt_dict` is **not** stored separately —
    it can be reconstructed at load time via
    `bing.rt.defs.rt_dict_from_p(p)`.
  - `priors`           — list of prior dicts for the absorption model
    followed by the backscattering model (same order as `pnames`).
  - `stats`            — output of `evaluate.calc_stats(chains, pnames,
    perc=stats_perc)`.  All numpy arrays inside are converted to lists
    so the entire stats dict (including parameter names and the
    median / percentile arrays) lives in JSON.
  - `stats_perc`       — `[lo_perc, hi_perc]` used in `calc_stats`.
  - `recon_perc`       — `[lo_perc, hi_perc]` used in
    `reconstruct_from_chains`.

This split honours the guideline "use `numpy.savez` for arrays and JSON
for inputs and simple stats".  The `evaluate.calc_stats` output is
small (one median + two percentile arrays at one number per parameter)
so it lives in JSON; the per-wavelength reconstruction arrays stay in
the `.npz` file.

### Functions to add to `bing/io.py`

All public functions take simple arguments — no new classes are
introduced.

- `save_fit(outroot, p, models, chains, p0, Rrs, varRrs,
            p0_init=None, stats_perc=(14, 86), recon_perc=(5, 95))`
  - Builds `rt_dict` internally via
    `bing.rt.defs.rt_dict_from_p(p)` — callers never have to pass it.
  - Computes `stats = evaluate.calc_stats(chains, pnames,
    perc=stats_perc)` and writes it to the JSON file.
  - Computes the reconstructed quantities via
    `evaluate.reconstruct_from_chains(models, chains, rt_dict,
    perc=recon_perc)` and writes them to the `.npz` file.
  - If `p0_init` is supplied, writes it to the `.npz` file as
    `p0_init`; otherwise the field is omitted.
  - Writes `<outroot>.npz` and `<outroot>.json`.
  - Returns `(npz_path, json_path)`.

- `load_fit(outroot) -> dict`
  - Loads both files, returns a single dict that contains every key
    saved above plus:
    - `'models'`: rebuilt via `bing.models.utils.init` with the saved
      priors re-attached, so callers can immediately reconstruct
      IOPs without serialising Python objects.
    - `'rt_dict'`: regenerated via
      `bing.rt.defs.rt_dict_from_p(p)` from the saved parameter tuple
      so it is available without being persisted.
    - `'p'`: the original named-tuple, rebuilt from the saved
      `params` dict via `p_ntuple.gen(**params)`.

- `_params_to_dict(p)` (private helper)
  - Converts a `BING20_tuple` (or any namedtuple) into a JSON-safe
    dict.  Tuples → lists, numpy arrays → lists, leaving floats / ints
    / bools / strings / None untouched.

- `_priors_from_models(models)` (private helper)
  - Walks `models[i].priors.priors` and returns a list of plain dicts
    (`flavor`, `pmin`, `pmax`, plus `mean`/`sigma` if Gaussian).  This
    matches the shape of `apriors` / `bpriors` so the saved priors can
    be fed straight back into `init` in the future.

### BING version

Add `__version__ = "0.0.dev0"` (matching `setup.py`) to
`bing/__init__.py` so `io.save_fit` can record it without re-parsing
`setup.py` at runtime.

### Test script (Development item 1)

`bing/tests/test_io.py` will be turned into a *runnable script* (no
`pytest` assertions yet) that:

1. Builds a small MCMC fit using L23 data
   (`fit_l23.fit_one` with `nsteps≈200`, `nburn≈20`,
   `satellite='PACE'`, `variable_Gordon=False`) so the script finishes
   in a few seconds in the `ocean14` conda env.
2. Calls `bing.io.save_fit(...)` to write `test_io.npz` + `test_io.json`
   into a temporary directory under `bing/tests/files/`.
3. Calls `bing.io.load_fit(...)` and prints a short summary so the
   round-trip is visible when the script is run by hand.
4. Cleans up its temporary files at the end.

A proper `pytest` test will be added in a later development step.

### Docs updates

- Create a new `docs/save_load.rst` that demonstrates
  `bing.io.save_fit` / `bing.io.load_fit`, lists the file layout, and
  references `rt_dict_from_p` so users understand the rt_dict is
  reconstructable.  Add it to the main `toctree` in `docs/index.rst`.
- `docs/api/index.rst` — add an entry for the new `bing.io` module.

### Development order

1. Add `__version__` to `bing/__init__.py`.
2. Implement helpers (`_params_to_dict`, `_priors_from_models`) and
   `save_fit` / `load_fit` in `bing/io.py`.
3. Write the script version of `bing/tests/test_io.py` and execute it
   in the `ocean14` conda env to confirm the round-trip works.
4. Promote the script to a `pytest` test (later development step).
5. Update the docs.

### Clarifications

1. **Re-creating models on load**: my plan re-builds the model objects
   from `model_names` + `wave` when loading rather than pickling them.
   That keeps the saved files portable but means a custom-prior model
   has its priors re-attached from the JSON.  Please confirm this is
   acceptable, or say if you'd rather we skip re-creating models in
   `load_fit` and just return the raw arrays/metadata.

This is acceptable.

2. **`p0` semantics**: the initial guess saved is the value passed to
   `inference.run_emcee` (post-LM, pre-MCMC).  If you'd also like the
   pre-LM seed saved, say so and I'll add a `p0_init` field.

Yes, save the pre-LM seed if it exists (have it be optional).

3. **Stats percentiles**: I default `calc_stats` to `(14, 86)` (its
   own default) and `reconstruct_from_chains` to `(5, 95)`.  If you'd
   prefer a single shared percentile pair for both, let me know.

This is fine, but record these percentiles.  Either in the npy or JSON file. 

4. **Single-fit vs. batch**: this design saves one fit per file pair.
   If you'd like a multi-spectrum container (e.g. one file with many
   chains stacked), I'd add a second pair of `save_fits` /
   `load_fits` methods on top.

Let's begin with a single fit per file pair.

### Modifications

Make these modifications to the plan:

- Save the output from evaluate.calc_stats() to the JSON file
- You can reconstruct rt_dict from the parameters in the parameter tuple using the rt_dict_from_p() function, so it does not need to be input or saved
- Call the new docs file docs/save_load.rst instead of docs/data_processing.rst

## Code modificattions

### First batch

Make the following modifications to the code:

- Move the _params_to_dict() function to the bing/parameters/p_ntuple.py module
- Move the _priors_from_models() and _split_priors() functions to the bing/priors/priors.py module

## Prompts

1. Read this doc.  Generate a plan for the development of the code and write it in the Planning section above.  If you have any questions, ask them in the Clarifications section above.

2. Read this doc.  Modify the plan to include the Modifications section above and my answers to the Clarifications section above.

3.  Read this doc.  Proceed with the development of the code as described in the Planning section above.
