# Radiative Transfer calculations

This module will guide prompts related to generic radiative transfer calculations in BING.  

## Skills

Consider using the skills in .claude/skills/

## Coding

Here are guidelines for coding: 

- Use Python
- Add inline comments to explain the effort
- Keep lines to 80 characters or less.
- Reuse existing code when possible
- Use methods, not classes
- Use matplotlib or seaborn for plotting
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods
- Use the PEP 8 style guide for coding.

## Development 

1. At the moment, the reconstruct_from_chains() method in evaluate.py calculates Rrs without using the calc_Rrs_from_models() method.  Please:

- Refactor the code to use the calc_Rrs_from_models() method.  
- Generate a new test module named tests/test_evaluate.py that tests the new method.  You may use the test_l23_fitting.py module as a template and may need to load and fit L23 data as part of the test.
- Log your work and results below in the Logs section.  

## Docs

## Polishing

## Prompts

1. Read this doc. Proceed with the 1st item under Development.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-04 (Refactor reconstruct_from_chains to delegate Rrs assembly to calc_Rrs_from_models)

**What changed**

- [bing/evaluate.py](../bing/evaluate.py) `reconstruct_from_chains` no longer
  hand-rolls the elastic / Raman / G0 / Gb / fluorescence branches. After
  burning + thinning the chains, it now:
  1. computes `a`, `bb` once for the credible-band statistics, then
  2. calls `calc_Rrs_from_models(...)` with the flattened parameter arrays
     to get the per-sample Rrs, and
  3. reduces those samples to median + std for the return.
  Net diff: ~60 lines of forward-model code collapse to a single call.
- Fixed a latent batch-shape bug in `calc_Rrs_from_models`: the
  `include_Chl_fl` branch was doing `aph_ex = aph[a_model.i_Chl_ex]`, which
  works for 1-D `a_params` (the `inference.log_prob` path) but indexes the
  sample axis -- not the wavelength axis -- when called with 2-D chains.
  Changed to `aph[..., a_model.i_Chl_ex]`, which is identical for the
  single-spectrum case and correct for the batch case. Required for the
  refactor; pre-existing code never hit it because batch fluorescence went
  through the duplicated logic in `reconstruct_from_chains`.

**New tests**

- [bing/tests/test_evaluate.py](../bing/tests/test_evaluate.py) — new module
  modeled on `test_l23_fitting.py`. Uses `pytest.fixture(scope="module")`
  so the L23 MCMC fits (one each for standard Gordon, variable Gordon, and
  Raman) run once per session and feed multiple assertions.
- Coverage:
  - `thin_burn_chains` shape semantics on synthetic chains.
  - `calc_stats` median / percentile recovery on a known Gaussian.
  - `reconstruct_from_chains` end-to-end through L23 fits for the three
    rt_dict variants, with the standard physical-bounds checks.
  - **Refactor invariant** (`test_reconstruct_matches_calc_Rrs_from_models`
    + Raman variant): runs `reconstruct_from_chains`, then independently
    re-runs `calc_Rrs_from_models` on the same flattened chains and asserts
    the median / std agree to `rtol=1e-12`. This is the property the
    refactor is designed to preserve.
  - `calc_Rrs_from_models` with a 1-D parameter vector (the log_prob
    contract) and `reconstruct_chisq_fits` 1-D + 2-D paths.
- Chain length tuned to `nsteps=8000` so the default
  `thin_burn_chains(burn=7000)` leaves a healthy post-burn sample. Whole
  module runs in ~20 s on the dev machine.

**Validation**

- `pytest bing/tests/test_evaluate.py -x -q` → 10 passed in ~20 s.
- `pytest bing/tests/test_l23_fitting.py::test_single_fit_standard_Gordon
  -x -q` → 1 passed (sanity-check that the refactored path is what the
  legacy test now exercises too).

**What I learned**

- `calc_Rrs_from_models` was effectively the canonical forward model
  already -- both `chisq_fit.fit_func` and `inference.log_prob` go through
  it. `reconstruct_from_chains` was the lone holdout, and keeping its
  Raman / Gb / fluorescence assembly in sync with the canonical path was
  exactly the maintenance hazard the refactor eliminates.
- `bb_R` was being wrapped with `np.outer(np.ones(N), bb_model.bb_R)` in
  the old reconstruct code. That tile-up is not required: the downstream
  Raman correction broadcasts a 1-D `bb_R` against the 2-D `a`, `bb`
  samples automatically. `calc_Rrs_from_models` already relied on that
  and the new path inherits it.
- Tests that touch `reconstruct_from_chains` must use ≥7000 MCMC steps
  (or pass an explicit `burn=` to `thin_burn_chains`) because the default
  burn is large. Worth a follow-up to either lower the default or surface
  the burn argument in the reconstruct API.