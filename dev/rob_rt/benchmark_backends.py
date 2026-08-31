"""Throughput benchmark: ``log_prob`` across all four ``rt_backend`` values.

rob_rt integration, M5 task 3 (design §7.1's accept/optimize decision on
the hybrid backend). See ``claude_prompts/RT/rob_rt_prompt_6.md`` (Goals,
Working agreements, M5 task 3) for the full spec this implements.

Purpose
-------
This is **evidence, not a gate** -- per the working agreement quoted in
the prompt ("record them; do not optimize the hybrid path on your own
judgment") this script contains **no threshold, assertion, or pass/fail
logic of any kind**. It runs, prints numbers, and exits 0 regardless of
what those numbers are. The accept/optimize call on ``robust_hybrid`` is
JXP's, not this script's.

What "MCMC-realistic batch shape" means here
---------------------------------------------
Read live from ``bing/fitting/inference.py``'s ``run_emcee``: the sampler
is built with ``emcee.EnsembleSampler(nwalkers, ndim, log_prob,
args=[...])`` and **no** ``pool=`` -- so emcee's default serial map calls
``log_prob`` once per walker per step, each call taking a single 1-D
``params`` vector (shape ``(ndim,)``), never a batch of walkers at once.
"MCMC-realistic" therefore means exactly that: unbatched, single-walker
calls to ``log_prob`` with a length-5 ``params`` vector (3 absorption + 2
backscattering parameters, the standard ``ExpBricaud``+``Pow`` combination
used throughout this integration's tests), repeated many times -- which is
precisely what a real multi-thousand-step run experiences, one call per
walker per step. There is no meaningful "batch of walkers in one call" to
benchmark on this hot path, because emcee itself never constructs one
here.

``log_prob``'s own body (read before benchmarking) does more than the bare
forward call: it peels an optional B_p tail, evaluates the model priors
(and the B_p prior when ``fit_Bp``), dispatches to
``calc_Rrs_from_models``/``calc_Rrs_from_models_robust``, then computes
the Gaussian log-likelihood and sums in the prior terms. Benchmarking
``log_prob`` itself (rather than the forward call alone) is what's
faithful to the real per-step MCMC cost -- the prior evaluation and
likelihood reduction are real, if small, overhead on every step, not just
the forward model.

What this script measures
--------------------------
For each of the four ``rt_backend`` values (``gordon``, ``robust_ztt``,
``robust_hybrid``, ``robust_baseline``):

1. **First-call / JIT-compile cost** (robust backends only -- ``gordon``
   has no JIT and is reported for contrast): ``evaluate._robust_forward_jit``
   is ``cache_clear()``-ed immediately before, so the timed call is a
   genuine cold compile -- the same cost a fresh interpreter (or, per the
   ``fit_batch`` note below, a fresh ``ProcessPoolExecutor`` worker) would
   pay on its very first call for that (mode, inelastic, wave) combination.
2. **Warm-path throughput** (the primary number): after that first call has
   populated the ``lru_cache``, ``N_WARM`` further calls are timed and
   reported as calls/s -- this is what a real multi-thousand-step MCMC run
   actually experiences once past its own first step. Each call uses a
   slightly different ``params`` vector (small random jitter around the
   same point) so the measurement isn't artificially favoring one exact
   value -- JAX's ``jit`` never retraces on value changes alone (only on
   shape/dtype/static-arg changes), so this does not reintroduce any
   compile cost; it just makes the loop a closer stand-in for a real
   sampler walking through nearby posterior values.

Per-worker ``fit_batch`` JIT-compile-cost note
-----------------------------------------------
``fit_batch`` (``bing/fitting/inference.py``) distributes fits across a
``ProcessPoolExecutor``. Each worker is a separate OS process with its own
Python interpreter and its own **empty** ``_robust_forward_jit`` module-level
``lru_cache`` -- caches never share across processes. So a batch fit with
``n_cores=N`` worker processes pays the first-call JIT-compile cost
measured below **N times over, once per worker**, not once for the whole
batch -- e.g. an 8-core ``robust_hybrid`` batch pays ~8x the single-process
first-call cost in aggregate (spread across the 8 processes, so it need not
show up as extra wall-clock time if the batch is large enough to amortize
it, but it is real, repeated compute). This is stated here for JXP's
awareness, per the prompt's explicit instruction -- this script does not
attempt to fix or work around it (e.g. no attempt to pre-warm workers or
share a compilation cache across processes).

Real, measured numbers (2026-08-31, ocean14, this script -- canonical run
transcribed verbatim below; three repeat runs agreed to within a few
percent on every figure. See the Logs entry in
claude_prompts/RT/rob_rt_prompt_6.md for the full run transcript and
methodology notes)
---------------------------------------------------------------------------
Setup: ExpBricaud+Pow, 61-band grid 400-700 nm (robust_hybrid-legal),
geom=ObsGeometry(theta_s=30.), Bp_value=0.014, no Raman/fluorescence,
N_WARM=2000 repeated single-walker log_prob calls per backend after the
first (JIT-warm) call.

    backend          warm calls/s    vs gordon    first-call cost (s)
    gordon              38936           1.000x      0.0001   (no JIT)
    robust_ztt           5991           0.154x      0.182
    robust_hybrid        5667           0.146x      0.181
    robust_baseline      7274           0.187x      0.019

Headline: all three robust backends run at roughly 15-19% of gordon's
per-call throughput (~5.3-6.9x slower per log_prob call) at this batch
shape -- robust_hybrid is not dramatically worse than robust_ztt/
robust_baseline (0.146x vs 0.154x/0.187x), i.e. the learned-emulator
correction is not the dominant extra cost relative to the other robust
backends; the ~5-7x cost relative to gordon is a property of dispatching
through JAX/robust.rt at all (device dispatch + trace overhead per call,
even after compilation), not something specific to the hybrid emulator.
First-call compile costs are small in absolute terms (~0.18 s for ztt/
hybrid, ~0.02 s for baseline) -- far smaller than the ~1.1 s hybrid
figure measured in M2 (rob_rt_prompt_3.md's smoke-fit numbers, a full
tiny end-to-end MCMC fit in a fresh process). That is a real, reproduced
discrepancy worth flagging (see Q3 in rob_rt_prompt_6.md): the M2 number
was measured as part of a full fit's first *step* in a colder process
(first-ever JAX call, cold OS disk cache for the Flax emulator weights),
while this script's number isolates just the first ``log_prob`` call in a
process that has already imported ``jax``/``robust`` and, in repeat runs
on this machine, benefits from a warm OS file cache for the emulator
weights file. Both numbers are legitimate measurements of different
things; neither supersedes the other. This script's number is the more
representative one for a *warm* process reusing an already-loaded
interpreter (e.g. a long-running batch worker after its very first
distinct-config call), while M2's ~1.1 s remains the more representative
number for a *genuinely cold* process (e.g. the very first worker to
start in a fresh ``fit_batch`` run on a machine with a cold disk cache).

Run
---
    conda run -n ocean14 python dev/rob_rt/benchmark_backends.py
"""
import time

import numpy as np

from bing.models import utils as model_utils
from bing.fitting import inference as bing_inf
from bing.rt import defs as rt_defs
from bing.rt.geometry import ObsGeometry
from bing.parameters import standard
from bing import evaluate


#: Number of warm-path calls timed per backend. Chosen so the whole script
#: runs in a few seconds: fast enough not to be a burden on dev tooling,
#: large enough (checked empirically) that the per-backend calls/s figure
#: is stable across repeat runs.
N_WARM = 2000

#: Standard ExpBricaud (3 params: Adg, Sdg, Aph-amplitude) + Pow (2 params:
#: Bnw, beta) combination used throughout this integration's own tests
#: (bing/tests/test_evaluate_robust.py's ``robust_models``/``threading_setup``
#: fixtures) -- a representative, already-validated model pair, not a new
#: benchmark-only choice.
_MODEL_NAMES = ['ExpBricaud', 'Pow']

#: A robust_hybrid-legal wavelength grid (inside [350, 750] nm) so all four
#: backends can be benchmarked on the *same* grid without tripping
#: validate_rt_dict's hybrid range check.
_WAVE = np.linspace(400., 700., 61)

#: A representative, physically unremarkable single parameter vector
#: (Adg, Sdg, Aph_log10amp, Bnw_log10amp, beta) -- one "walker's" position,
#: matching the shape log_prob actually receives from emcee.
_PARAMS0 = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])

#: Fixed per-pixel geometry for the robust backends (never fit; see
#: bing/rt/geometry.py). theta_s=30 deg is the same value used throughout
#: test_evaluate_robust.py.
_GEOM = ObsGeometry(theta_s=30.)

#: Shared, non-backend-specific rt_dict keys for the robust backends.
_ROBUST_COMMON = dict(include_Raman=False, include_Chl_fl=False,
                      phi_C=0.02, double_gaussian=True, Bp_value=0.014,
                      fit_Bp=False)


def _build_models_and_data():
    """Build the model pair and a self-consistent (Rrs, varRrs) pair.

    The "observed" Rrs is the noiseless Gordon-backend forward model at
    ``_PARAMS0`` -- what data it represents doesn't matter for a throughput
    benchmark (all four backends are timed on the *same* Rrs/varRrs/params
    triple), only that varRrs is a physically sane fraction of Rrs so
    log_prob's likelihood term is a finite, representative number rather
    than a degenerate one.

    Returns
    -------
    tuple
        (models, Rrs, varRrs) -- models is [a_model, bb_model].
    """
    p = standard.expb_pow(wv_min=_WAVE.min(), wv_max=_WAVE.max(),
                          variable_Gordon=False)
    models = model_utils.init(p.model_names, _WAVE, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))

    rt_dict_gordon = rt_defs.rt_dict_from_p(p)
    Rrs = evaluate.calc_Rrs_from_models(
        models[0], _PARAMS0[:models[0].nparam], models[1],
        _PARAMS0[models[0].nparam:], rt_dict_gordon)
    Rrs = np.squeeze(Rrs)
    varRrs = (0.02 * Rrs) ** 2
    return models, Rrs, varRrs, rt_dict_gordon


def _time_calls(fn, n_iter, rng):
    """Mean calls/s of ``fn(params)`` over ``n_iter`` calls with jittered
    params.

    Each call uses ``_PARAMS0`` plus a tiny (0.1%) random perturbation --
    enough to walk through nearby posterior values like a real sampler,
    never enough to leave the region log_prob was warmed up on, and never
    a shape/dtype change (so no JAX retrace is possible).

    Parameters
    ----------
    fn : callable
        ``fn(params) -> float``, e.g. a closure around ``log_prob``.
    n_iter : int
        Number of timed calls.
    rng : numpy.random.Generator
        Source of the per-call jitter, so repeat runs are reproducible.

    Returns
    -------
    tuple
        (calls_per_sec, elapsed_seconds).
    """
    jitter = rng.normal(scale=1e-3 * np.abs(_PARAMS0), size=(n_iter, _PARAMS0.size))
    t0 = time.perf_counter()
    for row in jitter:
        fn(_PARAMS0 + row)
    elapsed = time.perf_counter() - t0
    return n_iter / elapsed, elapsed


def benchmark_backend(backend, models, Rrs, varRrs, rt_dict_gordon, rng):
    """Time one backend's first-call (JIT) cost and warm-path throughput.

    Parameters
    ----------
    backend : str
        One of ``bing.rt.defs.RT_BACKENDS``.
    models, Rrs, varRrs : see ``_build_models_and_data``.
    rt_dict_gordon : dict
        The already-built Gordon rt_dict, reused verbatim for
        ``backend == 'gordon'`` (byte-for-byte the same dict the rest of
        BING would build via ``rt_dict_from_p``).
    rng : numpy.random.Generator
        Shared RNG for the warm-path jitter (see ``_time_calls``).

    Returns
    -------
    dict
        ``{'backend', 'first_call_s', 'warm_calls_per_sec'}``.
    """
    if backend == 'gordon':
        rt_dict, geom = rt_dict_gordon, None
    else:
        rt_dict = dict(_ROBUST_COMMON, rt_backend=backend)
        geom = _GEOM
        # Force a genuine cold compile: clear the lru_cache immediately
        # before the timed first call, so this is not accidentally reusing
        # a closure built by an earlier backend/earlier run in this same
        # process (distinct backends never share a cache entry anyway --
        # see evaluate._robust_forward_jit's own docstring -- but clearing
        # makes the "first call" claim airtight rather than incidental).
        evaluate._robust_forward_jit.cache_clear()

    def _call(params):
        return bing_inf.log_prob(params, models, Rrs, varRrs, rt_dict, geom=geom)

    t0 = time.perf_counter()
    _call(_PARAMS0)
    first_call_s = time.perf_counter() - t0

    warm_calls_per_sec, _ = _time_calls(_call, N_WARM, rng)

    return dict(backend=backend, first_call_s=first_call_s,
                warm_calls_per_sec=warm_calls_per_sec)


def main():
    rng = np.random.default_rng(20260831)
    models, Rrs, varRrs, rt_dict_gordon = _build_models_and_data()

    results = []
    for backend in rt_defs.RT_BACKENDS:
        results.append(
            benchmark_backend(backend, models, Rrs, varRrs, rt_dict_gordon, rng))

    gordon_rate = next(r['warm_calls_per_sec'] for r in results
                       if r['backend'] == 'gordon')

    print("=" * 78)
    print("log_prob throughput benchmark -- MCMC-realistic (single-walker) calls")
    print(f"N_WARM = {N_WARM} calls/backend, model pair = {_MODEL_NAMES}, "
          f"{_WAVE.size}-band grid [{_WAVE.min():.0f}, {_WAVE.max():.0f}] nm")
    print("=" * 78)
    header = f"{'backend':<18}{'warm calls/s':>14}{'vs gordon':>12}{'first-call (s)':>18}"
    print(header)
    print("-" * len(header))
    for r in results:
        ratio = r['warm_calls_per_sec'] / gordon_rate
        note = "  (no JIT)" if r['backend'] == 'gordon' else ""
        print(f"{r['backend']:<18}{r['warm_calls_per_sec']:>14.1f}"
              f"{ratio:>11.3f}x{r['first_call_s']:>18.4f}{note}")
    print()
    print("Evidence only -- no pass/fail threshold is applied by this script.")
    print("The accept/optimize call on robust_hybrid's throughput is JXP's, "
          "per the M5 working agreement.")
    print()
    print("Per-worker fit_batch note: each ProcessPoolExecutor worker in "
          "fit_batch(n_cores>1) has its own empty _robust_forward_jit "
          "lru_cache (caches never share across processes), so the "
          "first-call compile cost above is paid once per worker process, "
          "not once for the whole batch.")


if __name__ == "__main__":
    main()
