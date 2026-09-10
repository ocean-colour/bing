# RNG hardening follow-up (bing side)

**Filed:** 2026-09-07 · **Branch context:** `rob_cdom` · **Origin:** IOPtics
`claude_prompts/rt_tests.md` task 11, Q47b · **Status:** proposed, not started

## Root cause (2026-09-07)

`bing.fitting.inference` imports `bing.evaluate`, which imports **JAX** at module
scope (added 2026-08-30 with the `robust.rt` backends). Importing JAX **draws 10
values from the legacy global MT19937 stream** — the same `numpy.random` global
stream that

1. `emcee.EnsembleSampler` snapshots into `sampler._random` at construction, and
2. `bing.fitting.inference.init_walkers` draws its walker ball from.

So the RNG state at walker init depends on *whether JAX has already been
imported in this process*, not only on the seed. In a `ProcessPoolExecutor` the
import is lazy and per-worker, so a worker's **first** record is seeded from a
stream advanced past the seed while its later records are not — the same record,
the same seed, a different chain, purely as a function of pool layout. A serial
run (where prepping already imported bing) differs from a pooled one for the
same reason.

**IOPtics has patched its own side** (`ioptics/run.py::_warm_fit_imports`): the
fitting stack is imported *before* every per-record `np.random.seed(...)`, and
the same function is the `initializer=` of both fitting process pools. That makes
the state at walker init a function of the seed alone **for IOPtics' call
pattern**. It does not fix bing for anyone else, and it is a workaround for a
global-state dependency that should not exist.

Two hardening items, in priority order.

## 1. Thread an explicit `numpy.random.Generator` (the real fix)

Any caller that seeds by mutating a process-global stream is one stray import —
or one library that draws at import time — away from irreproducibility. Give the
fit an explicit generator instead.

Half of this already exists and is simply not reachable from the public entry
points: **`init_walkers` already takes `rng=`** (duck-typed on `.uniform`,
defaulting to the `np.random` module). Nothing above it passes one.

Sketch:

- `run_emcee(..., rng=None)` — accept it and forward to its `init_walkers` call
  (currently `init_walkers(p0, nwalkers, models=models, ...)` with no `rng`).
- `fit_one(items, models=..., pdict=..., rt_dict=..., rng=None)` — accept it and
  forward to `run_emcee`. This is the entry point IOPtics calls, so it is the one
  that has to grow the parameter for any of this to help.
- Prefer a `numpy.random.Generator` (`np.random.default_rng(seed)`) as the value
  passed: it satisfies `init_walkers`' `.uniform(low, high, size)` contract, so
  no change is needed inside `init_walkers` itself. Note that a `Generator` and
  the legacy module draw **different streams** from the same nominal seed, so
  callers who switch will see different (equally valid) chains — a one-time,
  deliberate break that wants a changelog line rather than silence.
- Seed the sampler's *proposal* stream from the same generator instead of from
  the global one:

  ```python
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, ...)
  if rng is not None:
      sampler._random = np.random.RandomState(          # emcee 3 wants a
          rng.integers(0, 2**32 - 1))                   #   legacy RandomState
  ```

  `emcee.EnsembleSampler` snapshots `numpy.random.mtrand._rand` at construction
  unless told otherwise; overwriting `sampler._random` immediately after is the
  supported way to pin the *proposal* stream (emcee ≥ 3.0). Check the installed
  emcee's attribute name before relying on it — it is private API, so pin the
  emcee version or feature-detect.
- `batch_fit` / `fit_batch`: derive a per-item child generator, e.g.
  `rng.spawn(n)` or `np.random.default_rng([seed, item_index])`, so the chains
  are a function of `(seed, item)` and not of the pool layout. This is exactly
  what IOPtics does today with `zlib.crc32` over `(seed, algorithm, dataset,
  obs_id)` and a `np.random.seed()` call; with a `Generator` it becomes clean.
- Keep `rng=None` behaving as today (global stream) for one release so nothing
  downstream breaks silently.

**Test:** two `fit_one` calls with `rng=np.random.default_rng(7)`, one of them
after `import jax` in a fresh interpreter, must produce bit-identical chains.
That is the regression the current code fails. A cheaper unit-level version:
`init_walkers(p0, 32, models=models, rng=np.random.default_rng(7))` twice, with
`np.random.seed(0)` and an `import jax` between them, must agree — this one
already passes, which is the point: the plumbing above it is what is missing.

## 2. Make the JAX import lazy for Gordon-only use

Even with (1), the import cost and the import-time side effects are worth
removing. `bing/evaluate.py` currently does `import jax` / `import jax.numpy as
jnp` / `from robust import rt as robust_rt` at module scope, so **every** bing
user pays a multi-second JAX import (and its RNG draws, and a CUDA probe warning
on GPU machines) even for a pure `rt_backend='gordon'` fit that never touches
robust.

Move both into the robust code path — e.g. import inside
`calc_Rrs_from_models_robust` (and any other robust-only helper), or behind a
small module-level `_robust()` accessor that memoizes the import. The Gordon
path then imports numpy and scipy only.

Two things to check while doing it:

- module-level `jnp` references outside the robust functions (there may be
  none — worth a grep for `jnp.` and `jax.` in `evaluate.py`);
- `bing/fitting/chisq_fit.py` imports `bing.evaluate` at module scope and
  `bing.evaluate` imports `chisq_fit` back, so the circular pair resolves
  today only because of import order — a lazy import here is likely to *help*
  that, but re-run the import-order tests.

**Test:** in a subprocess, `import bing.fitting.inference` then assert
`'jax' not in sys.modules`; and assert the global RNG state is unchanged across
that import.

## Why both, and in this order

(1) is the correctness fix: it removes the dependency on global state, so a
future library that draws at import time cannot reintroduce the bug. (2) is a
cost and hygiene fix that also happens to remove *today's* instance of it. Doing
(2) alone would look like a fix and would not be one.
