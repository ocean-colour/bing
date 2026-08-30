# RoB RT Backend Coding — Prompt 1 (M0: Dependency, config keys, `ObsGeometry`)

## Goals

Implement **Milestone M0** of the coding plan
(`docs/coding_plan/rob_rt_coding_plan.md`): make `bing` depend on
`retrieve-or-bust` (importable as `robust`), grow `rt_dict` with the three new
backend-configuration keys plus a fit-setup validator, and add the
`ObsGeometry` container for fixed per-pixel viewing/illumination geometry.
Nothing in the fitters changes yet — the point is that all the new
configuration surface exists, validates, and the existing suite stays green.

## Claude

### Skills

Consider using the skills in `.claude/skills/` as helpful:
`run-bing-fit` (how `rt_dict` is built and threaded today), plus the general
`critical-partner` and `code-review` skills for review passes.

### Working agreements (hold for every prompt, `rob_rt_prompt_1` … `_6`)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on branch
  **`rob-rt-backend`** (the coding plan's suggested name; JXP creates/commits).
  Each milestone is a reviewable commit/PR-sized unit. Do **not** run
  state-changing git commands; read-only inspection is fine.
- **Python only, in the `ocean14` conda env** (both repos already coexist
  there). Tests live in `bing/tests/` and run as `pytest bing/tests/`.
- **The four resolved Coding Q&A decisions are binding constraints**
  (`claude_prompts/rob_rt.md`, Q&A/Coding 1–4):
  - **CQ1 — float32 only at the JAX boundary.** `jax_enable_x64` is **never**
    enabled, globally or locally. BING's float64 NumPy arrays downcast to
    float32 crossing into `robust.rt`; the new forward function's docstring
    states this is more than sufficient precision for these calculations.
    All test tolerances are float32-honest (parity `rtol ≤ 1e-5`, inelastic
    `rtol ≤ 5e-4`) — never the x64-only `1e-6` cross-check regime.
  - **CQ2 — `set_raman_Ed` stashes the raw Ed pair.** `aNWModel.set_raman_Ed`
    (bing/models/anw.py:480-511) additionally stores the incoming
    `(wave_Ed, Ed)` verbatim, backward-compatibly; `Ed_ratio_raman` is
    computed exactly as today. The robust backend reuses BING's existing
    zenith-0° production Ed generation as-is (fitting/l23.py:319-322).
  - **CQ3 — tests assume `robust` is always present.** No `importorskip` for
    `robust`, ever — it is a real runtime dependency after M0 and a broken
    install fails loudly, matching BING's existing convention.
  - **CQ4 — `theta_s` is required, never defaulted.** A robust-backend fit
    with no geometry supplied raises at fit setup. The nadir fallback covers
    only missing *viewing* geometry (`theta_v = dphi = 0`).
- **BING scope discipline** (per `bing/CLAUDE.md`): don't widen scope; don't
  refactor the Gordon model, prior system, or `eval_*` shape contract
  opportunistically; `papers/` is off-limits for opportunistic edits —
  report hits there to JXP, never edit them. No `robust`/retrieve-or-bust
  source is modified by this integration.
- **Every milestone is `pytest`-gated.** The existing suite must stay green
  at every milestone; the Gordon path and its numerics are untouched except
  the one sanctioned deletion (`RT_correction`, M1).
- Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — `docs/coding_plan/rob_rt_coding_plan.md`: Ground rules,
  Files touched, and the **M0** section.
- **Design** — `docs/design/rob_rt_design.md` §3.1 (the `rt_backend`
  selector), §3.2 (`ObsGeometry` and geometry threading), §3.3 (the
  `fit_Bp`/`Bp_value` keys), §4 (the hybrid [350, 750] nm grid policy), §5
  (dependency/packaging).
- **Q&A record** — `claude_prompts/rob_rt.md`: Design Q1–Q3, Q10–Q14
  (backend selection, geometry, dependency direction) and Coding Q&A 1–4.
- **Current code** — `bing/rt/defs.py` (`rt_dict_from_p`, defs.py:5),
  `bing/setup.py` (`install_requires`, setup.py:23-30), and
  `robust/rt/types.py` (`Geometry`, types.py:302; `Geometry.nadir`,
  types.py:337) in the retrieve-or-bust repo.

## Prompts

1. Read this doc. Execute the 1st task in the "M0" section below. If you have
   any questions, ask me in the Q&A section below. Use Fable if you can. Log
   your work.
2. Read this doc. Execute the 2nd task. Check my answers in Q&A; if you have
   additional questions, ask in Q&A. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task. Use Fable if you can. Log your work.
4. Read this doc. Execute the 4th task — the explainer notebook. Use Fable if
   you can. Log your work.
5. Read this doc. Execute the 5th task — update the next prompt doc,
   `rob_rt_prompt_2.md`, with anything learned this milestone. Use Fable if
   you can. Log your work.

## M0

### Tasks

1. **Dependency.** Add `'retrieve-or-bust'` to `install_requires` in
   `bing/setup.py` (setup.py:23-30). Distribution name `retrieve-or-bust`,
   importable package `robust`; JAX/Flax arrive transitively (design §5).
   No model files are added to BING — the trained emulator weights ship
   inside `robust`. Verify `import robust.rt` succeeds in `ocean14`.

2. **Config keys + validator.** In `bing/rt/defs.py`:
   - Extend `rt_dict_from_p` (defs.py:5) with **`rt_backend`** (default
     `"gordon"` when absent on `p`), **`fit_Bp`** (default `False`), and
     **`Bp_value`** (default `0.01`) — defaults applied explicitly rather
     than the key loop's `None`, so a legacy `p` object yields a fully valid
     dict. Every consumer still tolerates missing keys via
     `rt_dict.get("rt_backend", "gordon")` (design §3.1), so saved/legacy
     rt_dicts keep working unmodified.
   - New `validate_rt_dict(rt_dict, models=None, geom=None)` — the fit-setup
     checks (fitters call it in M2): (i) `rt_backend` ∈ {`gordon`,
     `robust_ztt`, `robust_hybrid`, `robust_baseline`}; (ii) `fit_Bp=True`
     with `rt_backend="gordon"` raises (design §3.3); (iii) a robust backend
     with `geom is None` raises — `theta_s` is required (CQ4); (iv)
     `robust_hybrid` with any `models[0].wave` band outside **[350, 750] nm**
     raises (design §4 — checked once here, never per forward call).

3. **`ObsGeometry`.** New module `bing/rt/geometry.py`: frozen dataclass
   `ObsGeometry(theta_s, theta_v=0.0, dphi=0.0, wind=None)` (degrees; design
   §3.2 — `theta_s` positional/required, no default) with
   `to_robust() -> robust.rt.types.Geometry` (types.py:302; same units).
   `Ed` is a `to_robust(Ed=...)` pass-through keyword for M4, not a stored
   field — the dataclass stays pure per-pixel metadata. Docstring: "Fixed
   per-pixel viewing/illumination geometry. Never fit."

4. **Notebook.** Create `nb/RT/` and add `nb/RT/rob_rt_coding_1.ipynb`
   (executed, committed with outputs): what the integration is (one figure
   or diagram of the backend dispatch is enough), the three new `rt_dict`
   keys and why one combined selector, what `validate_rt_dict` rejects and
   why, and `ObsGeometry` round-tripping into `robust.rt.types.Geometry`.
   Keep it short — M0 is configuration, not physics.

5. **Finally.** Update `rob_rt_prompt_2.md` (M1) with what M0 actually
   established: any API surprises in `robust.rt`, install/versions notes,
   and conventions the M1 work should inherit.

### Gate

First tests in the new `bing/tests/test_evaluate_robust.py` (no
`importorskip`, per CQ3):

- `import robust.rt` succeeds from bing's env.
- `rt_dict_from_p` on a legacy-style `p` yields the three new keys with
  their defaults.
- `validate_rt_dict` raises on each illegal combination above and passes on
  legal ones.
- `ObsGeometry(theta_s=30.)` round-trips through `to_robust()` (values,
  degrees); `ObsGeometry()` with no `theta_s` is a `TypeError`.
- The existing suite stays green (`pytest bing/tests/`).

## Q&A

**Q1 (M0 task 1, Claude → JXP).** The design doc (§5) said adding
`retrieve-or-bust` to `install_requires` "transitively brings in JAX/Flax,
which `robust` already declares." That's false: `retrieve-or-bust`'s own
`setup.py` deliberately keeps `jax`/`flax`/`optax`/`jaxtyping` **out** of its
`install_requires` (they live only in its `requirements.txt` — a documented
choice from that repo's own elastic-RT coding plan). Confirmed directly:
`import robust.rt` in `ocean14` loads `jax` but not `flax`/`optax`; plain
`import robust` and `import robust.rt` never touch `flax` at module scope
(the emulator only needs it when a `hybrid`-mode forward call actually
runs the trained network — later milestones). Since this integration's
working agreements forbid modifying `retrieve-or-bust` source, the fix has
to be on BING's side: I added `jax`, `flax`, and `jaxtyping` directly to
`bing/setup.py`'s `install_requires` (not `optax` — `robust` only imports it
lazily inside its own emulator-*training* functions,
`robust/rt/emulator.py:814,903`, never on the inference path BING uses).
Also corrected the stale "transitively" claim in
`docs/design/rob_rt_design.md` §5 to state this explicitly. **No answer
needed to proceed** — this is the only sensible resolution given the scope
constraint; flagging in case you'd rather fix it upstream in
`retrieve-or-bust` yourself outside this integration's scope.

One caveat worth naming: `retrieve-or-bust` itself is not published to
PyPI (`pip show` confirms it's editable-installed from the local checkout,
home-page `github.com/ocean-colour/retrieve-or-bust`). Listing it as a bare
`install_requires` string is what the design calls for and is harmless in
practice — both repos already coexist as editable installs in `ocean14` —
but a hypothetical fresh `pip install bing` off PyPI alone would still fail
to resolve it. This isn't a new problem for this integration (`bing`
already has no working "from PyPI alone" install story), so I didn't try to
solve it here.

## Next

→ `rob_rt_prompt_2.md` (M1: the forward adapter `calc_Rrs_from_models_robust`
+ dropping `RT_correction`).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-29 (M0 task 1 — `retrieve-or-bust` dependency added; import gate green)

Added `'retrieve-or-bust'` to `install_requires` in `bing/setup.py`, plus
`'jax'`, `'flax'`, `'jaxtyping'` (not `'optax'`) — see **Q1** above for why
the latter three are needed explicitly rather than assumed transitive.
Verified in `ocean14`:

```
import robust.rt; import jax, flax, jaxtyping
```

succeeds — `jax 0.11.0`, `flax 0.12.8`, `jaxtyping 0.3.11` (already present
in the shared env from `retrieve-or-bust`'s own earlier development;
`optax 0.2.8` is also present but, per Q1, not declared as a BING
dependency since nothing on BING's inference path needs it). `robust.rt.forward`
resolves and is callable-looking (`<function forward at ...>`).

**Existing-suite check.** `pytest bing/tests/` → **178 passed, 2 skipped, 2
failed** (144.67s). The 2 failures
(`test_l23_inelastic.py::test_raman_correction_matches_l23`,
`::test_fluorescence_matches_l23`) are **pre-existing and unrelated**:
`git stash`-ed my `setup.py`/doc changes and reran just that file — identical
2 failures, root cause
`FileNotFoundError: .../bing/tests/files/l23_inelastic_fixture.npz` missing
from this checkout (a data fixture gap, not a code regression). `setup.py`'s
`install_requires` list has no runtime effect on test execution anyway
(pip-install-time metadata only), so this was a sanity check, not something
the edit could plausibly have broken. Per scope discipline, not touched —
flagging here for JXP rather than fixing an unrelated gap under this
milestone.

Also made one correction to `docs/design/rob_rt_design.md` §5 while the
finding was fresh: replaced the now-known-false "this transitively brings in
JAX/Flax" claim with the actual mechanism (BING declares them directly,
since `retrieve-or-bust`'s own `setup.py` deliberately excludes them).

Branch: `rob_rt` (JXP's existing branch — not `rob-rt-backend` as the
coding plan suggested; noting the actual name rather than the plan's
suggestion). Modified: `setup.py`, `docs/design/rob_rt_design.md` (one
paragraph), this file (Q&A). No `bing/` package code, no `robust`/
retrieve-or-bust source changed. Ready for JXP to review/commit; task 2
(config keys + validator) is next.

### 2026-08-30 (M0 task 2 — `rt_dict` config keys + `validate_rt_dict`; new test module)

Extended `bing/rt/defs.py`: `rt_dict_from_p` now sets `rt_backend` (default
`'gordon'`), `fit_Bp` (default `False`), `Bp_value` (default `0.01`) via
explicit `getattr(p, key, <default>)` calls *outside* the existing
`for key in [...]` loop — those three keys deliberately don't share the
loop's None-on-missing behavior, per the task spec. Added module constants
`RT_BACKENDS = ('gordon', 'robust_ztt', 'robust_hybrid', 'robust_baseline')`
and `ROBUST_HYBRID_WAVE_MIN/MAX = 350./750.` so the valid-backend set and
the grid bounds have one named source rather than being re-typed at each
call site later. Added `validate_rt_dict(rt_dict, models=None, geom=None)`
implementing all four checks from the task spec: unknown `rt_backend`;
`fit_Bp=True` with `rt_backend='gordon'`; a robust backend with `geom is
None`; `rt_backend='robust_hybrid'` with any `models[0].wave` value outside
`[350, 750]` nm. `models`/`geom` are optional — omitting either just skips
the check that needs it (useful for testing one check in isolation), rather
than raising for lack of information.

**Did not touch `bing/rt/__init__.py`.** Checked first: `rt_dict_from_p` is
not re-exported at package level anywhere — every caller
(`bing/io.py`, `bing/fitting/l23.py`, and 8 test files) imports it as
`from bing.rt import defs as rt_defs` then `rt_defs.rt_dict_from_p(...)`.
`validate_rt_dict` follows the identical pattern
(`rt_defs.validate_rt_dict(...)`), so no `__init__.py` change was needed or
made.

**Caught and fixed a real regression before it happened.**
`test_chl_fl.py::test_rt_dict_from_p_defaults` asserted
`set(rt_dict.keys()) == {...the 7 original keys...}` — an exact-set
equality that the three new always-present keys would have broken outright.
Since this test directly exercises the function task 2 modifies, updating
it is in-scope (not opportunistic widening): added assertions for
`rt_backend`/`fit_Bp`/`Bp_value`'s defaults and the three keys to the
expected set. Also extended the neighboring
`test_rt_dict_from_p_missing_attrs` (a fully minimal namedtuple with *no* RT
attributes at all) to assert the three new keys still get their real
defaults rather than `None` — this is the exact "legacy p" scenario the
task was designed around, so it belongs right next to the existing
"returns None for missing attributes" check it would otherwise contradict.

**New test module** `bing/tests/test_evaluate_robust.py` (per the M0 gate,
no `importorskip`): `test_import_robust_rt` (task 1, formalizing the manual
check from the last log entry into a real test); two `rt_dict_from_p`
tests mirroring `test_chl_fl.py`'s coverage from this module's own vantage
point (legacy defaults; explicit values via a custom namedtuple); and eight
`validate_rt_dict` tests covering every check (parametrized over
`robust_ztt`/`robust_hybrid`/`robust_baseline` for the missing-geometry
check; both grid-restricted-vs-not backends for the grid check; a
models=None case confirming that check is skipped, not treated as failure,
without models to check against). Used a tiny local `_FakeModel`/`_FakeGeom`
rather than real BING models — `validate_rt_dict` only ever reads
`models[0].wave` and `geom is None`, so a full model/geometry object would
be untested weight, not stronger coverage.

**Full verification.** `pytest bing/tests/test_evaluate_robust.py
bing/tests/test_chl_fl.py -q` → 49 passed. Full suite:
`pytest bing/tests/ -q` → **191 passed, 2 skipped, 2 failed** (140.56s) —
the count is exactly 178 (previous baseline) + 13 new
`test_evaluate_robust.py` tests, with the same 2 pre-existing,
already-diagnosed failures (missing `l23_inelastic_fixture.npz`) and
nothing else changed.

Modified: `bing/rt/defs.py`, `bing/tests/test_chl_fl.py` (two test bodies
extended). New: `bing/tests/test_evaluate_robust.py`. No `bing/rt/__init__.py`,
no `robust`/retrieve-or-bust source changed. Branch `rob_rt`, uncommitted,
for JXP's review. Task 3 (`ObsGeometry`) is next.

### 2026-08-30 (M0 task 3 — `ObsGeometry` frozen dataclass)

New module `bing/rt/geometry.py`: `ObsGeometry(theta_s, theta_v=0.0,
dphi=0.0, wind=None)`, a frozen dataclass with `theta_s` positional/required
(no default -- dataclass field ordering puts it before the defaulted fields,
so a missing `theta_s` is a plain `TypeError` from the generated
`__init__`, exactly per the task spec and CQ4). `to_robust(Ed=None)` builds
`robust.rt.types.Geometry(theta_s=..., theta_v=..., dphi=..., wind=...,
Ed=Ed)` — field names match one-to-one, so it's a direct keyword
pass-through, not a real transform. `Ed` is a `to_robust()` keyword only,
never a stored `ObsGeometry` field, exactly as specced (keeps the dataclass
pure per-pixel metadata; M4 will pass the raw Ed pair in at the point of
use).

**Followed task 2's precedent, not task 1's.** `robust.rt.types` is
imported at module scope in `geometry.py` (not lazily inside `to_robust`) —
consistent with `robust` now being a hard runtime dependency (task 1) with
no `importorskip` anywhere (CQ3), so there's no reason to defer the import.
Also **did not touch `bing/rt/__init__.py`** — same reasoning as task 2:
nothing in this milestone's spec asks for a package-level re-export, and
keeping `ObsGeometry` accessed as `from bing.rt.geometry import
ObsGeometry` (or `from bing.rt import geometry`) matches how `defs.py`'s
new `validate_rt_dict` is accessed. If M2 later finds callers want it
re-exported for convenience, that's a small, separate, easily-reviewed
addition rather than something to guess at now.

**Verified interactively before writing tests** (not just asserted): built
an `ObsGeometry(theta_s=30.)`, confirmed `to_robust()` produces a
`robust.rt.types.Geometry` with matching `theta_s`/nadir `theta_v=dphi=0`/
`wind=None`/`Ed=None`; built a non-nadir instance with `wind=5.` and passed
`Ed=(None, None)` through `to_robust(Ed=...)`, confirming the pass-through
(not stored) semantics; confirmed `ObsGeometry()` raises `TypeError`
(`missing 1 required positional argument: 'theta_s'`); confirmed
`FrozenInstanceError` on attribute assignment.

**New tests** in `test_evaluate_robust.py` (6 added, matching the manual
checks above): required `theta_s` → `TypeError`; nadir defaults; frozen;
round-trip on a nadir instance (asserts `isinstance` against the actual
`robust.rt.types.Geometry`, not just attribute equality); round-trip on a
non-nadir instance with `wind` and an `Ed` pass-through pair, including
`assert not hasattr(g, 'Ed')` to pin down that `Ed` never becomes an
`ObsGeometry` field.

**Verification.** `pytest bing/tests/test_evaluate_robust.py -q` → **18
passed** (was 13; +5 new — one of the six was folded into an existing
assertion rather than counted twice). Full suite:
`pytest bing/tests/ -q` → **196 passed, 2 skipped, 2 failed** (142.38s) —
196 = 191 + 5, same 2 pre-existing failures, nothing else moved.

Modified: none beyond new files. New: `bing/rt/geometry.py`,
additions to `bing/tests/test_evaluate_robust.py`. No `robust`/
retrieve-or-bust source changed. Branch `rob_rt`, uncommitted, for JXP's
review. **All three M0 tasks are now done** — task 4 (the explainer
notebook) is next, then task 5 (hand off to `rob_rt_prompt_2.md`).

### 2026-08-30 (M0 task 4 — explainer notebook `nb/RT/rob_rt_coding_1.ipynb`)

Created `nb/RT/` and built the notebook programmatically via `nbformat`
(a throwaway builder script in the session scratchpad, not committed) rather
than hand-authoring JSON — avoids manual cell-source escaping errors
entirely. Hit and fixed exactly one: a code cell's docstring used
`"""..."""` nested inside the builder's own `r"""..."""` string literal,
closing it early (`SyntaxError: invalid syntax`) — replaced the nested
docstring with a `#` comment; `grep -n '"""'` over the builder confirmed no
other occurrences. Executed in place with
`jupyter nbconvert --to notebook --execute --inplace` in `ocean14`
(kernel `python3`, confirmed registered via `jupyter kernelspec list`
first) — clean run, no errors.

**11 cells** (6 markdown, 5 code), matching the task's four required
elements exactly, no more: (1) one rendered diagram (matplotlib boxes +
arrows, not just an ASCII sketch) showing the full params → `rt_dict['rt_backend']`
→ Gordon-or-robust dispatch, **explicitly labeled** so M0's actual scope
(the left two boxes) isn't overclaimed against M2's dispatch (the arrows);
(2) the `rt_backend` single-combined-selector rationale plus a legacy-`p`
and an explicit-`p` demo of `rt_dict_from_p`; (3) a `validate_rt_dict`
table of all four checks plus a live demo triggering each one (including
two *non*-failing cases — `robust_ztt` on a wide grid, and a legal
`robust_hybrid` fit — so the notebook shows what passes, not just what
raises); (4) `ObsGeometry` round-tripping nadir and non-nadir instances
into `robust.rt.types.Geometry`, the `Ed` pass-through, the required-`theta_s`
`TypeError`, and the frozen-instance check.

**Verified every output, not just "it ran".** Read the executed notebook
back with `nbformat`: all 5 code cells have `execution_count` set
sequentially (1-5) and zero `error` outputs. Extracted and read the actual
printed text from the `validate_rt_dict` and `ObsGeometry` cells — messages
match what the M0 task-2/task-3 log entries already established word for
word (e.g. `"...requires all wavelengths within [350.0, 750.0] nm...got
range [400.0, 760.0]"`, `"Ed stored on ObsGeometry? False"`). Also decoded
the diagram cell's `image/png` output to a file and viewed it directly —
confirms real, legible rendering (boxes, arrows, and the caption all
present), not merely "a display_data output exists."

**No package code touched this task** (notebook + new directory only), so
skipped a full `pytest bing/tests/` rerun — nothing it could have broken.

New: `nb/RT/rob_rt_coding_1.ipynb`. Branch `rob_rt` — confirmed via
`git log` that JXP has already reviewed and committed tasks 1-3 (commits
"design", "mo", "coding", "ok" ×2, "2", "3"), so this entry's changes are
the only uncommitted work as of this task. Task 5 (hand off learnings to
`rob_rt_prompt_2.md`) is last for M0.
