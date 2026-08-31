"""Tests for the robust.rt backend integration (rob_rt).

See ``docs/design/rob_rt_design.md`` and ``docs/coding_plan/rob_rt_coding_plan.md``
for the design/plan these tests gate, and ``claude_prompts/rob_rt.md`` /
``claude_prompts/RT/rob_rt_prompt_*.md`` for the resolved Q&A and the
milestone-by-milestone execution record.

No ``pytest.importorskip`` for ``robust`` anywhere in this module: per
Q&A/Coding item 3, ``retrieve-or-bust`` is a real runtime dependency of
``bing`` (see ``setup.py``), not an optional one -- a broken install should
fail loudly here, the same as any other BING dependency.

Coverage so far (M0):
    * ``import robust.rt`` succeeds and exposes ``forward`` (task 1).
    * ``rt_dict_from_p``'s three new keys and their real (non-None) defaults
      -- also covered from the ``rt_dict_from_p`` side in
      ``test_chl_fl.py`` (task 2).
    * ``validate_rt_dict``'s four illegal-configuration checks (task 2).
    * ``ObsGeometry`` -- required ``theta_s``, nadir defaults, frozen,
      round-trips through ``to_robust()`` (task 3).

M1 task 1 (this addition): ``calc_Rrs_from_models_robust`` -- baseline-vs-
Gordon parity, shape contract across all three robust backends (1-D and
batch), ``full_return``, the free-``Bp`` override, Raman/fluorescence
branches, and the four error paths (missing ``geom``, a non-robust
``rt_backend``, ``robust_baseline`` + inelastic, and -- via M0's
``validate_rt_dict``, not repeated here -- the grid/backend/``fit_Bp``
checks).

M1 task 2: the ``_robust_forward_jit`` lru_cache (hit/miss counts, distinct
configs, None-vs-instance inelastic keys).

M1 task 3: ``robust_domain_check`` -- ``DomainWarning`` fires un-jitted on a
deliberately out-of-domain IOP set while the jitted hot path on the same
inputs is silent (Gate item 5), the in-domain/ztt/baseline silence, and the
shared argument-error paths.

M1 task 4: the ``RT_correction`` block is gone -- a stale key in an rt_dict
is silently ignored (Gate item 6). The companion regression pin (the Gordon
path's literal output, unchanged by the deletion) lives in
``test_evaluate.py`` against ``files/l23_gordon_fixture.npz``.

M2 task 2: observation-tuple geometry threading (design §3.2) -- the
``(Rrs, varRrs, params, idx[, geom])`` tuple through ``inference.fit_one``
(via emcee's positional ``args``), ``chisq_fit.fit`` (via the ``fit_func``
partial), and ``fit_batch`` with mixed 4-/5-tuples. Legacy 4-tuples are
pinned to deliver exactly the pre-change argument set (``geom=None``).

M2 task 3: setup validation + domain-check wiring -- ``validate_rt_dict``
raises through ``fit_one``/``chisq_fit.fit`` at setup, before any
sampling/optimizer work (the CQ4 ``theta_s`` message for a robust backend
with a legacy 4-tuple; the robust_hybrid out-of-range grid); the un-jitted
``robust_domain_check`` fires ``DomainWarning`` through both fitters on a
deliberately turbid initial guess (robust_hybrid, ``Bp_value=0.005``); the
Gordon backend never invokes the domain check at all (it would raise --
pinned with a bomb monkeypatch); and ``fit_batch``'s robust-raises half of
Gate item 6.

M3 task 1: the ``fit_Bp`` B_p tail peel + forward flow (design §3.3) --
``log_prob``/``fit_func`` peel the trailing B_p off the parameter vector
when ``rt_dict['fit_Bp']`` is True and forward it as the adapter's ``Bp``
(pinned by exact equivalence against the fixed-``Bp_value`` path);
``fit_Bp`` False/absent is pinned byte-identical to the M2 dispatch (same
values, no peel); the peeled tail also reaches the setup-time
``robust_domain_check`` calls in both fitters (p0 in both, posterior
median in ``fit_one``); and ``robust_baseline`` accepts a free B_p but the
value is inert (``robust.rt.baselines.Rrs_gordon`` discards
``phase_params`` by construction). The B_p *prior* and the ndim/walker
bookkeeping are M3 tasks 2/3, tested there.

M3 task 2: the free-B_p prior + p0 seeding (plan choice, design §7.3) --
``log_prob`` evaluates ``inference.BP_PRIOR`` (a linear-space uniform over
the inclusive [rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX] =
[0.004, 0.05]) on the peeled tail alongside the model priors: in-range
tails (bounds included) reach the forward call and give a finite
log-probability; out-of-range tails return -inf *without* the forward
model ever being invoked (pinned with a bomb monkeypatch on the adapter);
``fit_Bp`` False/absent never touches the B_p prior at all (bomb on
``BP_PRIOR.calc``). ``append_Bp_seed`` tails an initial guess with
rt_dict['Bp_value'] (default 0.01 -- nonzero and inside the prior) only
under ``fit_Bp``. The chi-squared path deliberately has no in-``fit_func``
range check -- B_p bounds ride curve_fit's ``bounds`` like every model
parameter (built in l23.fit_with_LM from the same rt_defs constants).

M3 task 3: the chain bookkeeping (design §3.3) -- ``init_mcmc``'s optional
``rt_dict`` (+1 ndim under ``fit_Bp``; the 'ndim' key), the
``prior_bounds``/``init_walkers`` B_p clip slot (from the rt_defs
constants), the synthetic ``robust_ztt`` round-trip at B_p=0.02 (Gate
items 1-2: extra chain column, posterior median/CI recover the truth,
prior edges excluded), ``chain_param_names``/``calc_stats`` names ending
in 'B_p' (Gate 3), ``reconstruct_from_chains``'s tail strip + backend
dispatch + Bp forwarding (Gate 4, pinned by task-1-style equivalence),
corner-plot 'B_p' labeling with ``log_param_mask`` semantics False
(linear), the fitter-level gordon+``fit_Bp`` setup rejection (Gate 5),
and Gate 6's fixed-B_p regression against the **provisional** pin
``files/m3_fixed_bp_pin.npz`` (M2 Q5 never made a real one; see the
fixture generator's docstring).
"""
import numpy as np

import pytest

from bing.models import utils as model_utils
from bing.parameters import standard
from bing.fitting import inference as bing_inf
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs
from bing.rt.geometry import ObsGeometry
from bing import evaluate


# ===== Task 1: the dependency =====

def test_import_robust_rt():
    """robust.rt is importable and exposes the public forward() entry point."""
    import robust.rt

    assert hasattr(robust.rt, 'forward')
    assert callable(robust.rt.forward)


# ===== Task 2: rt_dict_from_p's new keys =====

def test_rt_dict_from_p_robust_keys_default_to_gordon():
    """A legacy-style p (no rt_backend/fit_Bp/Bp_value attrs) yields the
    Gordon-backend defaults, not None -- see also test_chl_fl.py's
    ``test_rt_dict_from_p_missing_attrs`` for the same check from the
    'entirely minimal namedtuple' side."""
    from bing.parameters import p_ntuple

    p = p_ntuple.gen(model_names=['ExpBricaud', 'Pow'])
    rt_dict = rt_defs.rt_dict_from_p(p)

    assert rt_dict['rt_backend'] == 'gordon'
    assert rt_dict['fit_Bp'] is False
    assert rt_dict['Bp_value'] == 0.01


def test_rt_dict_from_p_robust_keys_explicit():
    """rt_dict_from_p propagates explicit rt_backend/fit_Bp/Bp_value when a
    p object actually carries them."""
    from collections import namedtuple

    Custom = namedtuple('Custom', ['model_names', 'rt_backend', 'fit_Bp', 'Bp_value'])
    p = Custom(model_names=['ExpBricaud', 'Pow'], rt_backend='robust_hybrid',
               fit_Bp=True, Bp_value=0.02)
    rt_dict = rt_defs.rt_dict_from_p(p)

    assert rt_dict['rt_backend'] == 'robust_hybrid'
    assert rt_dict['fit_Bp'] is True
    assert rt_dict['Bp_value'] == 0.02


# ===== Task 2: validate_rt_dict =====

class _FakeModel:
    """Minimal stand-in for an a-model/bb-model, carrying only what
    validate_rt_dict's grid check reads."""
    def __init__(self, wave):
        self.wave = np.asarray(wave)


class _FakeGeom:
    """Minimal stand-in for ObsGeometry -- validate_rt_dict only checks
    ``geom is None``, so any non-None sentinel exercises the same path."""
    pass


def test_validate_rt_dict_passes_on_legal_gordon():
    """The default Gordon rt_dict, with no models/geom, is legal."""
    rt_dict = {'rt_backend': 'gordon', 'fit_Bp': False}
    rt_defs.validate_rt_dict(rt_dict)  # should not raise


def test_validate_rt_dict_passes_on_legal_robust():
    """A robust backend with geom supplied and an in-range grid is legal."""
    rt_dict = {'rt_backend': 'robust_hybrid', 'fit_Bp': False}
    models = [_FakeModel(np.linspace(400., 700., 61)), None]
    rt_defs.validate_rt_dict(rt_dict, models=models, geom=_FakeGeom())  # no raise


def test_validate_rt_dict_rejects_unknown_backend():
    rt_dict = {'rt_backend': 'not_a_real_backend'}
    with pytest.raises(ValueError, match='rt_backend'):
        rt_defs.validate_rt_dict(rt_dict)


def test_validate_rt_dict_rejects_fit_Bp_with_gordon():
    rt_dict = {'rt_backend': 'gordon', 'fit_Bp': True}
    with pytest.raises(ValueError, match='fit_Bp'):
        rt_defs.validate_rt_dict(rt_dict)


@pytest.mark.parametrize('rt_backend', ['robust_ztt', 'robust_hybrid', 'robust_baseline'])
def test_validate_rt_dict_rejects_robust_backend_without_geom(rt_backend):
    rt_dict = {'rt_backend': rt_backend}
    with pytest.raises(ValueError, match='geometry'):
        rt_defs.validate_rt_dict(rt_dict, geom=None)


def test_validate_rt_dict_rejects_hybrid_grid_out_of_range():
    rt_dict = {'rt_backend': 'robust_hybrid'}
    # 760 nm is outside the emulator's [350, 750] nm training range.
    models = [_FakeModel(np.linspace(400., 760., 73)), None]
    with pytest.raises(ValueError, match='robust_hybrid'):
        rt_defs.validate_rt_dict(rt_dict, models=models, geom=_FakeGeom())


def test_validate_rt_dict_ztt_and_baseline_accept_any_grid():
    """Only robust_hybrid is grid-restricted; ztt/baseline are grid-agnostic
    (docs/design/rob_rt_design.md §4)."""
    wide_grid = [_FakeModel(np.linspace(300., 900., 61)), None]
    for rt_backend in ('robust_ztt', 'robust_baseline'):
        rt_dict = {'rt_backend': rt_backend}
        rt_defs.validate_rt_dict(rt_dict, models=wide_grid, geom=_FakeGeom())  # no raise


def test_validate_rt_dict_skips_grid_check_without_models():
    """models=None skips the grid check rather than raising -- it has
    nothing to check against."""
    rt_dict = {'rt_backend': 'robust_hybrid'}
    rt_defs.validate_rt_dict(rt_dict, models=None, geom=_FakeGeom())  # no raise


# ===== Task 3: ObsGeometry =====

def test_obsgeometry_requires_theta_s():
    """theta_s has no default -- omitting it is a TypeError, not a silent
    guess (CQ4: theta_s is never defaulted)."""
    with pytest.raises(TypeError):
        ObsGeometry()


def test_obsgeometry_nadir_defaults():
    """theta_v/dphi/wind default to nadir viewing when only theta_s is given."""
    g = ObsGeometry(theta_s=30.)
    assert g.theta_s == 30.
    assert g.theta_v == 0.0
    assert g.dphi == 0.0
    assert g.wind is None


def test_obsgeometry_is_frozen():
    g = ObsGeometry(theta_s=30.)
    with pytest.raises(Exception):
        g.theta_s = 99.


def test_obsgeometry_to_robust_roundtrip_nadir():
    """ObsGeometry(theta_s=30.) round-trips through to_robust() into an
    equivalent robust.rt.types.Geometry (values, degrees)."""
    from robust.rt.types import Geometry as RobustGeometry

    g = ObsGeometry(theta_s=30.)
    rg = g.to_robust()

    assert isinstance(rg, RobustGeometry)
    assert rg.theta_s == 30.
    assert rg.theta_v == 0.0
    assert rg.dphi == 0.0
    assert rg.wind is None
    assert rg.Ed is None


def test_obsgeometry_to_robust_roundtrip_non_nadir_and_ed():
    """Non-nadir viewing geometry and the wind field round-trip too; Ed is
    a to_robust() pass-through keyword, not a stored ObsGeometry field."""
    g = ObsGeometry(theta_s=45., theta_v=10., dphi=90., wind=5.)
    ed_pair = (np.array([400., 500.]), np.array([1.0, 0.9]))

    rg = g.to_robust(Ed=ed_pair)

    assert rg.theta_s == 45.
    assert rg.theta_v == 10.
    assert rg.dphi == 90.
    assert rg.wind == 5.
    assert rg.Ed is ed_pair

    # Ed is never stored on ObsGeometry itself.
    assert not hasattr(g, 'Ed')


# ===== M1 task 1: calc_Rrs_from_models_robust =====

# Three distinct water types (Chl-driven a_ph amplitude, independent Adg/Sdg/
# Bnw/beta) on a real BING model pair -- not L23-loaded truth, since these
# tests exercise the adapter's mapping logic, not L23 data loading, and a
# synthetic fixture needs no external data path ($OS_COLOR) to run anywhere.
_PARAM_SETS = [
    dict(Chl=1.0, a=[-1.5, 0.017], bb=[-3.0, 1.0]),
    dict(Chl=0.3, a=[-2.2, 0.014], bb=[-3.5, 0.5]),
    dict(Chl=5.0, a=[-0.8, 0.020], bb=[-2.2, 1.3]),
]


@pytest.fixture()
def robust_models():
    """A fresh ExpBricaud + Pow model pair on a robust_hybrid-legal grid."""
    wave = np.linspace(400., 700., 61)
    return model_utils.init(['ExpBricaud', 'Pow'], wave)


def _param_vector(ps):
    a_params = np.array(ps['a'] + [np.log10(0.05582 * ps['Chl'])])
    bb_params = np.array(ps['bb'])
    return a_params, bb_params


def test_calc_Rrs_from_models_robust_baseline_parity(robust_models):
    """robust_baseline matches calc_Rrs_from_models (elastic Gordon) at
    rtol <= 1e-5 on 3 distinct water types -- both use G1=0.0949/G2=0.0794
    and the same A_Rrs/B_Rrs conversion; float32 sets the tolerance
    (measured worst: ~3e-7, comfortably inside the gate)."""
    a_model, bb_model = robust_models
    rt_dict_robust = {'rt_backend': 'robust_baseline', 'include_Raman': False,
                      'include_Chl_fl': False, 'phi_C': 0.02,
                      'double_gaussian': True, 'Bp_value': 0.014}
    rt_dict_gordon = {'variable_Gordon': False, 'include_Raman': False,
                       'include_Chl_fl': False}
    geom = ObsGeometry(theta_s=30.)

    for ps in _PARAM_SETS:
        a_model.set_aph(np.array([ps['Chl']]))
        a_params, bb_params = _param_vector(ps)

        Rrs_robust = evaluate.calc_Rrs_from_models_robust(
            a_model, a_params, bb_model, bb_params, rt_dict_robust, geom=geom)
        Rrs_gordon = evaluate.calc_Rrs_from_models(
            a_model, a_params, bb_model, bb_params, rt_dict_gordon)

        # calc_Rrs_from_models preserves eval_a/eval_bb's (1, nwave) batch
        # axis for 1-D params too (see test_evaluate.py's own note on this);
        # squeeze both sides to compare like shapes.
        np.testing.assert_allclose(np.squeeze(Rrs_robust), np.squeeze(Rrs_gordon),
                                   rtol=1e-5)


@pytest.mark.parametrize('rt_backend', ['robust_ztt', 'robust_hybrid', 'robust_baseline'])
def test_calc_Rrs_from_models_robust_shapes(robust_models, rt_backend):
    """(nparam,) -> (1, nwave); (nsamples, nparam) -> (nsamples, nwave);
    finite throughout, for every robust backend."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': rt_backend, 'include_Raman': False,
               'include_Chl_fl': False, 'phi_C': 0.02, 'double_gaussian': True,
               'Bp_value': 0.014}
    geom = ObsGeometry(theta_s=30.)

    Rrs_1d = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    assert Rrs_1d.shape == (1, len(a_model.wave))
    assert np.all(np.isfinite(Rrs_1d))

    nsample = 4
    a_batch = np.tile(a_params, (nsample, 1))
    bb_batch = np.tile(bb_params, (nsample, 1))
    Rrs_batch = evaluate.calc_Rrs_from_models_robust(
        a_model, a_batch, bb_model, bb_batch, rt_dict, geom=geom)
    assert Rrs_batch.shape == (nsample, len(a_model.wave))
    assert np.all(np.isfinite(Rrs_batch))


def test_calc_Rrs_from_models_robust_full_return(robust_models):
    """full_return=True returns (Rrs, a, bb), matching calc_Rrs_from_models'
    convention."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': 'robust_ztt', 'include_Raman': False,
               'include_Chl_fl': False, 'phi_C': 0.02, 'double_gaussian': True,
               'Bp_value': 0.014}
    geom = ObsGeometry(theta_s=30.)

    Rrs, a, bb = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom,
        full_return=True)
    nwave = len(a_model.wave)
    assert Rrs.shape == (1, nwave)
    assert a.shape == (1, nwave)
    assert bb.shape == (1, nwave)


def test_calc_Rrs_from_models_robust_raman_branch(robust_models):
    """include_Raman=True runs and produces a finite, physically plausible
    Rrs when the a-model has an Ed spectrum set."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_model.set_raman_Ed(np.array([350., 750.]), np.array([1.0, 1.0]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': 'robust_ztt', 'include_Raman': True,
               'include_Chl_fl': False, 'phi_C': 0.02, 'double_gaussian': True,
               'Bp_value': 0.014}
    geom = ObsGeometry(theta_s=30.)

    Rrs = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    assert np.all(np.isfinite(Rrs))
    assert np.all(Rrs > 0)


def test_calc_Rrs_from_models_robust_fluorescence_adds_emission(robust_models):
    """include_Chl_fl=True adds a (net strictly positive) contribution
    relative to the elastic-only Rrs (fluorescence is additive, design
    §3.5). The elastic backbone itself is recomputed along a different
    static code path when Inelastic is set (even with fluorescence's own
    kernel ~0 far from the 685 nm peak), so a few wavelengths differ by
    float32 rounding noise (~1e-10, measured) rather than the kernel itself
    -- hence the small atol rather than a bare >=; the peak-region check
    below confirms the real signal isn't just noise."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)

    rt_dict_elastic = {'rt_backend': 'robust_ztt', 'include_Raman': False,
                        'include_Chl_fl': False, 'phi_C': 0.02,
                        'double_gaussian': True, 'Bp_value': 0.014}
    rt_dict_fl = dict(rt_dict_elastic, include_Chl_fl=True)

    Rrs_elastic = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict_elastic, geom=geom)
    Rrs_fl = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict_fl, geom=geom)

    assert np.all(Rrs_fl >= Rrs_elastic - 1e-8)
    assert (Rrs_fl - Rrs_elastic).max() > 1e-5  # a real signal, not just noise


def test_calc_Rrs_from_models_robust_free_Bp_changes_result(robust_models):
    """Bp overrides rt_dict['Bp_value'] when given."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': 'robust_ztt', 'include_Raman': False,
               'include_Chl_fl': False, 'phi_C': 0.02, 'double_gaussian': True,
               'Bp_value': 0.014}
    geom = ObsGeometry(theta_s=30.)

    Rrs_default = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    Rrs_override = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom, Bp=0.02)

    assert not np.allclose(Rrs_default, Rrs_override)


def test_calc_Rrs_from_models_robust_requires_geom(robust_models):
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': 'robust_ztt', 'Bp_value': 0.014}

    with pytest.raises(ValueError, match='geom'):
        evaluate.calc_Rrs_from_models_robust(
            a_model, a_params, bb_model, bb_params, rt_dict, geom=None)


def test_calc_Rrs_from_models_robust_rejects_gordon_backend(robust_models):
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    rt_dict = {'rt_backend': 'gordon', 'Bp_value': 0.014}
    geom = ObsGeometry(theta_s=30.)

    with pytest.raises(ValueError, match='robust backend'):
        evaluate.calc_Rrs_from_models_robust(
            a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)


def test_calc_Rrs_from_models_robust_baseline_rejects_inelastic(robust_models):
    """robust_baseline has no inelastic composition path -- requesting Raman
    or fluorescence with it must raise, not silently drop the term."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)

    for flag in ('include_Raman', 'include_Chl_fl'):
        rt_dict = {'rt_backend': 'robust_baseline', 'include_Raman': False,
                   'include_Chl_fl': False, 'phi_C': 0.02,
                   'double_gaussian': True, 'Bp_value': 0.014}
        rt_dict[flag] = True
        with pytest.raises(ValueError, match='robust_baseline'):
            evaluate.calc_Rrs_from_models_robust(
                a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)


# ===== M1 task 2: the JIT strategy =====

@pytest.mark.parametrize('rt_backend', ['robust_ztt', 'robust_hybrid', 'robust_baseline'])
def test_robust_forward_jit_cache_hits_on_repeat_config(robust_models, rt_backend):
    """A second call with an identical (mode, inelastic, wave) config hits
    the lru_cache -- no rebuild of the jax.jit closure, and (checked via
    the jitted function's own compile-cache size) no XLA recompile
    either."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)
    rt_dict = {'rt_backend': rt_backend, 'include_Raman': False,
               'include_Chl_fl': False, 'phi_C': 0.02, 'double_gaussian': True,
               'Bp_value': 0.014}

    evaluate._robust_forward_jit.cache_clear()
    evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    info_after_first = evaluate._robust_forward_jit.cache_info()
    assert info_after_first.misses == 1
    assert info_after_first.hits == 0

    evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    info_after_second = evaluate._robust_forward_jit.cache_info()
    assert info_after_second.misses == 1  # no new build
    assert info_after_second.hits == 1

    # The underlying jax.jit closure itself has compiled exactly once --
    # not just that we skipped rebuilding it.
    mode = 'baseline' if rt_backend == 'robust_baseline' else rt_backend[len('robust_'):]
    wave_key = np.asarray(a_model.wave, dtype=np.float64).tobytes()
    jit_fn = evaluate._robust_forward_jit(mode, None, wave_key)
    assert jit_fn._cache_size() == 1


def test_robust_forward_jit_separate_configs_do_not_collide(robust_models):
    """Different rt_backend values (hence different `mode`) get distinct
    cache entries, never sharing a compiled closure."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)

    evaluate._robust_forward_jit.cache_clear()
    for rt_backend in ('robust_ztt', 'robust_hybrid', 'robust_baseline'):
        rt_dict = {'rt_backend': rt_backend, 'include_Raman': False,
                   'include_Chl_fl': False, 'phi_C': 0.02,
                   'double_gaussian': True, 'Bp_value': 0.014}
        evaluate.calc_Rrs_from_models_robust(
            a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)

    info = evaluate._robust_forward_jit.cache_info()
    assert info.misses == 3
    assert info.currsize == 3


def test_robust_forward_jit_none_vs_instance_inelastic_are_distinct_entries(robust_models):
    """inelastic_key=None (elastic-only) and an actual (raman/fluorescence)
    key are cached separately -- Inelastic(raman=False, fluorescence=False)
    is not the same code path as inelastic=None (design §3.5)."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)

    evaluate._robust_forward_jit.cache_clear()
    rt_dict_elastic = {'rt_backend': 'robust_ztt', 'include_Raman': False,
                       'include_Chl_fl': False, 'phi_C': 0.02,
                       'double_gaussian': True, 'Bp_value': 0.014}
    rt_dict_raman = dict(rt_dict_elastic, include_Raman=True)

    evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict_elastic, geom=geom)
    evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict_raman, geom=geom)

    info = evaluate._robust_forward_jit.cache_info()
    assert info.misses == 2
    assert info.currsize == 2


# ===== M1 task 3: robust_domain_check (the un-jitted domain check) =====

# A deliberately out-of-domain configuration: B_p = 0.005 is well below the
# emulator's trained lower bound (~0.0103 -- the same breach the task-1 log
# hit incidentally at Bp_value=0.01/theta_s=30 deg before bumping the shape
# tests to 0.014). Everything else stays at the in-domain values used above,
# so the warning is attributable to B_p alone.
_TURBID_RT_DICT = {'rt_backend': 'robust_hybrid', 'include_Raman': False,
                   'include_Chl_fl': False, 'phi_C': 0.02,
                   'double_gaussian': True, 'Bp_value': 0.005}


def _domain_check_args(robust_models):
    """Model pair + params + geom shared by the task-3 tests."""
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)
    geom = ObsGeometry(theta_s=30.)
    return a_model, a_params, bb_model, bb_params, geom


def test_robust_domain_check_turbid_fires_domain_warning(robust_models):
    """Gate item 5, first half: on a deliberately out-of-domain IOP set the
    un-jitted check emits robust's DomainWarning -- the whole reason the
    helper exists, since the jitted hot path can never warn (the domain
    check needs concrete values and is skipped for traced inputs)."""
    from robust.rt.hybrid import DomainWarning

    a_model, a_params, bb_model, bb_params, geom = _domain_check_args(robust_models)

    with pytest.warns(DomainWarning, match='outside its training range'):
        evaluate.robust_domain_check(
            a_model, a_params, bb_model, bb_params, _TURBID_RT_DICT, geom=geom)


def test_robust_domain_check_jitted_path_same_inputs_no_error(robust_models):
    """Gate item 5, second half: the jitted hot path on the *same*
    out-of-domain inputs neither errors nor warns -- it silently produces a
    finite value (warn-and-continue, design §4 / Q8: the fit proceeds; the
    diagnosis belongs to robust_domain_check, outside the hot loop)."""
    import warnings as _warnings

    from robust.rt.hybrid import DomainWarning

    a_model, a_params, bb_model, bb_params, geom = _domain_check_args(robust_models)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        Rrs = evaluate.calc_Rrs_from_models_robust(
            a_model, a_params, bb_model, bb_params, _TURBID_RT_DICT, geom=geom)

    assert np.all(np.isfinite(Rrs))
    assert [w for w in caught if issubclass(w.category, DomainWarning)] == []


def test_robust_domain_check_in_domain_is_silent(robust_models):
    """An in-domain configuration (the Bp_value=0.014 the rest of this
    module uses precisely because it is in-domain) emits nothing, and the
    helper returns None -- it is a diagnostic, not a forward model."""
    import warnings as _warnings

    from robust.rt.hybrid import DomainWarning

    a_model, a_params, bb_model, bb_params, geom = _domain_check_args(robust_models)
    rt_dict = dict(_TURBID_RT_DICT, Bp_value=0.014)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        result = evaluate.robust_domain_check(
            a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)

    assert result is None
    assert [w for w in caught if issubclass(w.category, DomainWarning)] == []


@pytest.mark.parametrize('rt_backend', ['robust_ztt', 'robust_baseline'])
def test_robust_domain_check_noop_for_ztt_and_baseline(robust_models, rt_backend):
    """Only robust_hybrid has a trained domain: forward()'s check sits past
    the mode='ztt' early return, and baselines.Rrs_gordon has no emulator or
    domain logic at all (both confirmed by reading robust's source). For the
    other backends the helper validates arguments and returns silently, even
    on the out-of-domain B_p."""
    import warnings as _warnings

    from robust.rt.hybrid import DomainWarning

    a_model, a_params, bb_model, bb_params, geom = _domain_check_args(robust_models)
    rt_dict = dict(_TURBID_RT_DICT, rt_backend=rt_backend)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        result = evaluate.robust_domain_check(
            a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)

    assert result is None
    assert [w for w in caught if issubclass(w.category, DomainWarning)] == []


def test_robust_domain_check_shares_adapter_error_paths(robust_models):
    """robust_domain_check builds its call arguments through the same
    _build_robust_inputs as the adapter, so a direct call fails identically:
    missing geom and a non-robust backend raise the same ValueErrors."""
    a_model, a_params, bb_model, bb_params, geom = _domain_check_args(robust_models)

    with pytest.raises(ValueError, match='geom'):
        evaluate.robust_domain_check(
            a_model, a_params, bb_model, bb_params, _TURBID_RT_DICT, geom=None)

    with pytest.raises(ValueError, match='robust backend'):
        evaluate.robust_domain_check(
            a_model, a_params, bb_model, bb_params,
            dict(_TURBID_RT_DICT, rt_backend='gordon'), geom=geom)


# ===== M1 task 4: RT_correction is gone (Gate item 6) =====

def test_calc_Rrs_from_models_ignores_stale_RT_correction_key(robust_models):
    """An rt_dict carrying a stale ``RT_correction`` key is silently ignored:
    no multiplication, no KeyError (Gate item 6; design §6).

    Decisive by construction: before the deletion, this exact key (a uniform
    factor of 2) doubled Rrs on both the 1-D and batch paths -- verified live
    on the pre-deletion code -- so bit-identical output with and without the
    key proves the block is gone, not merely dormant. The companion pin that
    the deletion changed nothing for rt_dicts *without* the key is
    test_evaluate.py's l23_gordon_fixture suite.
    """
    a_model, bb_model = robust_models
    ps = _PARAM_SETS[0]
    a_model.set_aph(np.array([ps['Chl']]))
    a_params, bb_params = _param_vector(ps)

    rt_dict = {'variable_Gordon': False, 'include_Raman': False,
               'include_Chl_fl': False}
    rt_dict_stale = dict(rt_dict,
                         RT_correction=np.full(len(a_model.wave), 2.0))

    # 1-D parameter vector
    Rrs = evaluate.calc_Rrs_from_models(
        a_model, a_params, bb_model, bb_params, rt_dict)
    Rrs_stale = evaluate.calc_Rrs_from_models(
        a_model, a_params, bb_model, bb_params, rt_dict_stale)
    np.testing.assert_array_equal(Rrs_stale, Rrs)

    # Batch -- the deleted block had a separate np.outer branch for ndim == 2.
    a_batch = np.tile(a_params, (3, 1))
    bb_batch = np.tile(bb_params, (3, 1))
    Rrs_b = evaluate.calc_Rrs_from_models(
        a_model, a_batch, bb_model, bb_batch, rt_dict)
    Rrs_b_stale = evaluate.calc_Rrs_from_models(
        a_model, a_batch, bb_model, bb_batch, rt_dict_stale)
    np.testing.assert_array_equal(Rrs_b_stale, Rrs_b)


# ===== M2 task 2: observation-tuple geometry threading (design §3.2) =====
#
# The observation tuple is now (Rrs, varRrs, params, idx[, geom]).  These
# tests pin two things: (a) legacy 4-tuples deliver exactly the pre-change
# argument set to log_prob/fit_func (identity on every slot, geom=None), and
# (b) a 5-tuple's geom object -- the very instance, not a copy -- arrives in
# log_prob on every emcee walker evaluation (by value through the sampler's
# positional ``args`` list) and in fit_func on every curve_fit evaluation
# (through the ``partial``).  The Gordon backend is used throughout so the
# tests stay cheap and deterministic; robust-backend end-to-end smoke fits
# are the milestone Gate's item 1 (task 3).

# ExpBricaud (Adg, Sdg, Aph) + Pow (Bnw, beta) -- the same cheap synthetic
# recipe as test_chisq_fit.py; no external data tree needed.
_THREAD_TRUTH = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
_THREAD_P0 = np.array([-0.9, 0.014, -0.6, -1.9, 0.9])


@pytest.fixture()
def threading_setup():
    """A noiseless synthetic Gordon spectrum plus everything the fitters need.

    Returns
    -------
    dict
        models, rt_dict, Rrs, varRrs and an MCMC pdict (tiny nsteps/nburn,
        ``Chl`` indexed for tuple idx 0 and 1, per fit_one's lookup).
    """
    p = standard.expb_pow(wv_min=400., wv_max=700., variable_Gordon=False)
    wave = np.arange(400., 705., 5.)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    rt_dict = rt_defs.rt_dict_from_p(p)

    Rrs = chisq_fit.fit_func(wave, *_THREAD_TRUTH, models=models,
                             rt_dict=rt_dict)
    pdict = bing_inf.init_mcmc(models, nsteps=30, nburn=10)
    pdict['Chl'] = np.array([1.0, 1.0])   # indexed by the tuple's idx
    pdict['Y'] = None
    return dict(models=models, rt_dict=rt_dict, Rrs=Rrs,
                varRrs=(0.02*Rrs)**2, pdict=pdict)


def _recording_log_prob(record):
    """A log_prob wrapper recording the positional args emcee delivers.

    Built *before* monkeypatching so ``real`` binds the original; emcee's
    _FunctionWrapper calls ``f(x, *args)``, so everything after ``params``
    arrives positionally -- which is exactly what these tests pin.
    """
    real = bing_inf.log_prob

    def wrapper(params, models, Rrs, varRrs, rt_dict, geom=None):
        record.append((models, Rrs, varRrs, rt_dict, geom))
        return real(params, models, Rrs, varRrs, rt_dict, geom=geom)
    return wrapper


def test_fit_one_legacy_4tuple_args_unchanged(threading_setup, monkeypatch):
    """A 4-tuple still delivers the pre-change argument set to log_prob:
    the same objects (identity) in the same positional slots, geom=None."""
    ts = threading_setup
    record = []
    monkeypatch.setattr(bing_inf, 'log_prob', _recording_log_prob(record))

    np.random.seed(1234)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0)
    chains, idx = bing_inf.fit_one(items, models=ts['models'],
                                   pdict=ts['pdict'], chains_only=True,
                                   rt_dict=ts['rt_dict'])
    assert idx == 0
    assert chains.shape == (30, ts['pdict']['nwalkers'], _THREAD_P0.size)
    assert np.all(np.isfinite(chains))

    assert len(record) > 0
    for models_r, Rrs_r, varRrs_r, rt_r, geom_r in record:
        assert models_r is ts['models']
        assert Rrs_r is ts['Rrs']
        assert varRrs_r is ts['varRrs']
        assert rt_r is ts['rt_dict']
        assert geom_r is None


def test_fit_one_5tuple_geom_reaches_log_prob(threading_setup, monkeypatch):
    """The 5th element rides emcee's positional ``args`` into log_prob's
    trailing geom slot -- the identical object, on every walker evaluation,
    never as a sampled dimension (chain width stays nparam)."""
    ts = threading_setup
    record = []
    monkeypatch.setattr(bing_inf, 'log_prob', _recording_log_prob(record))
    geom = ObsGeometry(theta_s=30.)

    np.random.seed(1234)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 1, geom)
    chains, idx = bing_inf.fit_one(items, models=ts['models'],
                                   pdict=ts['pdict'], chains_only=True,
                                   rt_dict=ts['rt_dict'])
    assert idx == 1
    assert chains.shape == (30, ts['pdict']['nwalkers'], _THREAD_P0.size)
    assert np.all(np.isfinite(chains))

    assert len(record) > 0
    assert all(rec[4] is geom for rec in record)
    # ... and no slot shifted: Rrs is still Rrs
    assert all(rec[1] is ts['Rrs'] for rec in record)


def test_chisq_fit_4tuple_and_5tuple_none_identical(threading_setup):
    """chisq_fit.fit: a trailing None is a strict no-op (curve_fit is
    deterministic, so the answers must be bit-identical), and the legacy
    4-tuple still recovers the noiseless truth."""
    ts = threading_setup
    legacy = chisq_fit.fit((ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0),
                           ts['models'], ts['rt_dict'])
    with_none = chisq_fit.fit(
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0, None),
        ts['models'], ts['rt_dict'])
    np.testing.assert_array_equal(with_none[0], legacy[0])
    np.testing.assert_array_equal(with_none[1], legacy[1])
    assert legacy[2] == with_none[2] == 0
    np.testing.assert_allclose(legacy[0], _THREAD_TRUTH, atol=1e-3)


def test_chisq_fit_5tuple_geom_reaches_fit_func(threading_setup, monkeypatch):
    """The 5th element reaches fit_func's geom keyword -- the identical
    object, on every optimizer evaluation -- via the partial in fit."""
    ts = threading_setup
    record = []
    real = chisq_fit.fit_func

    def wrapper(wave, *params, models=None, rt_dict=None, geom=None, **kw):
        record.append(geom)
        return real(wave, *params, models=models, rt_dict=rt_dict,
                    geom=geom, **kw)
    monkeypatch.setattr(chisq_fit, 'fit_func', wrapper)

    geom = ObsGeometry(theta_s=30.)
    ans, cov, idx = chisq_fit.fit(
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 3, geom),
        ts['models'], ts['rt_dict'])
    assert idx == 3
    assert len(record) > 0
    assert all(g is geom for g in record)
    # Gordon ignores geom, so the fit itself is unperturbed
    np.testing.assert_allclose(ans, _THREAD_TRUTH, atol=1e-3)


def test_fit_batch_mixed_4_and_5_tuples_gordon(threading_setup):
    """fit_batch forwards each tuple to fit_one unchanged, so a mixed list
    of 4- and 5-tuples runs under the Gordon backend (the robust-raises
    half of Gate item 6 lands with task 3's setup validation)."""
    ts = threading_setup
    items = [
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0),
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 1,
         ObsGeometry(theta_s=30.)),
    ]
    chains, idxs = bing_inf.fit_batch(ts['models'], ts['pdict'], items,
                                      ts['rt_dict'], n_cores=1)
    assert chains.shape == (2, 30, ts['pdict']['nwalkers'],
                            _THREAD_P0.size)
    assert set(idxs.tolist()) == {0, 1}
    assert np.all(np.isfinite(chains))


# ===== M2 task 3: setup validation + domain-check wiring =====
#
# validate_rt_dict (M0) now runs once per fit inside fit_one/chisq_fit.fit,
# immediately after the tuple unpack -- so the CQ4 theta_s error and the
# robust_hybrid grid error surface at setup, before any sampling/optimizer
# work (pinned below with sentinel monkeypatches on run_emcee/curve_fit).
# robust_domain_check (M1) runs un-jitted on the initial guess in both
# fitters, and again on the posterior median in fit_one -- for robust
# backends only: called with rt_backend='gordon' it raises ValueError
# ("not a robust backend", via _build_robust_inputs -- verified above in
# test_robust_domain_check_shares_adapter_error_paths), so the fitters gate
# it on a non-gordon backend rather than calling it unconditionally.


@pytest.mark.parametrize('rt_backend',
                         ['robust_ztt', 'robust_hybrid', 'robust_baseline'])
def test_fit_one_robust_4tuple_raises_theta_s_at_setup(
        threading_setup, monkeypatch, rt_backend):
    """CQ4 as behavior (Gate item 3): a robust-backend fit_one with a legacy
    4-tuple (geom=None) raises at setup with a message naming theta_s --
    and run_emcee is never reached (sentinel), so no MCMC work happens."""
    ts = threading_setup
    called = []
    monkeypatch.setattr(bing_inf, 'run_emcee',
                        lambda *a, **k: called.append(1))

    rt_dict = dict(ts['rt_dict'], rt_backend=rt_backend)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0)
    with pytest.raises(ValueError, match='theta_s'):
        bing_inf.fit_one(items, models=ts['models'], pdict=ts['pdict'],
                         chains_only=True, rt_dict=rt_dict)
    assert called == []


@pytest.mark.parametrize('rt_backend',
                         ['robust_ztt', 'robust_hybrid', 'robust_baseline'])
def test_chisq_fit_robust_4tuple_raises_theta_s_at_setup(
        threading_setup, monkeypatch, rt_backend):
    """Same CQ4 check through chisq_fit.fit: the theta_s ValueError fires
    before curve_fit is ever called (sentinel)."""
    ts = threading_setup
    called = []
    monkeypatch.setattr(chisq_fit, 'curve_fit',
                        lambda *a, **k: called.append(1))

    rt_dict = dict(ts['rt_dict'], rt_backend=rt_backend)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0)
    with pytest.raises(ValueError, match='theta_s'):
        chisq_fit.fit(items, ts['models'], rt_dict)
    assert called == []


def test_fit_one_hybrid_grid_error_at_setup(threading_setup, monkeypatch):
    """M0's robust_hybrid wavelength-grid check now surfaces through
    fit_one at setup (not just from validate_rt_dict directly): a grid
    reaching 760 nm exceeds the emulator's [350, 750] nm training range,
    and run_emcee is never reached."""
    ts = threading_setup
    called = []
    monkeypatch.setattr(bing_inf, 'run_emcee',
                        lambda *a, **k: called.append(1))

    wide_wave = np.arange(400., 765., 5.)   # 400-760 nm inclusive
    wide_models = model_utils.init(['ExpBricaud', 'Pow'], wide_wave)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_hybrid')
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0,
             ObsGeometry(theta_s=30.))
    with pytest.raises(ValueError, match='training range'):
        bing_inf.fit_one(items, models=wide_models, pdict=ts['pdict'],
                         chains_only=True, rt_dict=rt_dict)
    assert called == []


def test_chisq_fit_hybrid_grid_error_at_setup(threading_setup, monkeypatch):
    """Same grid check through chisq_fit.fit, before curve_fit runs."""
    ts = threading_setup
    called = []
    monkeypatch.setattr(chisq_fit, 'curve_fit',
                        lambda *a, **k: called.append(1))

    wide_wave = np.arange(400., 765., 5.)
    wide_models = model_utils.init(['ExpBricaud', 'Pow'], wide_wave)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_hybrid')
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0,
             ObsGeometry(theta_s=30.))
    with pytest.raises(ValueError, match='training range'):
        chisq_fit.fit(items, wide_models, rt_dict)
    assert called == []


def test_fit_one_turbid_hybrid_fires_domain_warning(threading_setup):
    """A real (tiny) robust_hybrid MCMC fit with the deliberately
    out-of-domain Bp_value=0.005 (below the emulator's trained ~0.0103
    lower bound, per M1 task 3) emits DomainWarning through fit_one's
    un-jitted domain checks -- and still returns finite chains
    (warn-and-continue, design §4)."""
    from robust.rt.hybrid import DomainWarning

    ts = threading_setup
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_hybrid',
                   Bp_value=0.005)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0,
             ObsGeometry(theta_s=30.))

    np.random.seed(1234)
    with pytest.warns(DomainWarning, match='outside its training range'):
        chains, idx = bing_inf.fit_one(items, models=ts['models'],
                                       pdict=ts['pdict'], chains_only=True,
                                       rt_dict=rt_dict)
    assert idx == 0
    assert chains.shape == (30, ts['pdict']['nwalkers'], _THREAD_P0.size)
    assert np.all(np.isfinite(chains))


def test_chisq_fit_turbid_hybrid_fires_domain_warning_before_optimizer(
        threading_setup, monkeypatch):
    """The same turbid Bp_value=0.005 under robust_hybrid warns through
    chisq_fit.fit's p0 domain check. curve_fit is stubbed out (it just
    echoes p0), so the DomainWarning can only have come from the setup
    check on the initial guess -- there is no optimizer work at all."""
    from robust.rt.hybrid import DomainWarning

    ts = threading_setup
    called = []

    def fake_curve_fit(f, xdata, ydata, p0=None, **kwargs):
        called.append(1)
        return np.asarray(p0), np.eye(len(p0))
    monkeypatch.setattr(chisq_fit, 'curve_fit', fake_curve_fit)

    rt_dict = dict(ts['rt_dict'], rt_backend='robust_hybrid',
                   Bp_value=0.005)
    items = (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0,
             ObsGeometry(theta_s=30.))
    with pytest.warns(DomainWarning, match='outside its training range'):
        ans, cov, idx = chisq_fit.fit(items, ts['models'], rt_dict)
    assert called == [1]   # the warning preceded the (stubbed) optimizer
    assert idx == 0


def test_gordon_fit_never_calls_domain_check(threading_setup, monkeypatch):
    """The domain check is gated off the Gordon path entirely -- calling
    robust_domain_check with rt_backend='gordon' would raise ValueError
    ("not a robust backend"), so 'never invoked' is load-bearing, not a
    style choice. Pinned with a bomb: both fitters complete a normal
    Gordon fit with robust_domain_check replaced by an AssertionError."""
    ts = threading_setup

    def bomb(*args, **kwargs):
        raise AssertionError(
            'robust_domain_check must never be called for the gordon backend')
    monkeypatch.setattr(evaluate, 'robust_domain_check', bomb)

    np.random.seed(1234)
    chains, idx = bing_inf.fit_one(
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0),
        models=ts['models'], pdict=ts['pdict'], chains_only=True,
        rt_dict=ts['rt_dict'])
    assert np.all(np.isfinite(chains))

    ans, cov, idx = chisq_fit.fit(
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0),
        ts['models'], ts['rt_dict'])
    np.testing.assert_allclose(ans, _THREAD_TRUTH, atol=1e-3)


def test_fit_batch_robust_4tuple_raises_theta_s(threading_setup):
    """Gate item 6, robust half: a robust rt_dict with a legacy 4-tuple in
    the items list raises through fit_batch (the worker's fit_one setup
    validation), with the same theta_s-naming message. The offending
    4-tuple is listed first so the chunk fails before the 5-tuple's real
    robust fit would run."""
    ts = threading_setup
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt')
    items = [
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 0),          # no geom
        (ts['Rrs'], ts['varRrs'], _THREAD_P0.copy(), 1,
         ObsGeometry(theta_s=30.)),
    ]
    with pytest.raises(ValueError, match='theta_s'):
        bing_inf.fit_batch(ts['models'], ts['pdict'], items, rt_dict,
                           n_cores=1)


# ===== M3 task 1: the fit_Bp B_p tail peel + forward flow (design §3.3) =====
#
# When rt_dict['fit_Bp'] is True the combined vector is
# [a_params..., bb_params..., B_p]; log_prob and chisq_fit.fit_func peel the
# tail *before* the aparams/bparams split and forward it as the adapter's
# ``Bp`` argument.  The pins here follow the adapter's own contract
# (Bp overrides rt_dict['Bp_value'] when not None -- see
# test_calc_Rrs_from_models_robust_free_Bp_changes_result): a tailed call
# under fit_Bp=True must be *exactly* equal to the untailed call with
# Bp_value set to the tail value, and fit_Bp False/absent must be exactly
# the M2 dispatch (no peel, Bp=None).  Direct function calls throughout --
# an end-to-end fit_Bp=True fit needs task 3's ndim/init_walkers
# bookkeeping (prior_bounds has no B_p slot yet), the same precedent as
# M2 task 1's direct-call smoke tests.  The B_p tails used here all sit
# inside the task-2 prior range [0.004, 0.05], where the (task 2) prior
# contributes exactly 0 -- these equivalence pins are unchanged by it.

_BP_TAIL = 0.02        # a sentinel distinct from the fixture's Bp_value


def test_log_prob_fit_Bp_peels_tail_and_forwards(threading_setup):
    """fit_Bp=True: log_prob on [params..., B_p] equals log_prob on params
    with rt_dict['Bp_value'] = B_p (exact -- same adapter call), and
    differs from the fixture's own Bp_value (the tail is really used)."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)
    base = dict(ts['rt_dict'], rt_backend='robust_ztt')   # Bp_value=0.01

    tailed = np.append(_THREAD_P0, _BP_TAIL)
    lp_free = bing_inf.log_prob(tailed, ts['models'], ts['Rrs'],
                                ts['varRrs'], dict(base, fit_Bp=True),
                                geom=geom)
    lp_fixed = bing_inf.log_prob(_THREAD_P0, ts['models'], ts['Rrs'],
                                 ts['varRrs'],
                                 dict(base, fit_Bp=False, Bp_value=_BP_TAIL),
                                 geom=geom)
    lp_default = bing_inf.log_prob(_THREAD_P0, ts['models'], ts['Rrs'],
                                   ts['varRrs'], dict(base, fit_Bp=False),
                                   geom=geom)

    assert np.isfinite(lp_free)
    assert lp_free == lp_fixed          # exact: identical adapter inputs
    assert lp_free != lp_default        # the tail value actually flowed


def test_fit_func_fit_Bp_peels_tail_and_forwards(threading_setup):
    """Same equivalence through chisq_fit.fit_func: tailed params under
    fit_Bp=True give exactly the Rrs of the untailed call with
    Bp_value = tail, and not the fixture-default Bp_value's Rrs."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)
    base = dict(ts['rt_dict'], rt_backend='robust_ztt')
    wave = ts['models'][0].wave

    Rrs_free = chisq_fit.fit_func(wave, *_THREAD_P0, _BP_TAIL,
                                  models=ts['models'],
                                  rt_dict=dict(base, fit_Bp=True), geom=geom)
    Rrs_fixed = chisq_fit.fit_func(wave, *_THREAD_P0, models=ts['models'],
                                   rt_dict=dict(base, fit_Bp=False,
                                                Bp_value=_BP_TAIL),
                                   geom=geom)
    Rrs_default = chisq_fit.fit_func(wave, *_THREAD_P0, models=ts['models'],
                                     rt_dict=dict(base, fit_Bp=False),
                                     geom=geom)

    assert Rrs_free.shape == Rrs_fixed.shape == wave.shape
    assert np.array_equal(Rrs_free, Rrs_fixed)
    assert not np.allclose(Rrs_free, Rrs_default)


def test_fit_Bp_false_or_absent_byte_identical_to_m2(threading_setup):
    """The Goals section's hard constraint: fit_Bp False *or absent* takes
    exactly the M2 path -- no peel, Bp=None -> rt_dict['Bp_value'].
    Pinned on both backends by exact equality: (a) 'fit_Bp': False vs the
    key deleted outright, and (b) against the un-dispatched forward calls
    the M2 branches make (calc_Rrs_from_models / ..._robust with Bp=None)."""
    ts = threading_setup
    models = ts['models']
    geom = ObsGeometry(theta_s=30.)
    wave = models[0].wave
    nap = models[0].nparam

    for backend in ['gordon', 'robust_ztt']:
        rt_false = dict(ts['rt_dict'], rt_backend=backend)
        rt_false['fit_Bp'] = False
        rt_absent = {k: v for k, v in rt_false.items() if k != 'fit_Bp'}
        kw = {} if backend == 'gordon' else dict(geom=geom)

        out_false = chisq_fit.fit_func(wave, *_THREAD_P0, models=models,
                                       rt_dict=rt_false, **kw)
        out_absent = chisq_fit.fit_func(wave, *_THREAD_P0, models=models,
                                        rt_dict=rt_absent, **kw)
        assert np.array_equal(out_false, out_absent)

        # ... and both equal the M2 branch's own forward call, unchanged.
        if backend == 'gordon':
            direct = evaluate.calc_Rrs_from_models(
                models[0], _THREAD_P0[:nap], models[1], _THREAD_P0[nap:],
                rt_false)
        else:
            direct = evaluate.calc_Rrs_from_models_robust(
                models[0], _THREAD_P0[:nap], models[1], _THREAD_P0[nap:],
                rt_false, geom=geom, Bp=None)
        assert np.array_equal(out_false, np.asarray(direct).flatten())

        lp_false = bing_inf.log_prob(_THREAD_P0, models, ts['Rrs'],
                                     ts['varRrs'], rt_false, geom=geom)
        lp_absent = bing_inf.log_prob(_THREAD_P0, models, ts['Rrs'],
                                      ts['varRrs'], rt_absent, geom=geom)
        assert lp_false == lp_absent
        assert np.isfinite(lp_false)


def test_fit_Bp_tail_inert_for_robust_baseline(threading_setup):
    """robust_baseline *accepts* a free B_p mechanically (the adapter
    builds PhaseParams for every backend) but the value is inert:
    robust.rt.baselines.Rrs_gordon discards phase_params by construction
    (standard Gordon has no phase-function input).  Pinned so a fit_Bp
    fit on the baseline visibly does nothing -- the posterior would just
    return the prior.  Whether validate_rt_dict should reject the
    combination outright is an M3 Q&A item (task 3 owns validate pins)."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_baseline', fit_Bp=True)
    wave = ts['models'][0].wave

    out_lo = chisq_fit.fit_func(wave, *_THREAD_P0, 0.005,
                                models=ts['models'], rt_dict=rt_dict,
                                geom=geom)
    out_hi = chisq_fit.fit_func(wave, *_THREAD_P0, 0.05,
                                models=ts['models'], rt_dict=rt_dict,
                                geom=geom)
    assert np.array_equal(out_lo, out_hi)


def _recording_domain_check(record):
    """Replace robust_domain_check, recording (len(a), len(bb), Bp)."""
    def recorder(a_model, a_params, bb_model, bb_params, rt_dict,
                 geom=None, Bp=None):
        record.append((len(a_params), len(bb_params), Bp))
    return recorder


def test_chisq_fit_fit_Bp_p0_tail_reaches_domain_check(
        threading_setup, monkeypatch):
    """chisq_fit.fit's setup-time domain check peels the same tail: a
    tailed p0 under fit_Bp=True delivers the un-tailed aparams/bparams
    slices plus Bp = the tail value (not Bp=None).  curve_fit is stubbed
    (echoes p0) -- the end-to-end optimizer path is task 3's gate."""
    ts = threading_setup
    record = []
    monkeypatch.setattr(evaluate, 'robust_domain_check',
                        _recording_domain_check(record))
    monkeypatch.setattr(chisq_fit, 'curve_fit',
                        lambda f, x, y, p0=None, **k: (np.asarray(p0),
                                                       np.eye(len(p0))))

    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt', fit_Bp=True)
    tailed = np.append(_THREAD_P0, _BP_TAIL)
    items = (ts['Rrs'], ts['varRrs'], tailed, 0, ObsGeometry(theta_s=30.))
    ans, cov, idx = chisq_fit.fit(items, ts['models'], rt_dict)

    nap = ts['models'][0].nparam
    assert record == [(nap, _THREAD_P0.size - nap, _BP_TAIL)]
    assert ans.size == tailed.size   # the optimizer vector keeps the tail


def test_fit_one_fit_Bp_p0_and_median_tails_reach_domain_check(
        threading_setup, monkeypatch):
    """fit_one's two setup/teardown domain checks (p0 and posterior
    median) both peel the tail under fit_Bp=True.  run_emcee is stubbed
    with a fake sampler whose chain carries a known constant B_p column,
    so the two recorded Bp values are distinguishable: the p0 check sees
    the p0 tail, the median check sees the chain's tail median."""
    ts = threading_setup
    record = []
    monkeypatch.setattr(evaluate, 'robust_domain_check',
                        _recording_domain_check(record))

    median_tail = 0.03
    tailed = np.append(_THREAD_P0, _BP_TAIL)

    class _FakeSampler:
        def get_chain(self):
            # (nsteps, nwalkers, ndim+1) -- every sample identical, with
            # the B_p column at median_tail.
            vec = np.append(_THREAD_P0, median_tail)
            return np.tile(vec, (4, 3, 1))
    monkeypatch.setattr(bing_inf, 'run_emcee',
                        lambda *a, **k: _FakeSampler())

    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt', fit_Bp=True)
    items = (ts['Rrs'], ts['varRrs'], tailed, 0, ObsGeometry(theta_s=30.))
    chains, idx = bing_inf.fit_one(items, models=ts['models'],
                                   pdict=ts['pdict'], chains_only=True,
                                   rt_dict=rt_dict)

    nap = ts['models'][0].nparam
    nbp = _THREAD_P0.size - nap
    assert record == [(nap, nbp, _BP_TAIL), (nap, nbp, median_tail)]
    assert chains.shape[-1] == tailed.size   # the chain keeps the column


# ===== M3 task 2: the free-B_p prior + p0 seeding (design §7.3) =====
#
# log_prob evaluates inference.BP_PRIOR -- a *linear-space* uniform over
# the inclusive [rt_defs.BP_PRIOR_PMIN, rt_defs.BP_PRIOR_PMAX] =
# [0.004, 0.05] (plan choice; B_p is a ratio, not a log10 amplitude) --
# on the peeled tail, alongside the model priors and *before* the forward
# dispatch.  In range the uniform contributes exactly 0 (the codebase's
# UniformPrior convention -- no -log(width) normalization), so task 1's
# exact-equivalence pins above still hold verbatim.  Out of range it
# short-circuits to -inf with no forward-model call -- pinned here with a
# bomb monkeypatch on the adapter, not just by the -inf value.  The
# chi-squared path has no in-fit_func counterpart by design: bounds are
# curve_fit's job (l23.fit_with_LM builds the B_p slot from the same
# rt_defs constants), exactly as for the model parameters.


def test_log_prob_Bp_prior_in_range_reaches_forward(
        threading_setup, monkeypatch):
    """In-range tails -- including *exactly* the inclusive bounds
    0.004/0.05 -- pass the B_p prior, reach the forward call (recorded on
    a pass-through wrapper), and yield a finite log-probability."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt', fit_Bp=True)

    forwarded = []
    real = evaluate.calc_Rrs_from_models_robust

    def recording(*args, **kwargs):
        forwarded.append(kwargs.get('Bp'))
        return real(*args, **kwargs)
    monkeypatch.setattr(evaluate, 'calc_Rrs_from_models_robust', recording)

    for Bp in [rt_defs.BP_PRIOR_PMIN, 0.02, rt_defs.BP_PRIOR_PMAX]:
        lp = bing_inf.log_prob(np.append(_THREAD_P0, Bp), ts['models'],
                               ts['Rrs'], ts['varRrs'], rt_dict, geom=geom)
        assert np.isfinite(lp)
    assert forwarded == [rt_defs.BP_PRIOR_PMIN, 0.02, rt_defs.BP_PRIOR_PMAX]


def test_log_prob_Bp_prior_out_of_range_short_circuits(
        threading_setup, monkeypatch):
    """Out-of-range tails (just past either inclusive bound, and grossly
    out) return -inf *without* the forward model ever running: both
    adapters are replaced with bombs, so a single forward call would
    fail the test, not just change a value."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt', fit_Bp=True)

    def bomb(*args, **kwargs):
        raise AssertionError("forward model called despite out-of-range B_p")
    monkeypatch.setattr(evaluate, 'calc_Rrs_from_models_robust', bomb)
    monkeypatch.setattr(evaluate, 'calc_Rrs_from_models', bomb)

    for Bp in [0.0039, 0.0501, 0., -0.01, 0.2]:
        lp = bing_inf.log_prob(np.append(_THREAD_P0, Bp), ts['models'],
                               ts['Rrs'], ts['varRrs'], rt_dict, geom=geom)
        assert lp == -np.inf


def test_log_prob_fit_Bp_false_never_touches_Bp_prior(
        threading_setup, monkeypatch):
    """fit_Bp False/absent runs no B_p prior logic at all: BP_PRIOR.calc
    is a bomb, and the untailed vector's *last model parameter* (beta =
    0.9, far outside [0.004, 0.05]) shows the check is not misfiring on
    the model tail.  Pinned on both backends, False and key-absent."""
    ts = threading_setup
    geom = ObsGeometry(theta_s=30.)

    def bomb(param):
        raise AssertionError("BP_PRIOR.calc invoked with fit_Bp off")
    monkeypatch.setattr(bing_inf.BP_PRIOR, 'calc', bomb)

    assert not (rt_defs.BP_PRIOR_PMIN <= _THREAD_P0[-1]
                <= rt_defs.BP_PRIOR_PMAX)
    for backend in ['gordon', 'robust_ztt']:
        rt_false = dict(ts['rt_dict'], rt_backend=backend, fit_Bp=False)
        rt_absent = {k: v for k, v in rt_false.items() if k != 'fit_Bp'}
        lp_false = bing_inf.log_prob(_THREAD_P0, ts['models'], ts['Rrs'],
                                     ts['varRrs'], rt_false, geom=geom)
        lp_absent = bing_inf.log_prob(_THREAD_P0, ts['models'], ts['Rrs'],
                                      ts['varRrs'], rt_absent, geom=geom)
        assert np.isfinite(lp_false)
        assert lp_false == lp_absent


def test_append_Bp_seed_appends_under_fit_Bp():
    """fit_Bp=True: the seed rides as the tail -- rt_dict['Bp_value']
    verbatim (0.01 default when the key is missing, matching
    rt_dict_from_p), linear-space, inside the B_p prior (contribution
    exactly 0), and nonzero so init_walkers' floor spreads it."""
    p0 = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])

    tailed = bing_inf.append_Bp_seed(p0, {'fit_Bp': True, 'Bp_value': 0.02})
    assert tailed.size == p0.size + 1
    assert np.array_equal(tailed[:-1], p0)
    assert tailed[-1] == 0.02

    # Missing Bp_value falls back to rt_dict_from_p's 0.01 default
    defaulted = bing_inf.append_Bp_seed(p0, {'fit_Bp': True})
    assert defaulted[-1] == 0.01

    # The default seed really is a legal, spreadable starting point
    assert rt_defs.BP_PRIOR_PMIN <= defaulted[-1] <= rt_defs.BP_PRIOR_PMAX
    assert bing_inf.BP_PRIOR.calc(defaulted[-1]) == 0
    assert defaulted[-1] != 0.


def test_append_Bp_seed_noop_without_fit_Bp():
    """fit_Bp False, key absent, or rt_dict None: p0 comes back
    unchanged -- same values, same length, no tail."""
    p0 = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
    for rt_dict in [{'fit_Bp': False, 'Bp_value': 0.02},
                    {'Bp_value': 0.02}, {}, None]:
        out = bing_inf.append_Bp_seed(p0, rt_dict)
        assert np.array_equal(out, p0)


# ===== M3 task 3: chain bookkeeping (ndim/walkers/bounds/names/recon) =====
#
# The bookkeeping that makes a real fit_Bp=True MCMC run first-class
# (design §3.3): init_mcmc's ndim (+1 under fit_Bp), the
# prior_bounds/init_walkers B_p clip slot (per M3-Q3 this upgrades an
# *unclipped* tail column to a clipped one, not a crash fix),
# reconstruct_from_chains' tail strip + backend dispatch, the 'B_p'
# naming for calc_stats/corner plots (linear -- log_params False), the
# fitter-level gordon+fit_Bp setup rejection, and the Gate's synthetic
# round-trip.  Gate item 6's "identical to an M2-pinned value" is
# satisfied with a **provisional** pin (files/m3_fixed_bp_pin.npz) --
# M2 Q5 never produced one; see the fixture generator's docstring and
# prompt-4 Q&A Q6.

_ROUNDTRIP_BP_TRUE = 0.02


def test_init_mcmc_fit_Bp_adds_dimension(threading_setup):
    """Gate item 2 (setup side): ndim = sum(nparam) + 1 under fit_Bp,
    nwalkers = max(16, 2*ndim) accounts for it automatically; rt_dict
    None / fit_Bp False leave both exactly as before."""
    models = threading_setup['models']
    nmodel = sum(model.nparam for model in models)

    base = bing_inf.init_mcmc(models, nsteps=30, nburn=10)
    assert base['ndim'] == nmodel
    assert base['nwalkers'] == max(16, 2*nmodel)

    fixed = bing_inf.init_mcmc(models, nsteps=30, nburn=10,
                               rt_dict={'rt_backend': 'robust_ztt',
                                        'fit_Bp': False})
    assert fixed['ndim'] == base['ndim']
    assert fixed['nwalkers'] == base['nwalkers']

    free = bing_inf.init_mcmc(models, nsteps=30, nburn=10,
                              rt_dict={'rt_backend': 'robust_ztt',
                                       'fit_Bp': True})
    assert free['ndim'] == nmodel + 1
    assert free['nwalkers'] == max(16, 2*(nmodel + 1))
    # For the standard 5-parameter pair that is still 16 walkers.
    assert free['nwalkers'] == 16


def test_prior_bounds_fit_Bp_appends_slot(threading_setup):
    """prior_bounds gains the trailing B_p slot under fit_Bp, sourced
    from the rt_defs constants (the single place the range lives); the
    model-parameter bounds are untouched, and fit_Bp False/None rt_dicts
    reproduce the pre-M3 arrays exactly."""
    models = threading_setup['models']
    low0, high0 = bing_inf.prior_bounds(models)

    for rt_dict in [None, {}, {'fit_Bp': False}]:
        low, high = bing_inf.prior_bounds(models, rt_dict=rt_dict)
        assert np.array_equal(low, low0) and np.array_equal(high, high0)

    low, high = bing_inf.prior_bounds(models, rt_dict={'fit_Bp': True})
    assert low.size == low0.size + 1 and high.size == high0.size + 1
    assert np.array_equal(low[:-1], low0)
    assert np.array_equal(high[:-1], high0)
    assert low[-1] == rt_defs.BP_PRIOR_PMIN
    assert high[-1] == rt_defs.BP_PRIOR_PMAX


def test_init_walkers_clips_Bp_tail_into_prior(threading_setup):
    """M3-Q3's upgrade made real: a B_p seed near the lower prior edge
    (0.0045) with the 1e-3 perturbation floor throws some walkers below
    BP_PRIOR_PMIN -- without rt_dict they stay there (the pre-M3
    unclipped-tail behavior), with rt_dict they are clipped into
    [0.004, 0.05].  The model-parameter columns are identical either
    way (same seed)."""
    models = threading_setup['models']
    p0 = bing_inf.append_Bp_seed(_THREAD_P0,
                                 {'fit_Bp': True, 'Bp_value': 0.0045})

    np.random.seed(7)
    unclipped = bing_inf.init_walkers(p0, 200, models=models)
    np.random.seed(7)
    clipped = bing_inf.init_walkers(p0, 200, models=models,
                                    rt_dict={'fit_Bp': True})

    # The floor really pushes walkers out of range -- the clip is not
    # theoretical -- and the extension pulls exactly those back in.
    assert (unclipped[:, -1] < rt_defs.BP_PRIOR_PMIN).any()
    assert (clipped[:, -1] >= rt_defs.BP_PRIOR_PMIN).all()
    assert (clipped[:, -1] <= rt_defs.BP_PRIOR_PMAX).all()
    # Model columns are untouched by the tail slot.
    assert np.array_equal(unclipped[:, :-1], clipped[:, :-1])


def test_fit_Bp_roundtrip_recovers_Bp(threading_setup):
    """Gate items 1 + 2 (chain side): generate Rrs with robust_ztt at a
    known B_p=0.02, fit with fit_Bp=True (seeded at the 0.01 default) --
    the chain carries the extra trailing column, the posterior median of
    B_p lies inside its own 5-95% credible interval, the interval
    contains the truth and excludes both prior edges (the data, not the
    prior, constrains the answer).  Deterministic seed; a 0.5% assumed
    error keeps the posterior decisively narrower than the prior
    (measured 2026-08-31: median 0.0205, CI [0.0177, 0.0240], ~3 s)."""
    ts = threading_setup
    models = ts['models']
    wave = models[0].wave
    geom = ObsGeometry(theta_s=30.)
    rt_dict = dict(ts['rt_dict'], rt_backend='robust_ztt', fit_Bp=True)

    # Synthetic truth: the fixture's model parameters, B_p = 0.02.
    Rrs = chisq_fit.fit_func(wave, *_THREAD_TRUTH, _ROUNDTRIP_BP_TRUE,
                             models=models, rt_dict=rt_dict, geom=geom)
    varRrs = (0.005*Rrs)**2

    nsteps, nburn = 800, 200
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn,
                               rt_dict=rt_dict)
    pdict['Chl'] = np.array([1.0])
    pdict['Y'] = None
    p0 = bing_inf.append_Bp_seed(_THREAD_P0, rt_dict)  # seeds at 0.01

    np.random.seed(1234)
    chains, idx = bing_inf.fit_one((Rrs, varRrs, p0, 0, geom),
                                   models=models, pdict=pdict,
                                   chains_only=True, rt_dict=rt_dict)

    # Gate item 2: the extra trailing column, at the bookkept ndim.
    nmodel = sum(model.nparam for model in models)
    assert pdict['ndim'] == nmodel + 1
    assert chains.shape == (nsteps, pdict['nwalkers'], nmodel + 1)
    assert np.all(np.isfinite(chains))

    # Gate item 1, on the flattened post-burn B_p column.
    Bp_col = chains[nburn:, :, -1].ravel()
    med = np.median(Bp_col)
    p5, p95 = np.percentile(Bp_col, [5, 95])
    assert p5 <= med <= p95                       # the Gate's literal check
    assert p5 <= _ROUNDTRIP_BP_TRUE <= p95        # truth recovered
    assert abs(med - _ROUNDTRIP_BP_TRUE) < 0.005  # ... and reasonably well
    assert p5 > rt_defs.BP_PRIOR_PMIN             # prior edges excluded:
    assert p95 < rt_defs.BP_PRIOR_PMAX            # the data did the work
    # Every sample obeyed the B_p prior (log_prob's -inf gate + the
    # init_walkers clip).
    assert Bp_col.min() >= rt_defs.BP_PRIOR_PMIN
    assert Bp_col.max() <= rt_defs.BP_PRIOR_PMAX


def test_chain_param_names_and_calc_stats_end_in_B_p(threading_setup):
    """Gate item 3: the fitted-vector names gain the trailing 'B_p'
    under fit_Bp (and only then), and calc_stats carries them through
    with one statistic per chain column."""
    models = threading_setup['models']
    pnames = list(models[0].pnames) + list(models[1].pnames)

    for rt_dict in [None, {}, {'fit_Bp': False}]:
        assert evaluate.chain_param_names(models, rt_dict=rt_dict) == pnames

    names = evaluate.chain_param_names(models, rt_dict={'fit_Bp': True})
    assert names == pnames + ['B_p']

    # Through calc_stats on a fit_Bp-shaped chain (its internal
    # thin_burn_chains burns 7000 steps).
    rng = np.random.default_rng(3)
    truth = np.append(_THREAD_TRUTH, 0.02)
    chains = truth + 0.001*rng.standard_normal((7100, 4, truth.size))
    stats = evaluate.calc_stats(chains, names=names)
    assert stats['names'][-1] == 'B_p'
    assert stats['med'].size == truth.size
    assert abs(stats['med'][-1] - 0.02) < 0.001


def test_corner_plot_B_p_column_linear_label(threading_setup):
    """Corner-plot conventions for the extra column: log_param_mask
    appends False (B_p is linear-space -- the debug-priors trap, avoided
    deliberately), so the 'B_p' label is never log10-wrapped and the
    column is never exponentiated by show_log=False."""
    import matplotlib
    matplotlib.use('Agg', force=True)
    from bing import plotting as bing_plot

    models = threading_setup['models']
    rt_dict = {'rt_backend': 'robust_ztt', 'fit_Bp': True}

    # The mask itself: model entries unchanged, trailing False.
    assert bing_plot.log_param_mask(models, rt_dict=rt_dict) == \
        bing_plot.log_param_mask(models) + [False]
    assert bing_plot.log_param_mask(models, rt_dict={'fit_Bp': False}) == \
        bing_plot.log_param_mask(models)

    # Rendered labels, read back off the axes (test_plotting's recipe).
    rng = np.random.default_rng(11)
    truth = np.append(_THREAD_TRUTH, 0.02)
    chains = truth + 0.01*rng.standard_normal((7200, 16, truth.size))
    chains[..., -1] = np.clip(chains[..., -1], 0.004, 0.05)
    n = truth.size
    fig = bing_plot.corner_plot(chains, models=models, show=False,
                                rt_dict=rt_dict)
    axes = np.array(fig.axes).reshape(n, n)
    labels = [axes[n-1, kk].get_xlabel() for kk in range(n)]
    assert labels[-1] == 'B_p'                     # linear: no log10(...)
    assert r'\log_{10}' in labels[0]               # Adg still log-labelled
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_reconstruct_from_chains_fit_Bp_shapes_and_tail_flows(threading_setup):
    """Gate item 4: reconstruct_from_chains on a fit_Bp chain returns
    correctly-shaped, finite IOPs/Rrs -- the trailing B_p column is
    stripped before the model split and forwarded as the adapter's Bp.
    Pinned by the M3 task-1 equivalence style: a chain with a constant
    B_p tail of 0.02 under fit_Bp=True reconstructs like the same chain
    without the tail under Bp_value=0.02 -- exactly for the IOPs (which
    never see B_p), and to float32 ulp for Rrs (the scalar-B_p and
    per-sample-B_p jit traces are different XLA programs, so exact
    bitwise equality is not guaranteed there) -- and not like the 0.01
    default: the tail really flowed."""
    ts = threading_setup
    models = ts['models']
    nwave = len(models[0].wave)
    geom = ObsGeometry(theta_s=30.)
    base = dict(ts['rt_dict'], rt_backend='robust_ztt')

    # A synthetic chain long enough for the internal 7000-step burn.
    rng = np.random.default_rng(5)
    model_cols = (_THREAD_TRUTH
                  + 0.005*rng.standard_normal((7010, 4, _THREAD_TRUTH.size)))
    tail = np.full((7010, 4, 1), 0.02)
    chains_tailed = np.concatenate([model_cols, tail], axis=-1)

    out = evaluate.reconstruct_from_chains(
        chains=chains_tailed, models=models,
        rt_dict=dict(base, fit_Bp=True), geom=geom)
    assert len(out) == 8
    for arr in out:
        assert np.shape(arr) == (nwave,)
        assert np.all(np.isfinite(arr))

    out_fixed = evaluate.reconstruct_from_chains(
        models, model_cols, dict(base, fit_Bp=False, Bp_value=0.02),
        geom=geom)
    out_default = evaluate.reconstruct_from_chains(
        models, model_cols, dict(base, fit_Bp=False), geom=geom)  # 0.01

    # IOP outputs (a/bb medians and bands, indices 0-5) never see B_p:
    # exactly equal.  Rrs/sigRs (indices 6-7) do -- equal to float32 ulp.
    for free, fixed in zip(out[:6], out_fixed[:6]):
        assert np.array_equal(free, fixed)
    for free, fixed in zip(out[6:], out_fixed[6:]):
        np.testing.assert_allclose(free, fixed, rtol=1e-5)
    # ... and the 0.01 default gives genuinely different Rrs (the tail
    # flowed), while the IOPs stay put.
    assert not np.allclose(out[6], out_default[6])
    assert np.array_equal(out[0], out_default[0])   # a median unchanged
    assert np.array_equal(out[1], out_default[1])   # bb median unchanged


def test_fit_one_gordon_fit_Bp_raises_at_setup(threading_setup, monkeypatch):
    """Gate item 5 (MCMC side): fit_Bp=True + rt_backend='gordon' raises
    validate_rt_dict's configuration error at fit setup, before any
    sampling work -- run_emcee is bombed to prove it is never reached.
    (validate_rt_dict's own unit pin is
    test_validate_rt_dict_rejects_fit_Bp_with_gordon, M0.)"""
    ts = threading_setup

    def bomb(*args, **kwargs):
        raise AssertionError('run_emcee must not be reached')
    monkeypatch.setattr(bing_inf, 'run_emcee', bomb)

    rt_dict = dict(ts['rt_dict'], rt_backend='gordon', fit_Bp=True)
    p0 = bing_inf.append_Bp_seed(_THREAD_P0, rt_dict)
    with pytest.raises(ValueError, match='fit_Bp'):
        bing_inf.fit_one((ts['Rrs'], ts['varRrs'], p0, 0),
                         models=ts['models'], pdict=ts['pdict'],
                         chains_only=True, rt_dict=rt_dict)


def test_chisq_fit_gordon_fit_Bp_raises_at_setup(threading_setup, monkeypatch):
    """Gate item 5 (chi-squared side): the same configuration error
    through chisq_fit.fit, before any optimizer work."""
    ts = threading_setup

    def bomb(*args, **kwargs):
        raise AssertionError('curve_fit must not be reached')
    monkeypatch.setattr(chisq_fit, 'curve_fit', bomb)

    rt_dict = dict(ts['rt_dict'], rt_backend='gordon', fit_Bp=True)
    p0 = bing_inf.append_Bp_seed(_THREAD_P0, rt_dict)
    with pytest.raises(ValueError, match='fit_Bp'):
        chisq_fit.fit((ts['Rrs'], ts['varRrs'], p0, 0),
                      ts['models'], rt_dict)


def test_fit_Bp_false_matches_provisional_pin(threading_setup):
    """Gate item 6, via the **provisional** pin (see
    files/gen_m3_fixed_bp_pin.py): fixed-B_p (fit_Bp False) fitter
    results -- chisq_fit.fit params and seeded fit_one chains, gordon
    and robust_ztt -- match the values captured live under
    task-3-complete code on 2026-08-31.  M2 Q5 (a true pre-M3 pin) is
    still open; this fixture stands in for it and freezes the dispatch
    behavior the M3 task-1 byte-identity tests verified against M2, so
    any *future* change to the fixed-B_p path fails here.  If this test
    ever fails on a new environment (BLAS/emcee version) while
    test_fit_Bp_false_or_absent_byte_identical_to_m2 stays green,
    regenerate the fixture rather than suspecting a regression."""
    import os
    ts = threading_setup
    models = ts['models']
    pin = np.load(os.path.join(os.path.dirname(__file__), 'files',
                               'm3_fixed_bp_pin.npz'))

    # The fixture recipe is threading_setup's: same truth, p0, and
    # observation (guards against the two drifting apart).
    np.testing.assert_allclose(ts['Rrs'], pin['Rrs'], rtol=0, atol=0)
    assert np.array_equal(pin['truth'], _THREAD_TRUTH)
    assert np.array_equal(pin['p0'], _THREAD_P0)

    geom = ObsGeometry(theta_s=30.)
    for backend in ('gordon', 'robust_ztt'):
        rt = dict(ts['rt_dict'], rt_backend=backend)   # fit_Bp False
        items = ((pin['Rrs'], pin['varRrs'], _THREAD_P0.copy(), 0)
                 if backend == 'gordon' else
                 (pin['Rrs'], pin['varRrs'], _THREAD_P0.copy(), 0, geom))

        ans, _, _ = chisq_fit.fit(items, models, rt)
        np.testing.assert_allclose(ans, pin[f'chisq_{backend}'],
                                   rtol=1e-6, atol=1e-12)

        pdict = bing_inf.init_mcmc(models, nsteps=int(pin['nsteps']),
                                   nburn=int(pin['nburn']), rt_dict=rt)
        pdict['Chl'] = np.array([1.0])
        pdict['Y'] = None
        np.random.seed(int(pin['seed']))
        chains, _ = bing_inf.fit_one(items, models=models, pdict=pdict,
                                     chains_only=True, rt_dict=rt)
        assert chains.shape == pin[f'chain_{backend}'].shape
        np.testing.assert_allclose(chains, pin[f'chain_{backend}'],
                                   rtol=1e-6, atol=1e-12)
