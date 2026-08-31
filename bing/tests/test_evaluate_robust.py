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
