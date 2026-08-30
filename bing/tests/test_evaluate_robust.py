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
checks). The JIT strategy (task 2), the un-jitted domain check (task 3), and
dropping ``RT_correction`` (task 4) land in later additions.
"""
import numpy as np

import pytest

from bing.models import utils as model_utils
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
