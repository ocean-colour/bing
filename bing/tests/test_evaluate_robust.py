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

The forward adapter (M1+) lands in later additions to this file.
"""
import numpy as np

import pytest

from bing.rt import defs as rt_defs
from bing.rt.geometry import ObsGeometry


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
