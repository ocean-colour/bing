""" ``correct_atmosphere``-dependent Ed tests for the robust RT backend
(rob_rt integration, M4 task 3).

This file exists for one reason (the CQ3 sub-point in
``claude_prompts/RT/rob_rt_prompt_5.md``): tests that need the optional
``correct_atmosphere`` package -- the production zenith-0 downwelling-
irradiance generation that ``bing.fitting.l23`` feeds to ``set_raman_Ed``
-- cannot even be *collected* without it, and an ImportError during
collection cannot be converted to a skip. So every such test lives here,
and ``conftest.py`` drops this whole file from collection when
``correct_atmosphere`` is not importable (the same mechanism that drops
``test_evaluate.py``/``test_io.py``/``test_l23_fitting.py``). Nothing
gated in M0-M3 may require ``correct_atmosphere``; the synthetic-Ed M4
tests (Gate items 1, 3, 4) are independent of it and live in
``test_evaluate_robust.py`` instead -- the Gate preamble's placement rule.

Covered here (M4 task 3): the production-sky end-to-end seam -- the exact
``fitting/l23.py`` zenith-0 Ed recipe stashed via ``set_raman_Ed`` routes
verbatim (``is``-identity) onto robust's ``Geometry.Ed`` and yields
finite, positive Rrs that measurably differs from robust's packaged-L23
default sky.

NOT here yet: Gate item 2's ``robust_baseline``+Raman vs ``gordon``+Raman
cross-checks -- blocked on Q1 in ``rob_rt_prompt_5.md`` (M1's ValueError
guard makes the comparison as literally worded impossible; JXP's answer
pending). They belong in this file when written, since they need the
matched real Ed spectrum on both sides.
"""
import numpy as np

import pytest

# The optional dependency this file is dropped over (conftest.py's
# collect_ignore): the production Ed source, imported exactly as
# bing/fitting/l23.py imports it.
from correct_atmosphere import downwelling

from bing.models import utils as model_utils
from bing.rt.geometry import ObsGeometry
from bing import evaluate


@pytest.fixture()
def robust_models():
    """A fresh ExpBricaud + Pow model pair on a robust_hybrid-legal grid
    (mirrors test_evaluate_robust.py's fixture of the same name)."""
    wave = np.linspace(400., 700., 61)
    return model_utils.init(['ExpBricaud', 'Pow'], wave)


def _production_ed_pair(a_model):
    """The zenith-0 production Ed recipe, byte-for-byte the one in
    ``bing/fitting/l23.py`` (the grid covering wave_ex ~50 nm blueward of
    the model grid, 1-nm sampling, theta_s=0)."""
    wv_Ed = np.arange(np.floor(a_model.wave_ex.min()) - 5.,
                      a_model.wave.max() + 5.1, 1.)
    Ed = downwelling.downwelling_irradiance(wv_Ed, 0.)
    return wv_Ed, Ed


def test_production_ed_routes_and_changes_robust_raman(robust_models):
    """End-to-end with the real sky: the production zenith-0 Ed pair
    stashed by ``set_raman_Ed`` lands verbatim on ``Geometry.Ed``
    (``is``-identity through ``_build_robust_inputs``), the robust_ztt +
    Raman Rrs is finite and positive, and it differs measurably from the
    packaged-L23 default (``Ed=None``) -- the seam is live for the
    production spectrum too, not just the synthetic skies pinned in
    test_evaluate_robust.py."""
    a_model, bb_model = robust_models
    a_model.set_aph(np.array([1.0]))
    a_params = np.array([-1.5, 0.017, np.log10(0.05582 * 1.0)])
    bb_params = np.array([-3.0, 1.0])
    geom = ObsGeometry(theta_s=30.)
    rt_dict = {'rt_backend': 'robust_ztt', 'include_Raman': True,
               'include_Chl_fl': False, 'phi_C': 0.02,
               'double_gaussian': True, 'Bp_value': 0.014}

    # Packaged-L23 default first (no stash).
    Rrs_default = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)

    wv_Ed, Ed = _production_ed_pair(a_model)
    a_model.set_raman_Ed(wv_Ed, Ed)

    # Verbatim routing of the production pair.
    inp = evaluate._build_robust_inputs(
        a_model, a_params, bb_model, bb_params, rt_dict, geom, None)
    assert inp.geometry.Ed is not None
    assert inp.geometry.Ed[0] is wv_Ed
    assert inp.geometry.Ed[1] is Ed

    Rrs_prod = evaluate.calc_Rrs_from_models_robust(
        a_model, a_params, bb_model, bb_params, rt_dict, geom=geom)
    assert np.all(np.isfinite(Rrs_prod)) and np.all(Rrs_prod > 0)

    # Both are solar-shaped skies, but the generated zenith-0 spectrum is
    # not the packaged anchor table interpolated to theta_s=30, and the
    # seam must see that: measured max relative difference 1.5e-2 (at
    # 505 nm; mean 4.2e-3) at these IOPs. Gate an order of magnitude
    # below the measurement and far above float32 ULP noise (~1e-7).
    rel_diff = np.abs(Rrs_prod - Rrs_default) / np.abs(Rrs_default)
    assert rel_diff.max() > 1e-3
