"""Tests for the CDOM-fluorescence path of the robust.rt backend (rob_cdom).

The third inelastic term: ``rt_dict['include_CDOM_fl']`` composes
``robust.rt.cdom_fl``'s analytic Hawes et al. (1992) kernel inside
``robust.rt.forward``, alongside Raman and chlorophyll fluorescence.

**The proxy this whole module is built on.** robust's kernel takes the
*pure CDOM* absorption spectrum as its emission source term
(``b_bY = 0.5 a_cdom``). BING has no such spectrum: every a-model with a
separable exponential term lumps CDOM and detritus into one ``a_dg``. So
the adapter supplies

    a_cdom = rt_dict['cdom_fraction'] * a_dg      (default 0.8)

a **fixed-fraction proxy**, not a retrieved quantity -- a project decision
(JXP, 2026-09-05; ``claude_prompts/rt_tests.md`` Q32) recorded in
``bing.rt.defs.CDOM_FRACTION_DEFAULT``. The kernel amplitude
``robust.rt.CDOMFl.scale`` is held fixed at ``evaluate.CDOM_FL_SCALE``
= 1.0 (the published reference kernel); BING does not fit it.

Coverage, following the task list of the rob_cdom prompt:

(a) ``rt_dict_from_p``'s two new keys and their real (non-None) defaults,
    plus explicit pass-through. The exact-key-set pin lives in
    ``test_chl_fl.py::test_rt_dict_from_p_defaults``.
(b) Off by default: with ``include_CDOM_fl`` absent or explicitly False,
    the robust forward result is **byte-identical** to a config dict that
    predates the keys entirely -- the new branch is inert when off.
(c) On adds signal: cdom-on Rrs exceeds cdom-off everywhere, and the
    difference peaks where the Hawes emission lives (the kernel excites
    over 350-490 nm and re-emits, per Zhai et al. 2017 Eq. 7, in a
    Gaussian in *wavenumber* centred at ``A1/lambda_e + B1``; for the
    350-490 nm excitation band that puts emission peaks between roughly
    465 and 570 nm).
(d) ``cdom_fraction`` scales the term (near-)linearly -- the source term
    is linear in ``a_cdom``, so 0.4 gives ~half of 0.8.
(e) Validation: gordon + cdom raises, robust_baseline + cdom raises, and
    an a-model without a separable a_dg raises (naming the class).
(f) jit-cache separation: cdom-on and cdom-off are distinct
    ``_robust_forward_jit`` entries, and a repeat call hits the cache.
(g) An MCMC round-trip smoke test with all three inelastic terms plus
    ``fit_Bp``, sized like the existing robust MCMC tests.
(h) ``calc_Rrs_from_iops_robust``'s new ``a_cdom=`` argument, and its
    error when CDOM fluorescence is on without one.

No ``pytest.importorskip`` for ``robust``, for the same reason as
``test_evaluate_robust.py``: retrieve-or-bust is a real runtime dependency
of bing, not an optional one.
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


# The Hawes emission band: the kernel's excitation grid is 350-490 nm
# (robust.rt.cdom_fl.CDOM_EX_MIN/MAX) and each excitation wavelength
# re-emits around 1/(A1/lambda_e + B1) nm -- 465 nm at the blue end of the
# band, 566 nm at the red end. Widened generously here; the point of the
# assertion is "blue-green, not the 685 nm Chl-fl line", not a pin on the
# kernel's own arithmetic (robust's test suite owns that).
_HAWES_EMISSION_LO = 430.
_HAWES_EMISSION_HI = 620.

# The same cheap synthetic recipe as test_evaluate_robust.py: a real
# ExpBricaud + Pow pair on a robust_hybrid-legal grid, no external data
# tree ($OS_COLOR) needed.
_A_PARAMS = np.array([-1.5, 0.017, np.log10(0.05582 * 1.0)])
_BB_PARAMS = np.array([-3.0, 1.0])

_BASE_RT = {'rt_backend': 'robust_ztt', 'include_Raman': False,
            'include_Chl_fl': False, 'phi_C': 0.02,
            'double_gaussian': True, 'Bp_value': 0.014}


@pytest.fixture()
def cdom_models():
    """A fresh ExpBricaud + Pow model pair with a_ph set, on a legal grid."""
    wave = np.linspace(400., 700., 61)
    models = model_utils.init(['ExpBricaud', 'Pow'], wave)
    models[0].set_aph(np.array([1.0]))
    return models


@pytest.fixture()
def geom():
    return ObsGeometry(theta_s=30.)


def _Rrs(models, rt_dict, geom, **kwargs):
    """calc_Rrs_from_models_robust on the fixture parameters."""
    return evaluate.calc_Rrs_from_models_robust(
        models[0], _A_PARAMS, models[1], _BB_PARAMS, rt_dict, geom=geom,
        **kwargs)


# ===== (a) rt_dict_from_p =====

def test_rt_dict_from_p_cdom_keys_default_off():
    """A legacy-style p (no include_CDOM_fl/cdom_fraction attributes)
    yields real defaults -- the process off, and the documented 0.8 proxy
    fraction -- not None. Same "legacy p stays valid" rule as the three
    rob_rt keys."""
    from bing.parameters import p_ntuple

    p = p_ntuple.gen(model_names=['ExpBricaud', 'Pow'])
    rt_dict = rt_defs.rt_dict_from_p(p)

    assert rt_dict['include_CDOM_fl'] is False
    assert rt_dict['cdom_fraction'] == rt_defs.CDOM_FRACTION_DEFAULT
    assert rt_defs.CDOM_FRACTION_DEFAULT == 0.8


def test_rt_dict_from_p_cdom_keys_explicit():
    """Explicit values on p propagate verbatim."""
    from collections import namedtuple

    Custom = namedtuple('Custom',
                        ['model_names', 'rt_backend', 'include_CDOM_fl',
                         'cdom_fraction'])
    p = Custom(model_names=['ExpBricaud', 'Pow'], rt_backend='robust_ztt',
               include_CDOM_fl=True, cdom_fraction=0.55)
    rt_dict = rt_defs.rt_dict_from_p(p)

    assert rt_dict['include_CDOM_fl'] is True
    assert rt_dict['cdom_fraction'] == 0.55


# ===== (b) off by default: the new branch is inert =====

@pytest.mark.parametrize('inelastic', [
    dict(),
    dict(include_Raman=True),
    dict(include_Chl_fl=True),
    dict(include_Raman=True, include_Chl_fl=True),
])
def test_cdom_off_is_byte_identical_to_pre_change_rt_dict(
        cdom_models, geom, inelastic):
    """An rt_dict that never heard of the CDOM keys and one that carries
    them set to (False, 0.8) give **bit-identical** Rrs, for every
    combination of the pre-existing inelastic flags.

    Decisive because the pre-change code path is literally reachable: an
    rt_dict without the keys is exactly what the old ``rt_dict_from_p``
    produced, and ``.get(..., False)`` sends it down the same branch. Any
    accidental always-on arithmetic (multiplying by one, adding a
    computed zero) would show up as a float32 ulp difference here.
    """
    rt_pre = dict(_BASE_RT, **inelastic)                    # no CDOM keys
    rt_off = dict(rt_pre, include_CDOM_fl=False, cdom_fraction=0.8)

    Rrs_pre = _Rrs(cdom_models, rt_pre, geom)
    Rrs_off = _Rrs(cdom_models, rt_off, geom)

    np.testing.assert_array_equal(Rrs_off, Rrs_pre)


def test_cdom_off_leaves_iops_a_cdom_unset(cdom_models, geom):
    """With the process off, ``IOPs.a_cdom`` stays None -- an unset
    optional field contributes no pytree leaves and no treedef change, so
    the off-state cannot perturb the jit trace (robust.rt.types.IOPs)."""
    inp_off = evaluate._build_robust_inputs(
        cdom_models[0], _A_PARAMS, cdom_models[1], _BB_PARAMS,
        _BASE_RT, geom, None)
    assert inp_off.iops.a_cdom is None
    assert inp_off.include_cdom is False


# ===== (c) on adds signal, in the Hawes emission band =====

def test_cdom_on_adds_positive_signal_in_hawes_band(cdom_models, geom):
    """include_CDOM_fl=True adds a non-negative contribution everywhere,
    a real (not float-noise) one, and the increment peaks inside the
    Hawes emission band rather than at the 685 nm Chl-fl line.

    The tiny atol mirrors the fluorescence test in
    ``test_evaluate_robust.py``: switching an inelastic term on routes the
    *elastic* backbone down a different static code path, so a few
    wavelengths differ by float32 rounding noise on their own account.
    """
    wave = cdom_models[0].wave
    Rrs_off = np.squeeze(_Rrs(cdom_models, _BASE_RT, geom))
    Rrs_on = np.squeeze(_Rrs(cdom_models, dict(_BASE_RT,
                                               include_CDOM_fl=True), geom))

    delta = Rrs_on - Rrs_off
    assert np.all(delta >= -1e-8)              # additive emission
    assert delta.max() > 1e-6                  # a real signal, not noise
    peak_wave = wave[np.argmax(delta)]
    assert _HAWES_EMISSION_LO <= peak_wave <= _HAWES_EMISSION_HI

    # ... and broad/featureless, unlike the 685 nm Chl-fl line: the
    # increment at the red edge is a small fraction of the peak.
    assert delta[-1] < 0.2 * delta.max()


def test_cdom_on_works_with_all_three_inelastic_terms(cdom_models, geom):
    """CDOM fluorescence composes on top of Raman + Chl fluorescence (it
    is a third additive term inside robust's ``_apply_inelastic``), and
    the composed result is finite and strictly larger."""
    rt_raman_fl = dict(_BASE_RT, include_Raman=True, include_Chl_fl=True)
    Rrs_two = np.squeeze(_Rrs(cdom_models, rt_raman_fl, geom))
    Rrs_three = np.squeeze(_Rrs(cdom_models,
                                dict(rt_raman_fl, include_CDOM_fl=True), geom))

    assert np.all(np.isfinite(Rrs_three))
    assert np.all(Rrs_three >= Rrs_two - 1e-8)
    assert (Rrs_three - Rrs_two).max() > 1e-6


def test_cdom_only_configuration_is_not_silently_dropped(cdom_models, geom):
    """CDOM fluorescence alone (Raman and Chl-fl both off) still reaches
    robust: the adapter must build an ``Inelastic`` instance for a
    cdom-only rt_dict, not fall through to the elastic ``inelastic=None``
    closure. Pinned because that fall-through would be silent -- the same
    trap robust guards against in its own ``_apply_inelastic``."""
    Rrs_off = np.squeeze(_Rrs(cdom_models, _BASE_RT, geom))
    Rrs_cdom = np.squeeze(_Rrs(cdom_models,
                               dict(_BASE_RT, include_CDOM_fl=True), geom))
    assert (Rrs_cdom - Rrs_off).max() > 1e-6


@pytest.mark.parametrize('rt_backend', ['robust_ztt', 'robust_hybrid'])
def test_cdom_shapes_1d_and_batch(cdom_models, geom, rt_backend):
    """(nparam,) -> (1, nwave) and (nsamples, nparam) -> (nsamples, nwave),
    finite throughout, with CDOM on -- the a_cdom construction is batched
    over the chain axis exactly like ``a``/``bb`` (the reconstruct_from_
    chains path depends on this)."""
    nwave = len(cdom_models[0].wave)
    rt_dict = dict(_BASE_RT, rt_backend=rt_backend, include_CDOM_fl=True)

    Rrs_1d = _Rrs(cdom_models, rt_dict, geom)
    assert Rrs_1d.shape == (1, nwave)
    assert np.all(np.isfinite(Rrs_1d))

    nsample = 4
    Rrs_batch = evaluate.calc_Rrs_from_models_robust(
        cdom_models[0], np.tile(_A_PARAMS, (nsample, 1)),
        cdom_models[1], np.tile(_BB_PARAMS, (nsample, 1)),
        rt_dict, geom=geom)
    assert Rrs_batch.shape == (nsample, nwave)
    assert np.all(np.isfinite(Rrs_batch))
    # Every identical sample reproduces the 1-D answer.
    for row in Rrs_batch:
        np.testing.assert_allclose(row, np.squeeze(Rrs_1d), rtol=1e-6)


# ===== (d) cdom_fraction scales the term =====

def test_cdom_fraction_scales_the_increment(cdom_models, geom):
    """The source term is linear in ``a_cdom = cdom_fraction * a_dg``, and
    a_cdom enters the kernel linearly, so halving the fraction halves the
    Rrs increment (to float32 tolerance; the A*rrs/(1-B*rrs) surface
    transfer is mildly non-linear, hence the loose bracket rather than an
    equality).  Monotone by construction, which is the property a sweep
    over cdom_fraction relies on."""
    Rrs_off = np.squeeze(_Rrs(cdom_models, _BASE_RT, geom))
    d08 = np.squeeze(_Rrs(cdom_models, dict(_BASE_RT, include_CDOM_fl=True,
                                            cdom_fraction=0.8),
                          geom)) - Rrs_off
    d04 = np.squeeze(_Rrs(cdom_models, dict(_BASE_RT, include_CDOM_fl=True,
                                            cdom_fraction=0.4),
                          geom)) - Rrs_off

    assert np.all(d04 <= d08 + 1e-9)            # monotone in the fraction
    assert 0.45 < d04.max() / d08.max() < 0.55  # ... and ~linear


def test_cdom_fraction_default_used_when_key_absent(cdom_models, geom):
    """An rt_dict with include_CDOM_fl=True but no 'cdom_fraction' key
    falls back to CDOM_FRACTION_DEFAULT, not to 1.0 or 0.0 -- the same
    defaulting the adapter documents."""
    no_key = dict(_BASE_RT, include_CDOM_fl=True)
    explicit = dict(no_key, cdom_fraction=rt_defs.CDOM_FRACTION_DEFAULT)

    np.testing.assert_array_equal(_Rrs(cdom_models, no_key, geom),
                                  _Rrs(cdom_models, explicit, geom))


def test_a_cdom_is_exactly_cdom_fraction_times_a_dg(cdom_models, geom):
    """The proxy itself, pinned at the IOPs boundary: the ``a_cdom`` the
    adapter hands robust is exactly ``cdom_fraction * a_model.eval_a_dg``
    -- documented as a fixed-fraction stand-in for pure CDOM absorption
    (JXP 2026-09-05, rt_tests.md Q32), and *not* the full a_nw or a."""
    rt_dict = dict(_BASE_RT, include_CDOM_fl=True, cdom_fraction=0.8)
    inp = evaluate._build_robust_inputs(
        cdom_models[0], _A_PARAMS, cdom_models[1], _BB_PARAMS,
        rt_dict, geom, None)

    a_dg = cdom_models[0].eval_a_dg(_A_PARAMS)
    np.testing.assert_allclose(np.asarray(inp.iops.a_cdom),
                               np.float32(0.8 * a_dg), rtol=1e-6)
    # A component of a, never larger than it (robust's IOPs.validate rule).
    assert np.all(np.asarray(inp.iops.a_cdom) <= np.asarray(inp.iops.a))


def test_eval_a_dg_matches_eval_anw_components(cdom_models):
    """``eval_a_dg`` is exactly ``eval_anw(retsub_comps=True)[0]``, in both
    the 1-D and batched shapes -- the API the adapter relies on."""
    a_model = cdom_models[0]
    a_dg_ref, _ = a_model.eval_anw(_A_PARAMS, retsub_comps=True)
    np.testing.assert_array_equal(a_model.eval_a_dg(_A_PARAMS), a_dg_ref)

    batch = np.tile(_A_PARAMS, (3, 1))
    a_dg_batch = a_model.eval_a_dg(batch)
    assert a_dg_batch.shape == (3, len(a_model.wave))
    for row in a_dg_batch:
        np.testing.assert_allclose(row, np.squeeze(a_dg_ref), rtol=1e-12)


# ===== (e) validation =====

class _FakeGeom:
    """Non-None geometry sentinel -- validate_rt_dict only checks
    ``geom is None`` (mirrors test_evaluate_robust.py)."""


def test_validate_rt_dict_rejects_cdom_with_gordon():
    rt_dict = {'rt_backend': 'gordon', 'include_CDOM_fl': True}
    with pytest.raises(ValueError, match='include_CDOM_fl'):
        rt_defs.validate_rt_dict(rt_dict)


def test_validate_rt_dict_rejects_cdom_with_robust_baseline():
    """robust_baseline is elastic-only by construction -- the same rule
    the other two inelastic flags already obey."""
    rt_dict = {'rt_backend': 'robust_baseline', 'include_CDOM_fl': True}
    with pytest.raises(ValueError, match='robust_baseline'):
        rt_defs.validate_rt_dict(rt_dict, geom=_FakeGeom())


def test_validate_rt_dict_rejects_cdom_for_model_without_a_dg():
    """An a-model with a single lumped a_nw (here 'Exp') cannot supply the
    CDOM source term; the error names the offending class."""
    wave = np.linspace(400., 700., 61)
    models = model_utils.init(['Exp', 'Pow'], wave)
    rt_dict = {'rt_backend': 'robust_ztt', 'include_CDOM_fl': True}

    with pytest.raises(ValueError, match='aNWExp'):
        rt_defs.validate_rt_dict(rt_dict, models=models, geom=_FakeGeom())


def test_validate_rt_dict_accepts_cdom_for_model_with_a_dg(cdom_models):
    rt_dict = {'rt_backend': 'robust_ztt', 'include_CDOM_fl': True}
    rt_defs.validate_rt_dict(rt_dict, models=cdom_models,
                             geom=_FakeGeom())  # no raise


def test_validate_rt_dict_skips_a_dg_check_without_models():
    """models=None skips the a_dg check rather than raising -- it has
    nothing to check against (the grid-check precedent)."""
    rt_dict = {'rt_backend': 'robust_ztt', 'include_CDOM_fl': True}
    rt_defs.validate_rt_dict(rt_dict, models=None,
                             geom=_FakeGeom())  # no raise


def test_eval_a_dg_raises_for_model_without_a_dg():
    """The model-level half of the same guard, for a direct call that
    bypassed validate_rt_dict."""
    wave = np.linspace(400., 700., 61)
    a_model, _ = model_utils.init(['Exp', 'Pow'], wave)
    assert a_model.has_a_dg is False
    with pytest.raises(ValueError, match='separable a_dg'):
        a_model.eval_a_dg(np.array([-1.0, 0.017]))


def test_adapter_rejects_cdom_with_robust_baseline(cdom_models, geom):
    """The adapter's own copy of the baseline rule (a direct call that
    skipped fit setup) raises rather than silently dropping the term."""
    rt_dict = dict(_BASE_RT, rt_backend='robust_baseline',
                   include_CDOM_fl=True)
    with pytest.raises(ValueError, match='robust_baseline'):
        _Rrs(cdom_models, rt_dict, geom)


# ===== (f) jit cache separation =====

def test_cdom_on_and_off_are_distinct_jit_cache_entries(cdom_models, geom):
    """cdom-on and cdom-off never share a compiled closure: the presence
    of ``CDOMFl`` on the ``Inelastic`` pytree changes both the traced
    program and the treedef, so it belongs in the lru_cache key exactly
    like raman/fluorescence/emission_shape. A repeat call with the same
    config hits the cache -- no rebuild, and one XLA compile."""
    rt_off = dict(_BASE_RT, include_Raman=True)
    rt_on = dict(rt_off, include_CDOM_fl=True)

    evaluate._robust_forward_jit.cache_clear()
    _Rrs(cdom_models, rt_off, geom)
    _Rrs(cdom_models, rt_on, geom)
    info = evaluate._robust_forward_jit.cache_info()
    assert info.misses == 2
    assert info.currsize == 2

    # ... and the repeat is a hit, not a third build.
    _Rrs(cdom_models, rt_on, geom)
    _Rrs(cdom_models, rt_off, geom)
    info = evaluate._robust_forward_jit.cache_info()
    assert info.misses == 2
    assert info.hits == 2

    # The cdom-on closure itself compiled exactly once.
    wave_key = np.asarray(cdom_models[0].wave, dtype=np.float64).tobytes()
    jit_on = evaluate._robust_forward_jit('ztt', (True, False, 'double', True),
                                          wave_key)
    assert jit_on._cache_size() == 1


def test_cdom_only_and_elastic_are_distinct_jit_cache_entries(cdom_models,
                                                              geom):
    """A cdom-only rt_dict keys a real inelastic entry, never the
    ``inelastic_key=None`` elastic one."""
    evaluate._robust_forward_jit.cache_clear()
    _Rrs(cdom_models, _BASE_RT, geom)
    _Rrs(cdom_models, dict(_BASE_RT, include_CDOM_fl=True), geom)
    info = evaluate._robust_forward_jit.cache_info()
    assert info.misses == 2
    assert info.currsize == 2


# ===== (g) MCMC round trip =====

# ExpBricaud (Adg, Sdg, Aph) + Pow (Bnw, beta), the cheap synthetic recipe
# shared with test_evaluate_robust.py's threading fixture.
_MCMC_TRUTH = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
_MCMC_P0 = np.array([-0.9, 0.014, -0.6, -1.9, 0.9])
_MCMC_BP_TRUE = 0.02


def test_mcmc_roundtrip_all_inelastic_terms_plus_fit_Bp():
    """Smoke: a short robust_ztt MCMC with Raman + Chl fluorescence +
    **CDOM fluorescence** and a free B_p runs end to end and reconstructs.

    Sized like ``test_evaluate_robust.py``'s ``test_fit_Bp_roundtrip_
    recovers_Bp`` (800 steps, 200 burn, 0.5 % assumed error) so it stays a
    few seconds. This is a wiring test, not an accuracy gate: it pins that
    the CDOM keys survive ``init_mcmc``'s bookkeeping, the fitters'
    validation and domain-check calls, ``log_prob``'s B_p peel, and
    ``reconstruct_from_chains`` -- with the truth generated through the
    same forward model the fit uses (so the answer is recoverable).
    """
    p = standard.expb_pow(wv_min=400., wv_max=700., variable_Gordon=False)
    wave = np.arange(400., 705., 5.)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    obs_geom = ObsGeometry(theta_s=30.)

    rt_dict = dict(rt_defs.rt_dict_from_p(p), rt_backend='robust_ztt',
                   include_Raman=True, include_Chl_fl=True,
                   include_CDOM_fl=True, fit_Bp=True)
    rt_defs.validate_rt_dict(rt_dict, models=models, geom=obs_geom)

    # Synthetic truth through the very same forward model.
    Rrs = chisq_fit.fit_func(wave, *_MCMC_TRUTH, _MCMC_BP_TRUE,
                             models=models, rt_dict=rt_dict, geom=obs_geom)
    assert np.all(np.isfinite(Rrs))
    varRrs = (0.005*Rrs)**2

    nsteps, nburn = 800, 200
    pdict = bing_inf.init_mcmc(models, nsteps=nsteps, nburn=nburn,
                               rt_dict=rt_dict)
    pdict['Chl'] = np.array([1.0])
    pdict['Y'] = None
    p0 = bing_inf.append_Bp_seed(_MCMC_P0, rt_dict)

    np.random.seed(1234)
    chains, idx = bing_inf.fit_one((Rrs, varRrs, p0, 0, obs_geom),
                                   models=models, pdict=pdict,
                                   chains_only=True, rt_dict=rt_dict)

    nmodel = sum(model.nparam for model in models)
    assert chains.shape == (nsteps, pdict['nwalkers'], nmodel + 1)
    assert np.all(np.isfinite(chains))

    # The model parameters land near the truth (loose: this is a smoke
    # test on a 5+1 parameter fit, not a convergence gate).
    med = np.median(chains[nburn:].reshape(-1, nmodel + 1), axis=0)
    np.testing.assert_allclose(med[:nmodel], _MCMC_TRUTH, atol=0.15)
    assert rt_defs.BP_PRIOR_PMIN <= med[-1] <= rt_defs.BP_PRIOR_PMAX

    # And the reconstruction path runs with CDOM on (a_cdom is rebuilt
    # from the same chain samples).  reconstruct_from_chains burns 7000
    # steps internally, so hand it a chain long enough to survive that:
    # the converged final step, repeated (the recipe test_evaluate_robust
    # uses for its own reconstruction pins).
    long_chains = np.repeat(chains[-1:], 7010, axis=0)
    out = evaluate.reconstruct_from_chains(
        models, long_chains, rt_dict, geom=obs_geom)
    assert len(out) == 8
    for arr in out:
        assert np.shape(arr) == (len(wave),)
        assert np.all(np.isfinite(arr))


def test_chisq_fit_runs_with_cdom(cdom_models, geom):
    """The least-squares path threads the CDOM keys too (fit_func ->
    calc_Rrs_from_models_robust), and recovers a noiseless truth."""
    p = standard.expb_pow(wv_min=400., wv_max=700., variable_Gordon=False)
    wave = np.arange(400., 705., 5.)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    rt_dict = dict(rt_defs.rt_dict_from_p(p), rt_backend='robust_ztt',
                   include_CDOM_fl=True)

    Rrs = chisq_fit.fit_func(wave, *_MCMC_TRUTH, models=models,
                             rt_dict=rt_dict, geom=geom)
    ans, cov, idx = chisq_fit.fit(
        (Rrs, (0.02*Rrs)**2, _MCMC_P0.copy(), 0, geom), models, rt_dict)
    np.testing.assert_allclose(ans, _MCMC_TRUTH, atol=1e-2)


# ===== (h) the raw-IOP entry point =====

def test_calc_Rrs_from_iops_robust_accepts_a_cdom(cdom_models, geom):
    """The raw-spectrum entry point takes ``a_cdom`` directly and agrees
    with the model-parameter path when handed the very spectrum the
    adapter would have built (``cdom_fraction * a_dg``)."""
    a_model, bb_model = cdom_models
    wave = a_model.wave
    rt_dict = dict(_BASE_RT, include_CDOM_fl=True, cdom_fraction=0.8)

    a = np.squeeze(a_model.eval_a(_A_PARAMS))
    bb = np.squeeze(bb_model.eval_bb(_BB_PARAMS))
    a_cdom = 0.8 * np.squeeze(a_model.eval_a_dg(_A_PARAMS))

    Rrs_raw = evaluate.calc_Rrs_from_iops_robust(
        a, bb, wave, rt_dict, geom=geom, a_cdom=a_cdom)
    Rrs_model = _Rrs(cdom_models, rt_dict, geom)
    np.testing.assert_allclose(np.squeeze(Rrs_raw), np.squeeze(Rrs_model),
                               rtol=1e-6, atol=0)


def test_calc_Rrs_from_iops_robust_requires_a_cdom(cdom_models, geom):
    """CDOM fluorescence on without ``a_cdom`` raises a clear error naming
    it -- bulk absorption cannot stand in for the CDOM component."""
    a_model, bb_model = cdom_models
    a = np.squeeze(a_model.eval_a(_A_PARAMS))
    bb = np.squeeze(bb_model.eval_bb(_BB_PARAMS))
    rt_dict = dict(_BASE_RT, include_CDOM_fl=True)

    with pytest.raises(ValueError, match='a_cdom'):
        evaluate.calc_Rrs_from_iops_robust(a, bb, a_model.wave, rt_dict,
                                           geom=geom)


def test_calc_Rrs_from_iops_robust_ignores_a_cdom_when_off(cdom_models, geom):
    """A supplied ``a_cdom`` with the process off is inert -- it never
    reaches ``IOPs.a_cdom``, so the off-state stays bit-identical (the
    ``a_ph`` precedent)."""
    a_model, bb_model = cdom_models
    wave = a_model.wave
    a = np.squeeze(a_model.eval_a(_A_PARAMS))
    bb = np.squeeze(bb_model.eval_bb(_BB_PARAMS))
    a_cdom = 0.8 * np.squeeze(a_model.eval_a_dg(_A_PARAMS))

    Rrs_none = evaluate.calc_Rrs_from_iops_robust(a, bb, wave, _BASE_RT,
                                                  geom=geom)
    Rrs_given = evaluate.calc_Rrs_from_iops_robust(a, bb, wave, _BASE_RT,
                                                   geom=geom, a_cdom=a_cdom)
    np.testing.assert_array_equal(Rrs_given, Rrs_none)
