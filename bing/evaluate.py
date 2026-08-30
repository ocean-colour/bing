"""
Post-Fitting Evaluation Module for BING
========================================

This module provides functions for analyzing and evaluating bio-optical
parameter retrievals from MCMC chains or least-squares fits.

Key functionality:
- Computing statistics (median, percentiles) from MCMC chains
- Reconstructing IOPs (absorption, backscattering) from parameters
- Computing model Rrs with uncertainties
- Processing chain arrays (burn-in removal, thinning)

The module bridges the gap between raw fitting outputs and scientific
analysis, providing properly formatted results with uncertainties.

Examples
--------
>>> from bing import evaluate
>>> from bing.models import utils as model_utils
>>>
>>> # Reconstruct IOPs from MCMC chains
>>> a_mean, bb_mean, a_lo, a_hi, bb_lo, bb_hi, Rrs, sigRrs = \\
...     evaluate.reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))
>>>
>>> # Get parameter statistics
>>> stats = evaluate.calc_stats(chains, names=['Adg', 'Sdg', 'Aph', 'Bnw', 'beta'])
"""

import collections
import functools
import warnings

import numpy as np

import jax
import jax.numpy as jnp

from bing.rt import rrs as bing_rrs
from bing.rt import defs as rt_defs
from bing.fitting import chisq_fit

from robust import rt as robust_rt

from IPython import embed

def calc_stats(chains, names:list=None,
               perc=(14, 86)):
    """
    Compute summary statistics from MCMC chains.

    Calculates median and percentile bounds for each parameter after
    applying burn-in removal and thinning.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains with shape (nsteps, nwalkers, nparam).
    names : list of str, optional
        Parameter names. If None, uses generic names ['p0', 'p1', ...].
    perc : tuple, optional
        Lower and upper percentiles for credible interval. Default is (14, 86),
        corresponding to approximately 1-sigma for a Gaussian.

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - 'names' : list of str - Parameter names
        - 'med' : np.ndarray - Median values for each parameter
        - 'pXX' : np.ndarray - Lower percentile values (e.g., 'p14')
        - 'pYY' : np.ndarray - Upper percentile values (e.g., 'p86')

    Notes
    -----
    Chains are automatically processed with thin_burn_chains() which
    removes burn-in (default 7000 steps) and flattens walker dimension.

    Examples
    --------
    >>> stats = calc_stats(chains, names=['Adg', 'Sdg', 'Aph', 'Bnw', 'beta'])
    >>> print(f"Adg = {10**stats['med'][0]:.4f} "
    ...       f"[{10**stats['p14'][0]:.4f}, {10**stats['p86'][0]:.4f}]")
    """
    # Thin/burn
    chains = thin_burn_chains(chains)

    # Names
    if names is None:
        names = [f'p{ii}' for ii in range(chains.shape[1])]

    # Simple stats
    stats = {}
    stats['names'] = names

    stats['med'] = np.median(chains, axis=0)
    stats[f'p{perc[0]:02d}'] = np.percentile(chains, perc[0], axis=0)
    stats[f'p{perc[1]:02d}'] = np.percentile(chains, perc[1], axis=0)

    return stats

def calc_Rrs_from_models(a_model, a_params, bb_model, bb_params, 
        rt_dict:dict, debug:bool=False, full_return:bool=False):
    """
    Calculate Rrs from model parameters using Gordon radiative transfer.

    This is the central forward model function that computes remote sensing
    reflectance from absorption and backscattering model parameters, optionally
    including wavelength-dependent Gordon coefficients and Raman correction.

    Parameters
    ----------
    a_model : aNWModel
        Absorption model object (e.g., aNWExpBricaud).
    a_params : np.ndarray
        Absorption model parameters. Shape can be (nparam,) for single
        evaluation or (nsamples, nparam) for batch evaluation.
    bb_model : bbNWModel
        Backscattering model object (e.g., bbNWPow).
    bb_params : np.ndarray
        Backscattering model parameters. Shape matches a_params.
    rt_dict : dict
        Radiative transfer configuration with keys:
        - 'variable_Gordon' : bool - Use wavelength-dependent G1, G2
        - 'include_Raman' : bool - Apply Raman scattering correction
    debug : bool, optional
        If True, print debug information. Default is False.
    full_return : bool, optional
        If True, return the full Rrs, a, bb arrays. Default is False.

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs [sr^-1]. Shape is (nwave,) for
        single evaluation or (nsamples, nwave) for batch.

    Raises
    ------
    ValueError
        If variable_Gordon=True but model G1/G2 are not set.

    Notes
    -----
    When include_Raman=True, the function computes IOPs at both emission
    and excitation wavelengths to calculate the Raman correction factor.

    See Also
    --------
    bing.rt.rrs.calc_Rrs : Low-level Rrs calculation
    reconstruct_from_chains : Higher-level function using this internally
    """
    # IOPs for model wave
    a = a_model.eval_a(a_params)
    bb = bb_model.eval_bb(bb_params)

    # Elastic
    if rt_dict['variable_Gordon'] and a_model.G1 is None:
        raise ValueError("Need to set model G1, G2 for variable Gordon")

    # Raman?
    if rt_dict['include_Raman']:
        # a_ex, bb_ex
        a_ex = a_model.eval_a_ex(a_params)
        bb_ex = bb_model.eval_bb_ex(bb_params)
        # bb_R
        if a_params.ndim == 1:
            bb_R = bb_model.bb_R
        else:
            bb_R = np.outer(np.ones(a_params.shape[0]),
                            bb_model.bb_R)
        # Ed(lambda')/Ed(lambda): use the true solar-spectrum ratio when
        # available (set via a_model.set_raman_Ed); otherwise fall back to
        # a flat spectrum (ratio = 1), which is known to distort the
        # spectral shape of the Raman correction.
        Ed_ratio = getattr(a_model, 'Ed_ratio_raman', None)
        if Ed_ratio is None:
            warnings.warn(
                "include_Raman: no Ed spectrum set on the a_nw model "
                "(call set_raman_Ed); falling back to a flat "
                "Ed(lambda')/Ed(lambda) = 1", RuntimeWarning)
    else:
        a_ex, bb_ex, bb_R = None, None, None
        Ed_ratio = None

    # bbp required when Gb mode is on. Two conventions:
    #   4-param (G0 AND Gb set): bbp(700 nm) as a trophic-state proxy.
    #   3-param Gb-only:        bbp(λ) at every wavelength.
    _G0 = getattr(a_model, 'G0', None)
    _Gb = getattr(a_model, 'Gb', None)
    if _Gb is not None:
        _bbnw_full = bb_model.eval_bbnw(bb_params)
        if _G0 is not None:
            j700 = int(np.argmin(np.abs(a_model.wave - 700.)))
            _bbp = _bbnw_full[..., j700:j700 + 1]  # broadcast over wavelength
        else:
            _bbp = _bbnw_full
    else:
        _bbp = None

    Rrs = bing_rrs.calc_Rrs(a, bb,
                            in_G1=a_model.G1, in_G2=a_model.G2,
                            in_G0=_G0,
                            in_Gb=_Gb, in_bbp=_bbp,
                            a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R,
                            Ed_ratio=Ed_ratio)

    # Call me
    if debug:
        embed(header='174 of evaluate.py')

    # RT correction?
    #  THIS SHOULD BE REMOVED
    if rt_dict.get('RT_correction', None) is not None:
        if a_params.ndim == 1:
            Rrs = Rrs * rt_dict['RT_correction']
        else:
            Rrs = Rrs * np.outer(np.ones(a_params.shape[0]), rt_dict['RT_correction'])

    # Fluorescence? Accept rt_dicts that don't specify the key (ad-hoc dicts
    # built by tests / notebooks pre-date the include_Chl_fl field).
    if rt_dict.get('include_Chl_fl', False):
        # Batch (chains) is now supported: calc_Rrs_fluorescence loops over the
        # small emission axis instead of forming a 3-D (n_samples, n_em, n_ex)
        # tensor, so reconstruct_from_chains stays within a few GiB even at
        # nsteps=40000.  See dev/ChlFl/memory_profile.py.
        # a_ph -- shape (nwave,) for 1-D params, (nsample, nwave) for batch.
        # Use ellipsis-indexing so the wavelength slice works in both cases
        # (plain `aph[i_Chl_ex]` would silently index along the sample axis
        # when called from reconstruct_from_chains).
        aph = (10**a_params[...,-1:]) * a_model.a_ph
        aph_ex = aph[..., a_model.i_Chl_ex]

        # Call me
        Rrs_fl = bing_rrs.calc_Rrs_fluorescence(
            a_model.wave, a, bb,
            a[:,a_model.i_Chl_ex],
            bb[:,a_model.i_Chl_ex],
            aph_ex,
            a_model.wave[a_model.i_Chl_ex],
            a_model.Ed_ex,
            a_model.Ed_em,
            phi_C=rt_dict['phi_C'],
            double_gaussian=rt_dict['double_gaussian'])
        # Add
        Rrs += Rrs_fl

    # Return
    if full_return:
        return Rrs, a, bb
    else:
        return Rrs

@functools.lru_cache(maxsize=None)
def _robust_forward_jit(mode, inelastic_key, wave_key):
    """
    Build (and cache) a jax.jit'd closure for one (mode, inelastic
    configuration, wavelength grid) combination.

    rob_rt integration, M1 task 2 -- plan choice, docs/design/rob_rt_design.md
    §7.1. One compile per distinct combination, reused for the rest of the
    fit. Deliberately a `functools.lru_cache` on top of (not instead of)
    JAX's own compilation cache: this gives a stable, introspectable
    `_robust_forward_jit.cache_info()` for the M1 gate's "second call with
    identical config hits the cache, no recompile" check, which JAX's own
    internal cache doesn't expose as directly.

    `wave` is baked into the closure as a Python-level constant (via
    `wave_key = wave.tobytes()`, assumed float64 -- BING's convention) rather
    than passed as a traced argument: it never varies within a fit, so
    there is no reason to let it participate in tracing, and doing so this
    way keeps the traced signature to exactly the things that do vary
    (`iops`, `phase_params`, `geometry`, and -- when inelastic is on --
    `phi_C`). No `vmap` is used or needed: `robust.rt.forward`/
    `baselines.Rrs_gordon` are natively batched over leading axes, so one
    call handles a whole `(nsamples, nparam)` chain.

    NumPy crosses to JAX only at the boundary of the returned closure --
    plain float64 `IOPs`/`PhaseParams`/`Geometry` built outside `jit` are
    converted to JAX's default dtype (float32; `jax_enable_x64` is never
    enabled, CQ1) the moment they are passed into it. Nothing in this
    module casts dtypes explicitly; the jit boundary does it.

    Parameters
    ----------
    mode : str
        'ztt', 'hybrid', or 'baseline' (`rt_dict['rt_backend']` with its
        'robust_' prefix stripped, or literally 'baseline').
    inelastic_key : tuple or None
        `(raman: bool, fluorescence: bool, emission_shape: str)` -- the
        *static* `Inelastic` configuration -- or `None` for the
        elastic-only path. `None` is not the same as
        `Inelastic(raman=False, fluorescence=False)`: passing `None` to
        `robust.rt.forward` takes the pre-existing, bit-identical code
        route (robust.rt.types.Inelastic docstring; design §3.5), so the
        two are kept as genuinely different cache entries / closures,
        never conflated.
    wave_key : bytes
        `a_model.wave` as float64 bytes (`np.asarray(wave,
        dtype=np.float64).tobytes()`).

    Returns
    -------
    callable
        A `jax.jit`-wrapped function. Signature is `f(iops, phase_params,
        geometry)` when `mode == 'baseline'` or `inelastic_key is None`;
        `f(iops, phase_params, geometry, phi_C)` otherwise (`phi_C` is the
        one `Inelastic` field that is a real traced leaf, not static).
    """
    wave = jnp.asarray(np.frombuffer(wave_key, dtype=np.float64))

    # mode='hybrid' needs the trained emulator. forward()'s own default
    # (emulator=None) lazily loads it from disk on first use
    # (hybrid.py:_resolve_emulator -> emulator.load_default()) -- a side
    # effect that raises jax.errors.UnexpectedTracerError the first time it
    # happens inside a jit trace (found by running this exact code path,
    # not anticipated in advance -- the same failure mode as
    # corrections=None below, just a second, independent lazy-load). Fixed
    # the same way: load it here, once, outside jit (robust's own
    # load_default() is itself memoised -- "read once per process" -- so
    # this is not a redundant read), and pass the already-loaded object
    # explicitly so forward() never has a reason to load anything at trace
    # time.
    emulator_obj = robust_rt.emulator.load_default() if mode == 'hybrid' else None

    if mode == 'baseline':
        def _fn(iops, phase_params, geometry):
            return robust_rt.baselines.Rrs_gordon(iops, phase_params,
                                                   geometry, wave)
        return jax.jit(_fn)

    if inelastic_key is None:
        def _fn(iops, phase_params, geometry):
            return robust_rt.forward(iops, phase_params, geometry, wave,
                                     mode=mode, inelastic=None,
                                     corrections=False, emulator=emulator_obj)
        return jax.jit(_fn)

    raman, fluorescence, emission_shape = inelastic_key

    def _fn(iops, phase_params, geometry, phi_C):
        inelastic = robust_rt.Inelastic(
            raman=raman, fluorescence=fluorescence, phi_C=phi_C,
            emission_shape=emission_shape)
        # corrections=False: explicit analytic-only inelastic physics (no
        # M3 learned correction heads). Not a simplification of scope --
        # it is the only inelastic behavior this integration ever resolved
        # or cross-checked (claude_prompts/rob_rt.md, robust's own
        # test_inelastic_bing_xcheck.py). It is also load-bearing for
        # jit-safety, same reason as the emulator above: corrections=None
        # (the forward() default) tries to lazily load trained
        # correction-head weights from disk on first use
        # (hybrid.py:_resolve_corrections -> inelastic_corr.load_default()),
        # which raises jax.errors.UnexpectedTracerError under jit.
        return robust_rt.forward(iops, phase_params, geometry, wave,
                                 mode=mode, inelastic=inelastic,
                                 corrections=False, emulator=emulator_obj)
    return jax.jit(_fn)

#: Everything a robust.rt call needs, built once from BING-side arguments.
#: Shared by `calc_Rrs_from_models_robust` (the jitted hot path) and
#: `robust_domain_check` (the un-jitted diagnostic) so the BING -> robust
#: argument mapping is defined in exactly one place -- the two callers can
#: never drift apart on how IOPs/PhaseParams/Geometry are constructed.
_RobustInputs = collections.namedtuple(
    '_RobustInputs',
    ['rt_backend', 'iops', 'phase_params', 'geometry', 'a', 'bb',
     'include_raman', 'include_fl', 'emission_shape', 'phi_C'])


def _build_robust_inputs(a_model, a_params, bb_model, bb_params,
                         rt_dict, geom, Bp):
    """
    Validate and map BING-side arguments onto robust.rt call inputs.

    The single definition of the BING -> robust argument construction
    (design §3.4's mapping table): parameter evaluation via
    ``a_model.eval_a``/``bb_model.eval_bb`` (identical to the Gordon path),
    the ``IOPs.from_total_bb`` split, ``PhaseParams(B_p=...)`` with the
    free-``Bp`` override, ``geom.to_robust()`` (no ``Ed`` -- M4's job), and
    the inelastic flags. Also owns the three argument-validity errors
    (missing ``geom``, a non-robust ``rt_backend``, ``robust_baseline`` +
    inelastic), so a direct call to either consumer fails identically.

    Parameters and error semantics are exactly those documented on
    `calc_Rrs_from_models_robust`; see its docstring.

    Returns
    -------
    _RobustInputs
        The validated, fully-constructed robust.rt call ingredients
        (plus the raw ``a``/``bb`` arrays for ``full_return``).
    """
    if geom is None:
        raise ValueError(
            "calc_Rrs_from_models_robust requires geom (a bing.rt.geometry."
            "ObsGeometry) -- theta_s is never silently defaulted "
            "(claude_prompts/rob_rt.md, Q&A/Coding item 4). "
            "rt_defs.validate_rt_dict should have raised before this "
            "function was ever reached for a robust backend.")

    rt_backend = rt_dict.get('rt_backend', 'gordon')
    if rt_backend not in rt_defs.RT_BACKENDS or rt_backend == 'gordon':
        raise ValueError(
            f"calc_Rrs_from_models_robust: rt_dict['rt_backend']={rt_backend!r} "
            "is not a robust backend -- use one of "
            f"{[b for b in rt_defs.RT_BACKENDS if b != 'gordon']}, or "
            "dispatch 'gordon' to calc_Rrs_from_models instead.")

    include_raman = rt_dict.get('include_Raman', False)
    include_fl = rt_dict.get('include_Chl_fl', False)

    if rt_backend == 'robust_baseline' and (include_raman or include_fl):
        raise ValueError(
            "rt_dict['rt_backend']='robust_baseline' has no inelastic "
            "composition path -- robust.rt.baselines.Rrs_gordon takes "
            "no `inelastic` argument and is elastic-only by "
            "construction. Disable include_Raman/include_Chl_fl, or "
            "use rt_backend='robust_ztt'/'robust_hybrid' instead.")

    # IOPs for model wave -- identical evaluation to the Gordon path.
    a = a_model.eval_a(a_params)
    bb = bb_model.eval_bb(bb_params)

    # Fluorescence source term (full spectrum on a_model.wave -- see
    # calc_Rrs_from_models_robust's Notes; not sliced at a_model.i_Chl_ex
    # like calc_Rrs_from_models' aph_ex).
    a_ph = (10**a_params[..., -1:]) * a_model.a_ph if include_fl else None

    iops = robust_rt.IOPs.from_total_bb(a, bb, wave=a_model.wave, a_ph=a_ph)

    B_p = Bp if Bp is not None else rt_dict['Bp_value']
    phase_params = robust_rt.PhaseParams(B_p=B_p)

    geometry = geom.to_robust()

    emission_shape = ('double' if rt_dict.get('double_gaussian', True)
                      else 'single')

    return _RobustInputs(
        rt_backend=rt_backend, iops=iops, phase_params=phase_params,
        geometry=geometry, a=a, bb=bb, include_raman=include_raman,
        include_fl=include_fl, emission_shape=emission_shape,
        phi_C=rt_dict.get('phi_C', 0.02))


def calc_Rrs_from_models_robust(a_model, a_params, bb_model, bb_params,
        rt_dict:dict, geom=None, Bp:float=None, debug:bool=False,
        full_return:bool=False):
    """
    Calculate Rrs from model parameters using the robust.rt backend
    (rob_rt integration, M1).

    Sibling of calc_Rrs_from_models: same parameter-to-IOP evaluation
    (a_model.eval_a / bb_model.eval_bb, unchanged) and the same batch shape
    contract, but the IOPs -> Rrs step is retrieve-or-bust's robust.rt
    forward model instead of the Gordon relation. Called only for a robust
    rt_dict['rt_backend'] value -- the Gordon path stays on
    calc_Rrs_from_models unchanged. See docs/design/rob_rt_design.md §3.4
    for the full mapping table this implements, and claude_prompts/rob_rt.md
    (Q&A/Design 4, 13; Q&A/Coding 1) for the decisions behind it.

    Runs entirely at float32 -- robust.rt never enables jax_enable_x64
    (Q&A/Coding item 1), and BING's float64 NumPy arrays downcast crossing
    the JAX boundary here. Documented, not a bug: float32 is more than
    sufficient precision for these calculations, and no test tolerance
    against this function assumes float64 headroom.

    The actual IOPs -> Rrs call is dispatched through `_robust_forward_jit`,
    an `lru_cache`d builder of `jax.jit`'d closures (one compile per
    distinct `(mode, inelastic configuration, wavelength grid)`
    combination, reused for the rest of the fit -- see its own docstring
    for the caching design, M1 task 2).

    Parameters
    ----------
    a_model : aNWModel
        Absorption model object (e.g., aNWExpBricaud).
    a_params : np.ndarray
        Absorption model parameters. Shape can be (nparam,) for single
        evaluation or (nsamples, nparam) for batch evaluation.
    bb_model : bbNWModel
        Backscattering model object (e.g., bbNWPow).
    bb_params : np.ndarray
        Backscattering model parameters. Shape matches a_params.
    rt_dict : dict
        Radiative transfer configuration. Consulted keys:
        - 'rt_backend' : str - one of 'robust_ztt'/'robust_hybrid'/
          'robust_baseline' (see bing.rt.defs.RT_BACKENDS). 'gordon' is a
          caller error here -- dispatch to calc_Rrs_from_models instead.
        - 'include_Raman', 'include_Chl_fl', 'phi_C', 'double_gaussian' :
          same meaning as for calc_Rrs_from_models.
        - 'Bp_value' : float - constant B_p when Bp is not given.
    geom : bing.rt.geometry.ObsGeometry, optional
        Fixed per-pixel viewing/illumination geometry. Required (non-None)
        for every robust backend -- theta_s is never silently defaulted
        (claude_prompts/rob_rt.md, Q&A/Coding item 4). Fitters raise via
        rt_defs.validate_rt_dict before this function is ever reached; a
        direct call is held to the same rule.
    Bp : float, optional
        Free-parameter override for the particle phase-function ratio
        B_p = bb_p/b_p (design §3.3). Falls back to rt_dict['Bp_value']
        when None (the fixed-B_p, default case).
    debug : bool, optional
        If True, drop into an IPython shell after computing Rrs.
    full_return : bool, optional
        If True, return the full Rrs, a, bb arrays. Default is False.

    Returns
    -------
    np.ndarray
        Remote sensing reflectance Rrs [sr^-1]. Shape is (nwave,) for
        single evaluation or (nsamples, nwave) for batch.

    Raises
    ------
    ValueError
        If geom is None; if rt_dict['rt_backend'] is not a robust backend;
        or if rt_dict['rt_backend'] == 'robust_baseline' while Raman or
        chlorophyll fluorescence is requested (robust.rt.baselines.Rrs_gordon
        takes no `inelastic` argument -- it is elastic-only by
        construction, see Notes).

    Notes
    -----
    Unlike calc_Rrs_from_models, this function never evaluates a_model/
    bb_model at separate Raman-excitation wavelengths (eval_a_ex/eval_bb_ex):
    robust.rt.inelastic derives its own excitation-grid IOPs by interpolating
    (and, outside the supplied wave range, clamping) the single emission-grid
    IOPs passed in here. This is a real, load-bearing difference from BING's
    own Raman path, which evaluates the true parametric models at the wider
    excitation grid -- not a bug in this adapter, but an inherent property of
    delegating to robust.rt's public forward()/rrs_forward() API, which has
    no parameter for separately-evaluated excitation IOPs. Likewise a_ph is
    passed as the *full* spectrum on a_model.wave, not pre-sliced at
    a_model.i_Chl_ex -- robust.rt.inelastic.fluorescence_kernel interpolates
    onto its own fixed 370-690 nm excitation grid internally.

    'robust_baseline' has no inelastic composition path at all: it dispatches
    directly to robust.rt.baselines.Rrs_gordon (elastic-only by construction,
    the point of the like-for-like Gordon comparison), bypassing forward()
    entirely. Requesting Raman/fluorescence with 'robust_baseline' raises
    rather than silently dropping the inelastic terms.

    See Also
    --------
    calc_Rrs_from_models : the Gordon-backend sibling this mirrors.
    bing.rt.defs.validate_rt_dict : the fit-setup checks this function
        assumes have already run.
    """
    inp = _build_robust_inputs(a_model, a_params, bb_model, bb_params,
                               rt_dict, geom, Bp)

    wave_key = np.asarray(a_model.wave, dtype=np.float64).tobytes()

    if inp.rt_backend == 'robust_baseline':
        jit_fn = _robust_forward_jit('baseline', None, wave_key)
        Rrs = jit_fn(inp.iops, inp.phase_params, inp.geometry)
    else:
        # 'robust_ztt' -> mode='ztt'; 'robust_hybrid' -> mode='hybrid'.
        mode = inp.rt_backend[len('robust_'):]
        if inp.include_raman or inp.include_fl:
            inelastic_key = (inp.include_raman, inp.include_fl,
                             inp.emission_shape)
            jit_fn = _robust_forward_jit(mode, inelastic_key, wave_key)
            Rrs = jit_fn(inp.iops, inp.phase_params, inp.geometry, inp.phi_C)
        else:
            jit_fn = _robust_forward_jit(mode, None, wave_key)
            Rrs = jit_fn(inp.iops, inp.phase_params, inp.geometry)

    Rrs = np.asarray(Rrs)
    a, bb = inp.a, inp.bb

    # Call me
    if debug:
        embed(header='calc_Rrs_from_models_robust of evaluate.py')

    # Return
    if full_return:
        return Rrs, a, bb
    else:
        return Rrs


def robust_domain_check(a_model, a_params, bb_model, bb_params,
                        rt_dict:dict, geom=None, Bp:float=None):
    """
    Run robust.rt's out-of-domain check on concrete arrays, un-jitted, so
    its `DomainWarning` can actually fire (rob_rt integration, M1 task 3;
    design docs/design/rob_rt_design.md §4).

    The emulator's domain check (`robust.rt.hybrid._check_domain`,
    hybrid.py:139-163) is deliberately skipped whenever any input is a JAX
    tracer -- it needs concrete values to compare against the trained
    ranges -- so on the fitting hot path, which always goes through
    `_robust_forward_jit`'s `jax.jit`-wrapped closures, an out-of-domain
    evaluation is *silent by construction* (design §4; the warn-and-continue
    policy of claude_prompts/rob_rt.md Q8). This helper is the sanctioned
    complement: it rebuilds the exact same robust.rt call arguments as
    `calc_Rrs_from_models_robust` (through the shared `_build_robust_inputs`,
    so the two can never drift) and calls the **un-jitted**
    `robust.rt.forward` once on the concrete NumPy-backed inputs, letting
    `robust.rt.hybrid.DomainWarning` propagate to the caller. Fitters call
    it twice per fit (M2): on the initial guess before sampling and on the
    posterior median after -- never inside the hot loop.

    Only ``rt_backend='robust_hybrid'`` has a domain to check: the check
    lives past `forward()`'s ``mode='ztt'`` early return (hybrid.py:304),
    and `robust.rt.baselines.Rrs_gordon` (the ``robust_baseline`` dispatch
    target) is a closed-form expression with no emulator and no domain
    logic at all -- both confirmed by reading robust's source directly.
    For ``robust_ztt``/``robust_baseline`` this function is therefore a
    validated no-op: it still runs `_build_robust_inputs` (so the same
    argument errors raise as on the hot path) but performs no forward call.

    Runs the same `corrections=False` / explicitly-loaded-emulator
    configuration as `_robust_forward_jit` (see Q6 in
    claude_prompts/RT/rob_rt_prompt_2.md): not for jit-safety here (nothing
    is traced), but so the domain check evaluates the *identical* forward
    configuration the fit itself uses. `robust.rt.emulator.load_default()`
    is memoised process-wide, so this adds no I/O beyond the fit's own.

    Parameters
    ----------
    a_model : aNWModel
        Absorption model object (e.g., aNWExpBricaud).
    a_params : np.ndarray
        Absorption model parameters, ``(nparam,)`` or ``(nsamples, nparam)``
        -- e.g. the initial guess, or the posterior median.
    bb_model : bbNWModel
        Backscattering model object (e.g., bbNWPow).
    bb_params : np.ndarray
        Backscattering model parameters. Shape matches a_params.
    rt_dict : dict
        Radiative transfer configuration -- same keys as
        `calc_Rrs_from_models_robust`.
    geom : bing.rt.geometry.ObsGeometry, optional
        Viewing/illumination geometry. Required (non-None), same rule as
        the adapter.
    Bp : float, optional
        Free-parameter override for B_p; falls back to
        rt_dict['Bp_value'] when None.

    Returns
    -------
    None
        This is a diagnostic side-effect function: it exists to let
        `DomainWarning` reach the caller's warning filters, not to return
        Rrs -- use `calc_Rrs_from_models_robust` for values.

    Warns
    -----
    robust.rt.hybrid.DomainWarning
        If any input lies outside the emulator's training range
        (``robust_hybrid`` only). Callers that must not extrapolate can
        promote it: ``warnings.simplefilter('error', DomainWarning)``.

    Raises
    ------
    ValueError
        Same argument-validity errors as `calc_Rrs_from_models_robust`
        (missing geom, non-robust backend, robust_baseline + inelastic).

    See Also
    --------
    calc_Rrs_from_models_robust : the jitted hot-path twin whose inputs
        this function checks.
    """
    inp = _build_robust_inputs(a_model, a_params, bb_model, bb_params,
                               rt_dict, geom, Bp)

    # Only the hybrid backend carries an emulator, hence a trained domain.
    if inp.rt_backend != 'robust_hybrid':
        return

    inelastic = None
    if inp.include_raman or inp.include_fl:
        inelastic = robust_rt.Inelastic(
            raman=inp.include_raman, fluorescence=inp.include_fl,
            phi_C=inp.phi_C, emission_shape=inp.emission_shape)

    # Un-jitted, concrete-array call: the whole point. Same
    # corrections=False / explicit-emulator configuration as the jitted
    # closures (Q6) so the checked configuration is the fitted one.
    emulator_obj = robust_rt.emulator.load_default()
    wave = np.asarray(a_model.wave, dtype=np.float64)
    robust_rt.forward(inp.iops, inp.phase_params, inp.geometry, wave,
                      mode='hybrid', inelastic=inelastic,
                      corrections=False, emulator=emulator_obj)


def reconstruct_from_chains(models:list, chains:np.ndarray, rt_dict:dict,
                            perc=(5,95)):
    """
    Reconstruct IOPs and Rrs with uncertainties from MCMC chains.

    Evaluates the absorption and backscattering models for all chain samples
    to compute posterior distributions of IOPs and Rrs, then summarizes with
    median and percentile statistics.

    Parameters
    ----------
    models : list
        List of two model objects: [absorption_model, backscattering_model].
    chains : np.ndarray
        MCMC chains with shape (nsteps, nwalkers, nparam).
    rt_dict : dict
        Radiative transfer configuration dictionary.
    perc : tuple, optional
        Percentiles for credible interval bounds. Default is (5, 95),
        giving a 90% credible interval.

    Returns
    -------
    a_mean : np.ndarray
        Median total absorption coefficient at each wavelength [m^-1].
    bb_mean : np.ndarray
        Median total backscattering coefficient at each wavelength [m^-1].
    a_low : np.ndarray
        Lower percentile of absorption [m^-1].
    a_high : np.ndarray
        Upper percentile of absorption [m^-1].
    bb_low : np.ndarray
        Lower percentile of backscattering [m^-1].
    bb_high : np.ndarray
        Upper percentile of backscattering [m^-1].
    Rrs : np.ndarray
        Median model Rrs at each wavelength [sr^-1].
    sigRrs : np.ndarray
        Standard deviation of Rrs at each wavelength [sr^-1].

    Notes
    -----
    Chains are processed with thin_burn_chains() before evaluation.
    This removes burn-in and flattens the walker dimension.

    The function handles Raman correction if rt_dict['include_Raman']=True,
    computing IOPs at both emission and excitation wavelengths.

    Memory usage can be significant for long chains since all samples
    are evaluated simultaneously.

    Examples
    --------
    >>> a_med, bb_med, a_lo, a_hi, bb_lo, bb_hi, Rrs, sigRrs = \\
    ...     reconstruct_from_chains(models, chains, rt_dict, perc=(5, 95))
    >>> plt.fill_between(wave, a_lo, a_hi, alpha=0.3)
    >>> plt.plot(wave, a_med)
    """
    # Burn/thin the chains
    chains = thin_burn_chains(chains)

    # Split parameters once
    aparams = chains[..., :models[0].nparam]
    bparams = chains[..., models[0].nparam:]

    # Forward-model Rrs through the shared helper so the elastic, Raman,
    # G0/Gb, and fluorescence branches stay defined in a single place
    # (also used by inference.log_prob and chisq_fit.fit_func).
    #embed(header='287 of evaluate.py')
    Rrs, a, bb = calc_Rrs_from_models(models[0], aparams,
                               models[1], bparams, rt_dict,
                               full_return=True)
                               #debug=True)

    # Stats over the Rrs posterior
    sigRs = np.std(Rrs, axis=0)
    Rrs = np.median(Rrs, axis=0)

    # IOPs at every chain sample for the credible bands on a, bb.
    #a = models[0].eval_a(aparams)
    #bb = models[1].eval_bb(bparams)
    a_mean = np.median(a, axis=0)
    a_low, a_high = np.percentile(a, perc, axis=0)
    bb_mean = np.median(bb, axis=0)
    bb_low, bb_high = np.percentile(bb, perc, axis=0)

    # Return
    return a_mean, bb_mean, a_low, a_high, bb_low, bb_high, Rrs, sigRs


def reconstruct_chisq_fits(models:list, params:np.ndarray, rt_dict:dict,
                           Chl:np.ndarray=None,
                           bb_basis_params:np.ndarray=None):
    """
    Reconstructs the parameters and calculates statistics from chisq fits.

    Parameters:
        - models (list): A list of model objects.
        - params (ndarray): An array of the best-fit paramerers
            if ndim==1, then it is one fit
            if ndim==2, then it is an (nfits, nparams) array of fits
        - rt_dict (dict): dict describing the Radiative transfer
        - Chl (ndarray): The chlorophyll values to use for the fits. Default is None.
        - bb_basis_params (ndarray): The basis parameters to use for the fits. Default is None.
            (nspec, nparams)


    Returns:
        - a_mean (ndarray): The mean of the parameter 'a' across the fits.
        - bb_mean (ndarray): The mean of the parameter 'bb' across the fits.
        - a_5 (ndarray): The 5th percentile of the parameter 'a' across the fits.
        - a_95 (ndarray): The 95th percentile of the parameter 'a' across the fits.
        - bb_5 (ndarray): The 5th percentile of the parameter 'bb' across the fits.
        - bb_95 (ndarray): The 95th percentile of the parameter 'bb' across the fits.
        - Rrs (ndarray): The calculated model Rrs.
        - sigRs (ndarray): The standard deviation of Rrs.

    """
    all_Rrs = []
    all_a = []
    all_bb = []
    # Fit
    in_ndim = params.ndim
    if params.ndim == 1:
        params = params.reshape(1, -1)

    for ss, param in enumerate(params):
        # Chl?
        if models[0].uses_Chl:
            models[0].set_aph(np.atleast_1d(Chl)[ss])
        # Lee?
        if models[1].uses_basis_params:
            models[1].set_basis_func(np.atleast_1d(bb_basis_params)[ss])
        model_Rrs, a_mean, bb_mean = chisq_fit.fit_func(
            models[0].wave, *param, models=models, return_full=True,
            rt_dict=rt_dict)
        # Save
        all_Rrs.append(model_Rrs)
        all_a.append(a_mean)
        all_bb.append(bb_mean)

    # Flatten?
    if in_ndim == 1:
        all_Rrs = all_Rrs[0]
        all_a = all_a[0]
        all_bb = all_bb[0]

    # Return
    return np.array(all_Rrs), np.array(all_a), np.array(all_bb)


def thin_burn_chains(chains:np.ndarray,
                     burn:int=7000, thin:int=1):
    """
    Remove burn-in and thin MCMC chains.

    Processes raw MCMC chains by removing initial burn-in samples,
    applying optional thinning, and flattening the walker dimension.

    Parameters
    ----------
    chains : np.ndarray
        Raw MCMC chains with shape (nsteps, nwalkers, nparam).
    burn : int, optional
        Number of initial steps to discard as burn-in. Default is 7000.
    thin : int, optional
        Thinning factor (keep every thin-th sample). Default is 1 (no thinning).

    Returns
    -------
    np.ndarray
        Processed chains with shape (nsamples, nparam), where
        nsamples = (nsteps - burn) // thin * nwalkers.

    Notes
    -----
    The walker dimension is flattened, treating all walkers as independent
    samples from the posterior. This is valid after burn-in when walkers
    have converged to sampling the same distribution.

    Examples
    --------
    >>> chains.shape
    (40000, 16, 5)
    >>> processed = thin_burn_chains(chains, burn=7000, thin=1)
    >>> processed.shape
    (528000, 5)  # (40000-7000) * 16
    """
    # Burn/thin the chains
    return chains[burn::thin, :, :].reshape(-1, chains.shape[-1])