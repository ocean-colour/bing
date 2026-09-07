
import numpy as np

# RT dict

#: Valid values of rt_dict['rt_backend']. 'gordon' is the existing Gordon
#: (1988) elastic + bing.rt raman/chl_fl path; the 'robust_*' values select
#: retrieve-or-bust's robust.rt forward model at increasing fidelity
#: (ztt: analytic only; hybrid: ztt + learned emulator correction;
#: baseline: robust's Gordon-compatible refit). See docs/design/rob_rt_design.md
#: §3.1.
RT_BACKENDS = ('gordon', 'robust_ztt', 'robust_hybrid', 'robust_baseline')

#: robust.rt's hybrid-mode emulator is only valid inside this wavelength
#: range (its L23/HydroLight training domain); see
#: docs/design/rob_rt_design.md §4.
ROBUST_HYBRID_WAVE_MIN = 350.
ROBUST_HYBRID_WAVE_MAX = 750.

#: Default prior range for a *free* B_p (rt_dict['fit_Bp']=True): a
#: **linear-space** uniform over [BP_PRIOR_PMIN, BP_PRIOR_PMAX] -- B_p is a
#: ratio (bb_p/b_p), like the slopes, not a log10 amplitude (plan choice,
#: docs/design/rob_rt_design.md §7.3 -> coding-plan M3).  The range sits
#: inside robust's PhaseParams.validate definitional (0, 1] bound.  Bounds
#: are inclusive, matching bing.priors.priors.UniformPrior's convention.
#: Consumers: inference.log_prob evaluates the prior (via
#: inference.BP_PRIOR); the fitters' bounds machinery
#: (inference.prior_bounds / init_walkers -- M3 task 3 -- and the
#: chi-squared bounds built in l23.fit_with_LM) reuses the same numbers so
#: sampling, clipping, and optimization all agree.
BP_PRIOR_PMIN = 0.004
BP_PRIOR_PMAX = 0.05

#: Default value of rt_dict['cdom_fraction'] -- the fraction of BING's
#: **combined** dissolved+detrital absorption a_dg that is taken to be
#: *pure CDOM* when the robust CDOM-fluorescence term is switched on
#: (``rt_dict['include_CDOM_fl']``):
#:
#: .. math:: a_{cdom}(\\lambda) = f_{cdom} \\times a_{dg}(\\lambda)
#:
#: **This is a fixed-fraction proxy, not a retrieval.** robust's Hawes
#: (1992) kernel (``robust.rt.cdom_fl.cdom_kernel``) wants the *pure*
#: CDOM absorption as its emission source term (``b_bY = 0.5 a_cdom``),
#: but every BING a-model with a separable exponential term lumps CDOM
#: and detritus together into one a_dg -- there is no free parameter that
#: splits them. The fraction 0.8 is a project decision (JXP, 2026-09-05;
#: ``claude_prompts/rt_tests.md`` Q32: "a_cdom = 0.8 x a_dg -- fixed CDOM
#: fraction 0.8, to be documented prominently in code, provenance, and
#: report text"), *not* a measured or fitted quantity. Override it per
#: fit with ``p.cdom_fraction`` / ``rt_dict['cdom_fraction']``.
CDOM_FRACTION_DEFAULT = 0.8


def rt_dict_from_p(p):
    """
    Prepare data and models for L23 fitting.
    This function initializes the necessary data, models, priors, and MCMC
    parameters for fitting L23 data. It also handles wavelength conversions,
    noise scaling, and initial guesses for the fitting process.

    Args:
        p (object): Parameter object containing configuration
            radiative transfer options. Recognized RT-backend attributes
            (all optional -- missing attributes fall back to the Gordon-
            backend defaults below, so a legacy `p` still yields a valid
            dict):

            rt_backend (str): which forward model computes Rrs from
                (a, bb). One of the four values in RT_BACKENDS, default
                ``'gordon'``:

                - ``'gordon'`` -- BING's own Gordon (1988) elastic model
                  (bing.rt.rrs), with Raman/fluorescence composed via
                  bing.rt.raman / bing.rt.chl_fl when
                  include_Raman/include_Chl_fl are set. No phase-function
                  input, so fit_Bp=True is invalid with this backend
                  (validate_rt_dict raises).
                - ``'robust_ztt'`` -- retrieve-or-bust's analytic
                  (Zaneveld/Twardowski/Tassan-style) forward model.
                  Supports include_Raman/include_Chl_fl via
                  robust.rt.inelastic.
                - ``'robust_hybrid'`` -- robust_ztt plus a learned
                  emulator correction, valid only for wavelengths inside
                  [ROBUST_HYBRID_WAVE_MIN, ROBUST_HYBRID_WAVE_MAX] nm (its
                  L23/HydroLight training domain; validate_rt_dict raises
                  outside that range). Also supports
                  include_Raman/include_Chl_fl.
                - ``'robust_baseline'`` -- robust's Gordon-compatible
                  refit. **Elastic only**: passing include_Raman=True or
                  include_Chl_fl=True with this backend raises ValueError
                  in calc_Rrs_from_models_robust (robust.rt.baselines.
                  Rrs_gordon has no `inelastic` argument by construction)
                  -- use 'robust_ztt'/'robust_hybrid' for an inelastic fit
                  on the robust side.

                See docs/design/rob_rt_design.md §3.1 and
                bing/CLAUDE.md's rt-subpackage bullet. NOTE: as of M4/M5,
                robust's inelastic (Raman/fluorescence) terms are wired
                and tested but measured to diverge from BING's own
                Gordon+Raman/fluorescence physics by ~5-18% on real L23
                data (bing/.claude/skills/inelastic-rrs/SKILL.md has the
                full numbers and caveats) -- 'robust_ztt'/'robust_hybrid'
                are a real, selectable alternative for the elastic terms
                (agreement ~2.3e-7 relative to 'gordon' via
                'robust_baseline'), not yet a validated equivalent
                replacement for the inelastic ones.
            fit_Bp (bool): whether B_p (the backscattering-ratio /
                phase-function parameter robust's forward models take)
                is a free MCMC parameter (True) or held fixed at
                Bp_value (False, the default). Only meaningful for a
                robust backend; requires rt_backend != 'gordon'
                (validate_rt_dict raises otherwise). When True, B_p is
                sampled linearly (not log10) over
                [BP_PRIOR_PMIN, BP_PRIOR_PMAX] = [0.004, 0.05] (see
                BP_PRIOR_PMIN/BP_PRIOR_PMAX module constants).
            Bp_value (float): the fixed/seed value for B_p, used
                whenever fit_Bp is False (B_p held fixed) and as the
                walker-ball seed when fit_Bp is True. Default 0.01.
            include_CDOM_fl (bool): include robust's **CDOM
                fluorescence** term (the analytic Hawes et al. 1992
                kernel, ``robust.rt.cdom_fl``) as a third inelastic
                process alongside Raman and chlorophyll fluorescence.
                Default False. Robust backends only -- and not
                'robust_baseline', which is elastic-only
                (validate_rt_dict raises in both cases). The amplitude
                ``robust.rt.CDOMFl.scale`` is held **fixed at 1.0** (the
                published reference kernel); it is not a free parameter.
                The kernel's emission source term is the *CDOM*
                absorption spectrum, which BING supplies as
                ``cdom_fraction * a_dg`` -- see below and
                :func:`bing.evaluate.calc_Rrs_from_models_robust`.
            cdom_fraction (float): the fraction of the a-model's
                **combined** dissolved+detrital absorption a_dg treated
                as pure CDOM when include_CDOM_fl is True:
                ``a_cdom = cdom_fraction * a_dg``. Default
                CDOM_FRACTION_DEFAULT = 0.8. A **fixed-fraction proxy**,
                not a retrieved quantity -- BING's a_dg is CDOM +
                detritus combined while the Hawes kernel expects pure
                CDOM absorption (project decision, JXP 2026-09-05,
                ``claude_prompts/rt_tests.md`` Q32). Consulted only when
                include_CDOM_fl is True; carried (and ignored)
                otherwise.

    Returns:
        dict: rt_dict with keys 'variable_Gordon', 'variable_Gordon_G0',
            'variable_Gordon_bbp', 'include_Raman', 'include_Chl_fl',
            'phi_C', 'double_gaussian' (each taken verbatim from `p`, or
            None if `p` lacks the attribute), plus 'rt_backend',
            'fit_Bp', 'Bp_value', 'include_CDOM_fl', 'cdom_fraction'
            (each taken from `p` if present, else the Gordon-backend
            defaults 'gordon' / False / 0.01 / False / 0.8 documented
            above).
    """

    rt_dict = {}
    for key in ['variable_Gordon',
                'variable_Gordon_G0',
                'variable_Gordon_bbp',
                'include_Raman',
                'include_Chl_fl',
                'phi_C',
                'double_gaussian']:
        if hasattr(p,key):
            rt_dict[key] = getattr(p, key)
        else:
            rt_dict[key] = None

    # RT backend selection (rob_rt integration, M0). Unlike the keys above,
    # these get real defaults -- not None -- so a legacy `p` object (and any
    # rt_dict saved before this integration) yields a fully valid,
    # Gordon-backend dict with no code changes required elsewhere.
    rt_dict['rt_backend'] = getattr(p, 'rt_backend', 'gordon')
    rt_dict['fit_Bp'] = getattr(p, 'fit_Bp', False)
    rt_dict['Bp_value'] = getattr(p, 'Bp_value', 0.01)

    # CDOM fluorescence (rob_cdom). Same "real default, never None"
    # pattern as the three keys above, for the same reason: a legacy `p`
    # (or an rt_dict pickled before this branch) must still describe a
    # complete, valid configuration -- here, one with the process off.
    # NOTE: cdom_fraction is the a_cdom = cdom_fraction * a_dg proxy
    # factor -- see CDOM_FRACTION_DEFAULT for the full rationale.
    rt_dict['include_CDOM_fl'] = getattr(p, 'include_CDOM_fl', False)
    rt_dict['cdom_fraction'] = getattr(p, 'cdom_fraction',
                                       CDOM_FRACTION_DEFAULT)

    # Return
    return rt_dict


def validate_rt_dict(rt_dict, models=None, geom=None):
    """
    Validate an rt_dict at fit setup, before any forward-model calls.

    Intended to be called once per fit (e.g. by fit_one / chisq_fit.fit),
    never per forward-model call. Checks only what its arguments allow --
    passing `models=None` or `geom=None` simply skips the checks that need
    them, rather than raising, so this can also be used to probe a single
    check in isolation.

    Args:
        rt_dict (dict): RT configuration, as built by rt_dict_from_p.
        models (list, optional): [a_model, bb_model] for this fit.
            Consulted for the robust_hybrid wavelength-grid check (uses
            models[0].wave) and for the CDOM-fluorescence a_dg check
            (uses models[0].has_a_dg).
        geom (ObsGeometry, optional): fixed per-pixel geometry for this fit.
            Required (non-None) whenever rt_dict['rt_backend'] selects a
            robust backend -- theta_s is never silently defaulted
            (claude_prompts/rob_rt.md, Q&A/Coding item 4).

    Raises:
        ValueError: on any illegal configuration:
            (i) rt_backend not one of RT_BACKENDS;
            (ii) fit_Bp=True with rt_backend='gordon' (Gordon has no
                phase-function input);
            (iii) a robust backend with geom is None;
            (iv) rt_backend='robust_hybrid' with any model wavelength
                outside [ROBUST_HYBRID_WAVE_MIN, ROBUST_HYBRID_WAVE_MAX];
            (v) include_CDOM_fl=True with rt_backend='gordon' (BING's own
                Gordon path has no CDOM-fluorescence physics);
            (vi) include_CDOM_fl=True with rt_backend='robust_baseline'
                (elastic-only by construction, like the other two
                inelastic flags);
            (vii) include_CDOM_fl=True with an a-model that has no
                separable a_dg component (``models[0].has_a_dg`` False) --
                the CDOM source term is ``cdom_fraction * a_dg``, so there
                is nothing to build it from. Skipped when models is None.
    """
    rt_backend = rt_dict.get('rt_backend', 'gordon')
    if rt_backend not in RT_BACKENDS:
        raise ValueError(
            f"rt_dict['rt_backend']={rt_backend!r} is not one of {RT_BACKENDS}")

    if rt_dict.get('fit_Bp', False) and rt_backend == 'gordon':
        raise ValueError(
            "rt_dict['fit_Bp']=True requires a robust backend -- the "
            "Gordon backend has no phase-function input "
            "(rt_dict['rt_backend']='gordon')")

    if rt_backend != 'gordon' and geom is None:
        raise ValueError(
            f"rt_dict['rt_backend']={rt_backend!r} requires geometry -- "
            "pass an ObsGeometry via geom= (theta_s is never silently "
            "defaulted; claude_prompts/rob_rt.md Q&A/Coding item 4)")

    if rt_backend == 'robust_hybrid' and models is not None:
        wave = np.asarray(models[0].wave)
        if np.any(wave < ROBUST_HYBRID_WAVE_MIN) or np.any(wave > ROBUST_HYBRID_WAVE_MAX):
            raise ValueError(
                "rt_dict['rt_backend']='robust_hybrid' requires all "
                f"wavelengths within [{ROBUST_HYBRID_WAVE_MIN}, "
                f"{ROBUST_HYBRID_WAVE_MAX}] nm (the emulator's training "
                f"range); got range [{wave.min()}, {wave.max()}]")

    # --- CDOM fluorescence (rob_cdom) ---
    if rt_dict.get('include_CDOM_fl', False):
        if rt_backend == 'gordon':
            raise ValueError(
                "rt_dict['include_CDOM_fl']=True requires a robust "
                "backend -- CDOM fluorescence is robust.rt's Hawes (1992) "
                "kernel (robust.rt.cdom_fl), and BING's own Gordon path "
                "has no CDOM-fluorescence physics at all. Use "
                "rt_backend='robust_ztt' or 'robust_hybrid'")
        if rt_backend == 'robust_baseline':
            raise ValueError(
                "rt_dict['include_CDOM_fl']=True is illegal with "
                "rt_dict['rt_backend']='robust_baseline' -- "
                "robust.rt.baselines.Rrs_gordon takes no `inelastic` "
                "argument and is elastic-only by construction (the same "
                "rule as include_Raman/include_Chl_fl). Use "
                "rt_backend='robust_ztt' or 'robust_hybrid'")
        if models is not None and not getattr(models[0], 'has_a_dg', False):
            # The CDOM source term is a_cdom = cdom_fraction * a_dg (see
            # CDOM_FRACTION_DEFAULT), so an a-model with no separable
            # dissolved+detrital component simply cannot supply it.
            a_model = models[0]
            raise ValueError(
                "rt_dict['include_CDOM_fl']=True requires an absorption "
                "model with a separable a_dg (dissolved + detrital) "
                f"component, but {type(a_model).__name__} "
                f"(name={getattr(a_model, 'name', None)!r}) has none "
                "(has_a_dg is False) -- the CDOM-fluorescence source term "
                "is a_cdom = cdom_fraction * a_dg. Use an a-model with an "
                "explicit a_dg term (ExpBricaud/ExpBricaudFix/"
                "ExpBricaudFree, GIOP, GSM, ExpNMF), or disable "
                "include_CDOM_fl")