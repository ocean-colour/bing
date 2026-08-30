
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


def rt_dict_from_p(p):
    """
    Prepare data and models for L23 fitting.
    This function initializes the necessary data, models, priors, and MCMC
    parameters for fitting L23 data. It also handles wavelength conversions,
    noise scaling, and initial guesses for the fitting process.

    Args:
        p (object): Parameter object containing configuration
            radiative transfer options
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
        models (list, optional): [a_model, bb_model] for this fit. Only
            consulted for the robust_hybrid wavelength-grid check (uses
            models[0].wave).
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
                outside [ROBUST_HYBRID_WAVE_MIN, ROBUST_HYBRID_WAVE_MAX].
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