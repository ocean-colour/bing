"""I/O helpers for BING fits.

These helpers save the outputs of a BING MCMC fit to disk (one ``.npz``
for large numerical arrays and one ``.json`` for small inputs / simple
stats) and load them back into a Python dictionary, re-instantiating
the model objects so the caller can immediately reconstruct IOPs.

The layout intentionally separates "big" from "small":

* ``<outroot>.npz`` — wavelengths, observed Rrs, variance, MCMC chains,
  the initial guess(es), and the per-wavelength reconstruction from
  :func:`bing.evaluate.reconstruct_from_chains`.
* ``<outroot>.json`` — BING version, model names, parameter named-tuple
  contents, priors, the small output of
  :func:`bing.evaluate.calc_stats`, and the percentile choices used.

Following the project guideline, all functionality is exposed as plain
functions (no new classes).
"""

import json
import os

import numpy as np

from bing import __version__ as bing_version
from bing import evaluate as bing_eval
from bing.models import utils as model_utils
from bing.parameters import p_ntuple
from bing.priors import priors as bing_priors
from bing.rt import defs as rt_defs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_jsonable(value):
    """Recursively convert numpy / tuple / namedtuple values to JSON-safe ones.

    Plain Python scalars, strings and ``None`` are returned unchanged.
    This is a generic helper kept here because it is primarily used to
    serialise the parameter named-tuple, but it is safe to use on any
    nested structure (e.g. the output of ``bing.evaluate.calc_stats``).
    """
    # numpy scalar → Python scalar
    if isinstance(value, np.generic):
        return value.item()
    # numpy array → list (preserving nested structure)
    if isinstance(value, np.ndarray):
        return value.tolist()
    # namedtuple → dict via _asdict
    if hasattr(value, "_asdict") and isinstance(value, tuple):
        return {k: to_jsonable(v) for k, v in value._asdict().items()}
    # tuple / list → list (recurse)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    # dict → dict (recurse)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def params_to_dict(p):
    """Convert a BING parameter named-tuple into a JSON-safe dictionary.

    The parameter named-tuple is produced by :func:`gen` and contains a
    mix of strings, floats, booleans, ``None`` and lists of dicts
    (priors).  All numpy objects and nested tuples are converted so
    :func:`json.dump` can handle the result.
    """
    return to_jsonable(p)

def save_fit(outroot, p, models, chains, p0, Rrs, varRrs,
             p0_init=None, stats_perc=(14, 86), recon_perc=(5, 95)):
    """Save a BING MCMC fit to ``<outroot>.npz`` and ``<outroot>.json``.

    Parameters
    ----------
    outroot : str
        Path *without* extension.  Two files are written: ``outroot + '.npz'``
        and ``outroot + '.json'``.
    p : namedtuple
        Parameter named-tuple produced by ``bing.parameters.standard.*``.
    models : list
        Two-element ``[absorption_model, backscattering_model]`` list, with
        priors already attached.
    chains : np.ndarray
        MCMC chains of shape ``(nsteps, nwalkers, nparam)``.
    p0 : np.ndarray
        Initial guess seeded into emcee (post-LM, if a least-squares pass
        was run).
    Rrs, varRrs : np.ndarray
        Observed Rrs and its variance at ``models[0].wave``.
    p0_init : np.ndarray, optional
        Pre-LM seed.  Only written to the ``.npz`` when supplied.
    stats_perc : tuple of int, optional
        Percentiles passed to :func:`bing.evaluate.calc_stats`.
    recon_perc : tuple of int, optional
        Percentiles passed to
        :func:`bing.evaluate.reconstruct_from_chains`.

    Returns
    -------
    npz_path, json_path : tuple of str
        The two files that were written.
    """
    # ---- Sanity / setup ------------------------------------------------
    # Drop any trailing extension the caller may have included by accident.
    base, ext = os.path.splitext(outroot)
    if ext in (".npz", ".json"):
        outroot = base
    npz_path = outroot + ".npz"
    json_path = outroot + ".json"

    # Reconstruct rt_dict from the parameter tuple — it does not need to
    # be passed in by the caller.
    rt_dict = rt_defs.rt_dict_from_p(p)

    # Concatenated parameter names (absorption first, then backscattering).
    pnames = list(models[0].pnames) + list(models[1].pnames)

    # ---- Per-wavelength reconstruction --------------------------------
    a, bb, a_lo, a_hi, bb_lo, bb_hi, Rrs_recon, sigRrs_recon = \
        bing_eval.reconstruct_from_chains(models, chains, rt_dict,
                                          perc=recon_perc)

    # ---- Simple stats from the chains ---------------------------------
    stats = bing_eval.calc_stats(chains, names=pnames, perc=stats_perc)

    # ---- Write the .npz file ------------------------------------------
    npz_payload = dict(
        wave=np.asarray(models[0].wave),
        Rrs=np.asarray(Rrs),
        varRrs=np.asarray(varRrs),
        chains=np.asarray(chains),
        p0=np.asarray(p0),
        a=a, bb=bb,
        a_lo=a_lo, a_hi=a_hi,
        bb_lo=bb_lo, bb_hi=bb_hi,
        Rrs_recon=Rrs_recon,
        sigRrs_recon=sigRrs_recon,
    )
    if p0_init is not None:
        npz_payload["p0_init"] = np.asarray(p0_init)
    np.savez(npz_path, **npz_payload)

    # ---- Write the .json file -----------------------------------------
    json_payload = dict(
        bing_version=bing_version,
        model_names=list(p.model_names),
        pnames=pnames,
        params=params_to_dict(p),
        #priors=bing_priors.priors_from_models(models),
        stats=to_jsonable(stats),
        stats_perc=list(stats_perc),
        recon_perc=list(recon_perc),
    )
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    return npz_path, json_path


def load_fit(outroot):
    """Load a BING fit saved by :func:`save_fit`.

    Parameters
    ----------
    outroot : str
        Path to the fit, with or without extension.  ``outroot + '.npz'``
        and ``outroot + '.json'`` are both read.

    Returns
    -------
    dict
        Dictionary with every saved field plus the convenience entries
        ``'models'``, ``'rt_dict'`` and ``'p'`` (rebuilt named-tuple).
    """
    # ---- Resolve paths -------------------------------------------------
    base, ext = os.path.splitext(outroot)
    if ext in (".npz", ".json"):
        outroot = base
    npz_path = outroot + ".npz"
    json_path = outroot + ".json"

    # ---- Load the JSON metadata first ---------------------------------
    with open(json_path, "r") as f:
        meta = json.load(f)

    # Rebuild the parameter named-tuple so callers can drive other BING
    # functions that expect a namedtuple (e.g. rt_dict_from_p).
    params_dict = dict(meta["params"])
    p = p_ntuple.gen(**params_dict)

    # ---- Load the .npz payload ----------------------------------------
    with np.load(npz_path, allow_pickle=True) as data:
        npz_payload = {key: np.asarray(data[key]) for key in data.files}
    wave_arr = npz_payload["wave"]

    # Build a bare model pair to learn nparam for each, then rebuild with
    # the saved priors split into ``apriors`` / ``bpriors``.
    bare_models = model_utils.init(meta["model_names"], wave_arr)
    #a_pdicts, b_pdicts = bing_priors.split_priors(meta["priors"], bare_models)
    a_pdicts = p.apriors
    b_pdicts = p.bpriors
    models = model_utils.init(meta["model_names"], wave_arr,
                              prior_dicts=(a_pdicts, b_pdicts))

    # Rebuild rt_dict from the parameter tuple — not stored on disk.
    rt_dict = rt_defs.rt_dict_from_p(p)

    # ---- Assemble the result ------------------------------------------
    result = dict(npz_payload)
    result.update(meta)
    result["models"] = models
    result["rt_dict"] = rt_dict
    result["p"] = p
    return result
