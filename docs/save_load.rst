.. _save_load:

==========================
Saving and Loading Fits
==========================

BING provides two helper functions in :mod:`bing.io` for persisting the
outputs of an MCMC fit and reading them back without having to re-run the
inference.  The design follows the project conventions:

* large numerical arrays are written with :func:`numpy.savez` to
  ``<outroot>.npz``;
* small inputs and simple stats are written as JSON to
  ``<outroot>.json``.

The two files share a basename so a fit is identified by a single
``outroot`` path (without extension).

File layout
-----------

``<outroot>.npz``
~~~~~~~~~~~~~~~~~

================== ========================================================
Key                Contents
================== ========================================================
``wave``           Wavelengths used for the fit (``shape (nwave,)``).
``Rrs``            Observed remote sensing reflectance.
``varRrs``         Variance of the observed Rrs.
``chains``         MCMC chains, shape ``(nsteps, nwalkers, nparam)``.
``p0``             Initial guess seeded into emcee (post-LM if any).
``p0_init``        *Optional.* Pre-LM seed, only present when supplied
                   to :func:`bing.io.save_fit`.
``a``, ``bb``      Median reconstructed total absorption and
                   backscattering from
                   :func:`bing.evaluate.reconstruct_from_chains`.
``a_lo`` / ``a_hi`` Lower/upper percentile bounds for ``a``.
``bb_lo`` / ``bb_hi`` Lower/upper percentile bounds for ``bb``.
``Rrs_recon``      Median model Rrs reconstructed from the chains.
``sigRrs_recon``   Standard deviation of model Rrs across the chains.
================== ========================================================

``<outroot>.json``
~~~~~~~~~~~~~~~~~~

=================== ======================================================
Key                 Contents
=================== ======================================================
``bing_version``    Value of ``bing.__version__`` at save time.
``model_names``     ``[absorption_model, backscattering_model]``.
``pnames``          Concatenated parameter names (absorption first).
``params``          Full parameter named-tuple, converted to a JSON-safe
                    dict.  All radiative-transfer flags live here, so
                    ``rt_dict`` itself is **not** persisted — it is
                    rebuilt via
                    :func:`bing.rt.defs.rt_dict_from_p` when needed.
``priors``          Flat list of prior dicts (absorption then
                    backscattering).
``stats``           Output of :func:`bing.evaluate.calc_stats`, with all
                    arrays converted to lists.
``stats_perc``      ``[lo, hi]`` percentiles used by ``calc_stats``.
``recon_perc``      ``[lo, hi]`` percentiles used by
                    ``reconstruct_from_chains``.
=================== ======================================================

Saving a fit
------------

After running a single-spectrum MCMC fit (for example with
:func:`bing.fitting.l23.fit_one`), call :func:`bing.io.save_fit`:

.. code-block:: python

    from bing import io as bing_io
    from bing.fitting import l23 as fit_l23
    from bing.parameters import standard

    p = standard.expb_pow(satellite='PACE', nsteps=40000, nburn=1000)
    chains, models, prep_dict, idx, extras = fit_l23.fit_one(p, idx=170)

    npz_path, json_path = bing_io.save_fit(
        'fits/L23_170', p, models, chains,
        p0=prep_dict['p0'],
        Rrs=prep_dict['model_Rrs'],
        varRrs=prep_dict['model_varRrs'],
    )

The ``rt_dict`` used during reconstruction is derived from the parameter
named-tuple via :func:`bing.rt.defs.rt_dict_from_p`, so callers do not
have to provide it.

If you have a pre-LM initial guess that you would like to keep around
for reproducibility, pass it via the optional ``p0_init`` argument:

.. code-block:: python

    bing_io.save_fit(outroot, p, models, chains,
                     p0=p0_post_lm, p0_init=p0_pre_lm,
                     Rrs=Rrs, varRrs=varRrs)

Loading a fit
-------------

:func:`bing.io.load_fit` returns a single dictionary with every saved
field plus three convenience entries built on the fly:

* ``models`` — re-instantiated absorption / backscattering models with
  their priors re-attached, ready for :meth:`eval_a` / :meth:`eval_bb`.
* ``rt_dict`` — regenerated from the saved parameters.
* ``p`` — the parameter named-tuple, rebuilt via
  :func:`bing.parameters.p_ntuple.gen`.

.. code-block:: python

    loaded = bing_io.load_fit('fits/L23_170')

    chains = loaded['chains']
    models = loaded['models']
    rt_dict = loaded['rt_dict']

    # Re-evaluate IOPs on the original wavelength grid
    a = models[0].eval_a(chains[..., :models[0].nparam])

Percentile choices
------------------

By default :func:`bing.io.save_fit` calls
:func:`bing.evaluate.calc_stats` with ``perc=(14, 86)`` and
:func:`bing.evaluate.reconstruct_from_chains` with ``perc=(5, 95)``.
Both choices are recorded in the JSON file under ``stats_perc`` and
``recon_perc`` so the meaning of the saved percentile arrays is never
ambiguous.

Pass ``stats_perc`` / ``recon_perc`` to override the defaults.
