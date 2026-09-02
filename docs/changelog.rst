.. _changelog:

=========
Changelog
=========

BING has no released versions yet (``bing.__version__`` reports a
development string), so this page records notable changes by theme rather
than by release. ``git log`` remains the authoritative history.

Unreleased
----------

robust.rt backend integration (2026-08/09, PR #27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BING's forward Rrs model is now pluggable: ``rt_dict['rt_backend']``
selects between BING's own Gordon (1988) path and the sibling
`retrieve-or-bust <https://github.com/ocean-colour/retrieve-or-bust>`_
package's ``robust.rt`` forward models. See :ref:`rt-backends` for the
full documentation. **Non-breaking**: ``rt_backend`` defaults to
``'gordon'``, which is byte-identical to pre-integration BING
(regression-pinned by the test suite); legacy 4-tuple observations and
saved rt dicts keep working unchanged.

* **Four backends** — ``'gordon'`` (the legacy default),
  ``'robust_ztt'`` (robust's analytic model), ``'robust_hybrid'``
  (ztt + a learned emulator correction, valid on 350–750 nm only) and
  ``'robust_baseline'`` (robust's Gordon-compatible refit, elastic
  only) — all fit end-to-end through ``chisq_fit.fit``,
  ``inference.fit_one``/``fit_batch`` and the ``l23`` wrappers.
* **Geometry threading.** The observation tuple gains an optional
  fifth element, a ``bing.rt.geometry.ObsGeometry``; robust-backend
  fits without one raise at setup (``theta_s`` is never silently
  defaulted). The ``l23`` wrappers build it from ``p`` or the L23
  dataset's own documented geometry (``l23.geom_from_p``).
* **Free or fixed** :math:`B_p`: ``rt_dict['fit_Bp']`` adds robust's
  phase-function parameter as an extra free parameter (linear uniform
  prior over [0.004, 0.05]) or holds it at ``rt_dict['Bp_value']``.
* **Accuracy, measured on real L23 data**: ``robust_baseline`` matches
  ``gordon`` elastically to ~2.3e-7 relative;
  ``robust_ztt``/``robust_hybrid`` are deliberately different, more
  physically complete elastic RT (a few percent higher Rrs). robust's
  *inelastic* terms diverge from BING's own Gordon+Raman/fluorescence
  physics by several percent (Raman max ~11 %, fluorescence max
  ~9–18 %) — a known, documented composition difference, not yet
  closed. The project has adopted the robust path for new work; the
  ``gordon`` backend and ``bing.rt.raman``/``chl_fl`` are kept for
  backward compatibility.
* New evaluation entry points: ``calc_Rrs_from_models_robust``,
  ``calc_Rrs_from_iops_robust``, ``robust_domain_check``;
  ``reconstruct_from_chains``/``reconstruct_chisq_fits`` dispatch on
  the backend and accept ``geom``.
* **Packaging**: ``bing`` now depends on ``retrieve-or-bust``
  (installed from GitHub via a PEP 508 direct reference in
  ``setup.py`` — it is not on PyPI) plus ``jax``/``flax``/
  ``jaxtyping``, and requires Python >= 3.12 to match
  retrieve-or-bust's floor. A bare ``'retrieve-or-bust'`` name briefly
  in ``install_requires`` broke any plain ``pip install .`` (including
  the ReadTheDocs build) with "No matching distribution found"; fixed
  by the direct reference.

Inelastic RT fixes (2026-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two errors were identified by validating BING against the paired
inelastic scenarios of the Loisel et al. (2023) HydroLight database
(see the retrieve-or-bust report ``context/RT/rt_inelastic_bing_summary.md``)
and fixed. **Both change numerical results.**

* **Fluorescence normalization (breaking, ~×3).**
  ``rt.rrs.calc_Rrs_fluorescence`` treated the two-flow *irradiance*
  reflectance :math:`R^F = E_u/E_d` as if it were a remote-sensing
  reflectance. For isotropic emission :math:`L_u = E_u/\pi`, so the
  term now applies :math:`r_{rs}^F = R^F/\pi` before the
  :math:`A\,r_{rs}/(1-B\,r_{rs})` conversion. Validated against
  L23 X4−X2: median model/truth at 685 nm is now 1.01/0.96/0.87
  (zenith 0°/30°/60°) versus 3.18/3.00/2.73 before. Fits run with
  ``include_Chl_fl`` before this fix overestimated
  :math:`R_{rs}^{fl}` by ~3× (equivalently, their effective quantum
  yield was ~π× smaller than nominal). The unused additive path
  ``rt.raman.calc_Rrs_with_raman`` had the same flaw and received the
  same conversion; the production multiplicative Raman path was never
  affected (it uses only a reflectance ratio, which cancels the
  normalization).
* **Raman correction now uses the true solar spectrum.** The
  production path assumed a flat spectrum,
  :math:`E_d(\lambda')/E_d(\lambda) = 1`, which distorts the spectral
  shape of the correction (validated against L23 X2/X1: ~+60 %
  increment error at 490 nm, −15 % and worse in the red). New
  ``aNWModel.set_raman_Ed(wave_Ed, Ed)`` stores the true ratio and
  ``evaluate.calc_Rrs_from_models`` uses it automatically;
  ``include_Raman`` runs without it fall back to flat-Ed **with a
  RuntimeWarning**. The L23 fitting pipeline sets it from
  ``correct_atmosphere``. Note the Ed grid must extend ~50 nm blueward
  of the model grid to cover the Raman excitation wavelengths.
* ``init_Chl_fluorescence`` now defaults ``Ed_em`` to the full Ed
  vector on the model grid (exact per-wavelength normalization of the
  fluorescence term); a legacy scalar is still accepted.
* New L23-anchored regression tests
  (``bing/tests/test_l23_inelastic.py`` with a committed 40-scene
  fixture) pin the fluorescence term to ±15 % of HydroLight truth at
  685 nm and the true-Ed Raman correction to ±15 % median increment
  error over 550–700 nm.

Turbid-water backscattering
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New backscattering models for turbid, mineral-dominated water:
  ``Pow2`` (a mineral term pivoted at 700 nm plus an organic term at
  600 nm, 4 parameters) and ``Pow2Flat`` (the same with the mineral
  exponent fixed at zero, 3 parameters). See :doc:`models`.
* New standard combinations ``expb_pow2``, ``expb_pow2flat`` and
  ``expb_powflex`` -- the last being the ordinary power law with its
  slope prior widened to allow a flat or rising spectrum, useful as a
  control.
* ``bbNWModel`` now dispatches evaluation polymorphically: each model
  implements ``_eval_bbnw(params, wave)`` and the base class resolves the
  wavelength grid, replacing a string comparison on the model name.
* Models declare ``log_params``, and the plotting code honours it, so
  figures no longer exponentiate linear parameters such as ``Sdg`` or
  ``beta``.

Fitting
~~~~~~~

* ``chisq_fit.fit`` accepts ``maxfev``, the optimizer's evaluation
  budget. Parameter-rich models need it: at scipy's default budget the
  two-component models fail to converge on a substantial fraction of
  spectra.
* MCMC walker initialisation is additive with an absolute floor and is
  clipped into the prior bounds. Previously the perturbation was purely
  multiplicative, so a parameter seeded at exactly zero received no
  spread at all and never moved for the whole run -- silently, with a
  healthy acceptance fraction.
* ``bbNWModel.check_priors`` rejects a prior list whose length does not
  match the parameter count, at construction and when
  ``set_standard_priors`` attaches them.

Corrections
~~~~~~~~~~~

* ``Lee``, ``GSM`` and ``Every`` now honour the wavelength grid passed to
  ``eval_bbnw``. They previously evaluated on the model grid regardless,
  so ``eval_bb_ex`` combined pure-water backscattering on the Raman
  *excitation* grid with particle backscattering on the *emission* grid.
  The error in ``bb_ex`` was 16-22 per cent, though its effect on fitted
  parameters was ~1e-4 dex.

Infrastructure
~~~~~~~~~~~~~~

* GitHub Actions workflow running the test suite on Python 3.12-3.14 and
  building the documentation.
* ``bing/tests/conftest.py`` reports a missing reference dataset as a
  skip, so the suite is usable without the data tree.
