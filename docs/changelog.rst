.. _changelog:

=========
Changelog
=========

BING has no released versions yet (``bing.__version__`` reports a
development string), so this page records notable changes by theme rather
than by release. ``git log`` remains the authoritative history.

Unreleased
----------

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

* GitHub Actions workflow running the test suite on Python 3.11-3.13 and
  building the documentation.
* ``bing/tests/conftest.py`` reports a missing reference dataset as a
  skip, so the suite is usable without the data tree.
