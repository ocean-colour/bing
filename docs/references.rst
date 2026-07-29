.. _references:

==========
References
==========

The papers BING implements, and where each is used. Topic-specific
bibliographies live with their pages: :doc:`models` for the bio-optical
models and the turbid-water backscattering literature, :doc:`raman` for
Raman scattering, and :doc:`chlorophyll_fluorescence` for fluorescence.

Reflectance model
-----------------

* Gordon, H. R., Brown, O. B., Evans, R. H., Brown, J. W., Smith, R. C.,
  Baker, K. S., & Clark, D. K. (1988). A semianalytic radiance model of
  ocean color. *J. Geophys. Res.* 93, 10909-10924.
  doi:10.1029/JD093iD09p10909 -- the ``Rrs`` :math:`\leftrightarrow`
  :math:`b_b/(a+b_b)` relation that gives BING its name.
* Lee, Z., Carder, K. L., & Arnone, R. A. (2002). Deriving inherent
  optical properties from water color: a multiband quasi-analytical
  algorithm for optically deep waters. *Appl. Opt.* 41, 5755-5772.
  doi:10.1364/AO.41.005755 -- the QAA, and the dynamic backscattering
  slope used by the ``Lee`` model.

Absorption
----------

* Bricaud, A., Babin, M., Morel, A., & Claustre, H. (1995). Variability
  in the chlorophyll-specific absorption coefficients of natural
  phytoplankton. *J. Geophys. Res.* 100, 13321-13332.
  doi:10.1029/95JC00463 -- the ``a_ph`` shape used by the ``Bricaud``
  and ``ExpBricaud`` families.
* Pope, R. M., & Fry, E. S. (1997). Absorption spectrum (380-700 nm) of
  pure water. II. Integrating cavity measurements. *Appl. Opt.* 36,
  8710-8723. doi:10.1364/AO.36.008710 -- pure-water absorption.
* Chase, A. P. et al. (2017). Decomposition of in situ particulate
  absorption spectra. *Methods Oceanogr.* 22, 100024.
  doi:10.1016/j.mio.2017.100024 -- the multi-Gaussian ``Chase2017``
  model.

Semi-analytical algorithms
--------------------------

* Maritorena, S., Siegel, D. A., & Peterson, A. R. (2002). Optimization
  of a semianalytical ocean color model for global-scale applications.
  *Appl. Opt.* 41, 2705-2714. doi:10.1364/AO.41.002705 -- GSM.
* Werdell, P. J. et al. (2013). Generalized ocean color inversion model
  for retrieving marine inherent optical properties. *Appl. Opt.* 52,
  2019-2037. doi:10.1364/AO.52.002019 -- GIOP.

Datasets
--------

* Loisel, H., Stramski, D., Dessailly, D., Jamet, C., Li, L., &
  Reynolds, R. A. (2023). Hydrolight radiative-transfer simulations.
  Dryad, doi:10.6076/D1630T -- the synthetic dataset with known IOPs
  used for validation, and the source of the pure-water backscattering
  spectrum every model loads at construction.
* IOCCG Report No. 5 (2006). *Remote Sensing of Inherent Optical
  Properties: Fundamentals, Tests of Algorithms, and Applications* --
  reference water IOPs.

Method
------

* Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013).
  emcee: The MCMC Hammer. *PASP* 125, 306. doi:10.1086/670067 -- the
  affine-invariant sampler behind :mod:`bing.fitting.inference`.
