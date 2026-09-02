.. _evaluation_api:

==============
Evaluation API
==============

Post-fitting analysis: reconstructing IOPs and Rrs from chains or
least-squares results, and the information criteria used to compare
models.

This module also hosts the forward-model dispatch for the RT backends
(see :ref:`rt-backends`): ``calc_Rrs_from_models`` (the Gordon path),
``calc_Rrs_from_models_robust`` and ``calc_Rrs_from_iops_robust`` (the
``robust.rt`` paths, from model parameters or raw IOP spectra
respectively), ``robust_domain_check``, and the backend-aware
``reconstruct_from_chains`` / ``reconstruct_chisq_fits`` — all
documented below.

Evaluation
----------

.. automodule:: bing.evaluate
   :members:

Statistics
----------

.. automodule:: bing.stats
   :members:
