.. _fitting_api:

===========
Fitting API
===========

Parameter estimation: least-squares for a fast point estimate, MCMC for
the full posterior. See :doc:`../fitting` for the workflow in prose.

Least squares
-------------

.. automodule:: bing.fitting.chisq_fit
   :members:

MCMC inference
--------------

.. automodule:: bing.fitting.inference
   :members:

.. note::

   ``bing.fitting.l23``, the driver for the Loisel et al. (2023)
   synthetic dataset, is not auto-documented here: it imports
   ``correct_atmosphere`` (for the downwelling irradiance used by the
   chlorophyll-fluorescence term), which is not installed in the
   documentation build. See :doc:`../fitting` for its usage.
