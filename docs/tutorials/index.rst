.. _tutorials:

=========
Tutorials
=========

Step-by-step material for common BING workflows. The narrative guides are
the pages listed below; the runnable, output-carrying material lives in
the repository's notebooks and scripts, which is where to look for a
worked end-to-end example.

Where to start
--------------

1. :doc:`../getting_started` -- installation, a first fit, and the shape
   of the API.
2. :doc:`../models` -- the absorption and backscattering models, their
   parameters and priors, and which suit open-ocean versus turbid water.
3. :doc:`../parameters` -- the standard model combinations and the
   configuration tuple that carries a complete fit setup.
4. :doc:`../fitting` -- least squares for a fast point estimate, MCMC for
   the posterior, and the settings that matter (evaluation budget,
   walker initialisation).
5. :doc:`../save_load` -- persisting a fit and reading it back.
6. :doc:`../data_processing` -- getting observations onto a satellite
   band set with a matching noise model.

Radiative transfer, in increasing depth: :doc:`../radiative_transfer`,
then :doc:`../raman` and :doc:`../chlorophyll_fluorescence` for the
inelastic terms.

Runnable examples
-----------------

:doc:`../examples` maps the notebooks under ``nb/`` and the scripts under
``dev/``. Two make good starting points because they are executed with
their figures embedded, so they can be read without running anything:

* ``nb/TurbidWaters/turbid_bbnw_models.ipynb`` -- fitting one spectrum
  with four backscattering models and comparing what each can represent.
* ``nb/TurbidWaters/walker_init_fix.ipynb`` -- diagnosing an MCMC chain,
  including a failure mode that leaves the acceptance fraction looking
  perfectly healthy.

To run them:

.. code-block:: bash

    cd nb/TurbidWaters
    jupyter lab turbid_bbnw_models.ipynb

Data
----

The Loisel et al. (2023) synthetic dataset (Dryad,
doi:10.6076/D1630T) provides spectra with known IOPs and is the basis of
most examples; it is **not** bundled with BING. Point ``$OS_COLOR`` at
the directory holding it. PACE granules come from `NASA Earthdata
<https://earthdata.nasa.gov/>`_ and Argo BGC profiles from `Argo Data
Management <https://argo.ucsd.edu/>`_.

Still to be written
-------------------

Prose tutorials for PACE processing, Argo matching, model comparison and
uncertainty analysis do not exist yet, though the underlying code does --
see ``papers/`` for the analyses and ``dev/`` for focused
investigations. If you write one, add it to this directory and list it in
a toctree here; :doc:`../contributing` has the conventions.
