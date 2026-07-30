.. _examples:

========
Examples
========

Worked examples live in the repository rather than in these pages, so that
they stay runnable. This page is a map of what is where.

Notebooks
---------

``nb/`` holds topic notebooks, executed with their figures embedded:

``nb/TurbidWaters/turbid_bbnw_models.ipynb``
    The two-component turbid backscattering models: the shapes they can
    make, a controlled experiment showing that two components are needed
    (and that widening a single power law's slope prior is not enough),
    and a no-regression check on a clear Loisel+2023 spectrum.

``nb/TurbidWaters/walker_init_fix.ipynb``
    Why an MCMC walker ball needs a floor: a parameter seeded at exactly
    zero used to stay frozen for an entire run, with a healthy-looking
    acceptance fraction.

``nb/Raman/``, ``nb/ChlFl/``
    Inelastic scattering: Raman and chlorophyll fluorescence.

``nb/MODIS/``, ``nb/Priors/``, ``nb/EMA/``
    Satellite band handling, prior construction, and exploratory work.

Development and benchmark scripts
---------------------------------

``dev/`` holds scripts that answer a specific question and write their
figures next to themselves:

``dev/turbid_bbp/turbid_bbp.py``
    Benchmarks the turbid backscattering models: synthetic recovery under
    noise, an identifiability comparison across four fitting strategies,
    and a no-regression test on open-ocean spectra.

``dev/ChlFl/``, ``dev/Gordon/``
    Fluorescence line-shape investigations and the derivation of the
    wavelength-dependent Gordon coefficients.

Analyses behind papers
----------------------

``papers/<topic>/`` holds the analysis code and figure scripts for
specific studies (``biomass``, ``bing_2.0``, ``phytoplankton``). Reusable
code belongs in ``bing/`` rather than there.

Running a fit from the command line
-----------------------------------

.. code-block:: bash

    bing_fit_Rrs input_table.csv Exp,Pow --outroot output \
        --satellite PACE --fit_method mcmc

The input CSV needs ``wave,Rrs,sigRrs`` columns, optionally with ``anw``
and ``bbnw`` truth for comparison. See :doc:`fitting` for the Python API.
