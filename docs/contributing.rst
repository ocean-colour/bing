.. _contributing:

============
Contributing
============

Environment
-----------

Development uses the ``ocean14`` conda environment. BING depends on two
sibling packages from the same organisation, ``ocpy`` (data readers,
water IOPs, satellite bands) and, for the chlorophyll-fluorescence
excitation term, ``correct_atmosphere``. Both are installed from their
checkouts:

.. code-block:: bash

    pip install -e .            # bing itself
    pip install -e ../ocpy
    pip install -e ../correct-atmosphere

Reference datasets sit outside the repository and are located through the
``$OS_COLOR`` environment variable. The Loisel et al. (2023) Hydrolight
files are the ones the test suite needs most (see below).

Running the tests
-----------------

.. code-block:: bash

    pytest bing/tests -q            # everything
    pytest bing/tests/test_bbnw.py  # one module
    pytest bing/tests -q -ra        # show why tests skipped

Most of the suite needs the L23 data, because constructing *any*
backscattering model reads ``Hydrolight400.nc`` for pure-water
backscattering. Without ``$OS_COLOR`` those tests **skip** rather than
fail -- ``bing/tests/conftest.py`` converts that specific missing-file
error into a skip, so ``pytest -q`` is meaningful either way. Roughly 110
of 180 tests skip in that state.

Continuous integration
----------------------

``.github/workflows/tests.yml`` runs the suite on Python 3.12-3.14 and
builds these docs. CI has no data tree, so it exercises the
data-independent tests; the reasons for each skip are printed in the log.

Conventions
-----------

* Amplitude parameters are stored and fitted in **log10**; spectral
  slopes and exponents stay linear. Models declare which is which in
  ``log_params`` -- see :ref:`log-params`.
* Model evaluation returns ``(nsample, nwave)``, for a single parameter
  vector as well as for MCMC chains.
* Docstrings carry the inputs and outputs of every method; the API pages
  are generated from them.
* Lines under 80 characters, PEP 8, and reuse the existing spectral
  helpers in ``bing.models.functions`` rather than re-deriving them.

Adding a model
--------------

Absorption and backscattering models follow a fixed contract: class
attributes (``name``, ``nparam``, ``pnames``, ``log_params``), a
``_eval_bbnw``/``eval_anw`` implementation that honours the wavelength
grid it is given, an entry in the module's ``init_model`` factory, and a
combination in ``bing.parameters.standard`` that passes its priors
explicitly. ``CLAUDE.md`` in the repository root carries the annotated
template, including the contracts that are easy to violate silently.
