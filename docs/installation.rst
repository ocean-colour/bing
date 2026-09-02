.. _installation:

============
Installation
============

This guide covers the installation of BING and its dependencies.

Requirements
------------

BING requires Python 3.12 or higher (matching the ``retrieve-or-bust``
dependency's floor) and depends on the following core packages:

* **NumPy** >= 1.20.0
* **SciPy** >= 1.7.0
* **pandas** >= 1.3.0
* **xarray** >= 0.19.0
* **matplotlib** >= 3.4.0
* **seaborn** >= 0.11.0
* **corner** >= 2.2.0
* **earthaccess** (for NASA Earthdata access)
* **ocpy** (Ocean Color Python library; see below — needs an editable
  install from source)
* **retrieve-or-bust** (the ``robust`` package, providing the
  ``robust.rt`` RT backends; installed automatically from GitHub — see
  below) together with the **JAX** stack (``jax``, ``flax``,
  ``jaxtyping``)

Installation Methods
--------------------

Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install BING is using pip:

.. code-block:: bash

    pip install bing-ocean

From Source
~~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

    git clone https://github.com/ocean-colour/bing.git
    cd bing
    pip install -e .

This installs BING in development mode, allowing you to modify the source code.

.. note::

   The install pulls ``retrieve-or-bust`` directly from GitHub (it is
   not published on PyPI) via a PEP 508 direct reference in
   ``setup.py``, currently pinned to that repository's ``cdom-rt``
   branch — so ``pip install .`` needs ``git`` on your ``PATH`` and
   network access to github.com. The JAX stack (``jax``, ``flax``,
   ``jaxtyping``) rides in with it; CPU-only JAX is sufficient. This
   is what powers the ``robust_*`` RT backends (see
   :ref:`rt-backends`) — and it is not optional: ``bing.evaluate``
   imports ``jax`` and ``robust`` unconditionally.

Using Conda
~~~~~~~~~~~

If you prefer using conda, first create a new environment:

.. code-block:: bash

    conda create -n bing python=3.12
    conda activate bing
    conda install numpy scipy pandas xarray matplotlib seaborn
    pip install bing-ocean

Dependencies Setup
------------------

OCPY Installation
~~~~~~~~~~~~~~~~~

BING depends on the Ocean Color Python (ocpy) library. It is not on
PyPI, and it must be installed **editable** from a source checkout —
its ``ocpy/hydrolight`` subpackage has no ``__init__.py``, so a regular
(non-editable) install silently omits ``ocpy.hydrolight``, which BING's
backscattering models need:

.. code-block:: bash

    git clone https://github.com/ocean-colour/ocpy.git
    pip install -e ocpy

(Add ``--no-deps`` if you want to skip ocpy's heavier dependencies —
everything BING itself needs is already covered by BING's own install.)

retrieve-or-bust (robust.rt) Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing to do in the normal case: ``pip install .`` installs
``retrieve-or-bust`` from GitHub automatically (see the note under
*From Source* above), and the installed copy is fully functional —
including the ``robust_hybrid`` emulator, whose trained weights ship
inside the package. For development on both packages side by side, an
editable checkout works too and takes precedence if installed after:

.. code-block:: bash

    git clone --branch cdom-rt https://github.com/ocean-colour/retrieve-or-bust.git
    pip install --no-deps -e retrieve-or-bust

correct_atmosphere Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The chlorophyll fluorescence module (:doc:`chlorophyll_fluorescence`)
requires the ``correct_atmosphere`` package to provide downwelling
irradiance spectra at the excitation wavelengths.

.. code-block:: bash

    pip install git+https://github.com/ocean-colour/correct-atmosphere.git

It is only needed if you enable ``include_Chl_fl=True`` in your parameter
configuration; the rest of BING runs without it.

NASA Earthdata Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To access PACE and other NASA satellite data, you need to configure Earthdata authentication:

1. Create an account at https://urs.earthdata.nasa.gov/
2. Create a `.netrc` file in your home directory:

.. code-block:: bash

    echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > ~/.netrc
    chmod 600 ~/.netrc

3. Or configure using earthaccess:

.. code-block:: python

    import earthaccess
    earthaccess.login(persist=True)

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Set the following environment variables for data paths:

.. code-block:: bash

    export OS_COLOR=/path/to/ocean/color/data
    export PACE_DATA=/path/to/pace/data
    export ARGO_DATA=/path/to/argo/data

Add these to your shell configuration file (`.bashrc`, `.zshrc`, etc.) for persistence.

Verification
------------

Verify your installation:

.. code-block:: python

    import bing
    print(bing.__version__)
    
    # Test core imports
    from bing import evaluate
    from bing.parameters import standard
    from bing.models import utils as model_utils
    from bing.fitting import chisq_fit, inference
    
    print("Installation successful!")

Optional Dependencies
---------------------

For advanced features, you may want to install:

.. code-block:: bash

    # For parallel processing
    pip install joblib
    
    # For advanced plotting
    pip install plotly
    
    # For geospatial operations
    pip install cartopy shapely
    
    # For development and testing
    pip install pytest pytest-cov ipython jupyter

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error for ocpy**
    Ensure ocpy is installed *editable* from a source checkout (see
    *OCPY Installation* above) — a non-editable install lacks
    ``ocpy.hydrolight``.

**Import Error for jax or robust**
    ``bing.evaluate`` imports both unconditionally. Reinstall with
    ``pip install .`` (which pulls ``retrieve-or-bust`` from GitHub and
    the JAX stack from PyPI), or follow *retrieve-or-bust
    Installation* above.

**NASA Earthdata Authentication Failed**
    Check your `.netrc` file permissions (should be 600) and credentials

**Missing Environment Variables**
    Verify that OS_COLOR and other paths are set correctly

**Memory Issues with Large Datasets**
    Consider using dask for out-of-core computations:
    
    .. code-block:: bash
    
        pip install dask[complete]

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/yourusername/bing/issues>`_
2. Consult the :doc:`/tutorials/index` for examples
3. Contact the development team

Next Steps
----------

After installation, proceed to:

* :doc:`getting_started` - Learn the basics
* :doc:`/tutorials/index` - Work through tutorials
* :doc:`/api/index` - Explore the API reference
