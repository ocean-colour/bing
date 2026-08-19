""" Shared pytest configuration for the BING test suite.

Why this file exists
--------------------
Most of BING needs the Loisel et al. (2023) Hydrolight dataset. It is not
an optional extra: ``bbNWModel.init_bbw`` loads ``Hydrolight400.nc`` to
get pure-water backscattering, so merely *constructing* a backscattering
model requires the file. ocpy resolves its location from ``$OS_COLOR``.

Those files live outside the repository (~17 MB each, from
https://datadryad.org/stash/dataset/doi:10.6076/D1630T) and are not
available on CI, where roughly 60% of the suite would otherwise fail with
``FileNotFoundError``. Rather than decorate a hundred tests with a marker
that future tests would forget, this module converts that one specific
failure into a **skip**, wherever it happens -- in a fixture or in the
test body. Everything else is left alone, so a real error is still a real
error.

Consequences worth knowing:

* ``pytest -q`` is meaningful with or without the data tree. Without it
  you get skips, not failures, and ``-ra`` prints why.
* Test modules whose *imports* are unavailable (``bing.fitting.l23``
  needs the ``correct_atmosphere`` package) are dropped from collection
  instead, since an import error cannot be turned into a skip.
* ``needs_l23`` is exported for tests that would rather skip explicitly.

Requires pytest >= 8 for the ``wrapper=True`` hook style.
"""
import os

import matplotlib
import pytest

# Force a non-interactive backend for the whole suite. Several tests
# exercise bing.plotting functions that end in ``plt.show()``; under an
# interactive backend (e.g. TkAgg with a reachable DISPLAY) that call
# blocks forever waiting for a human to close the window, which hangs
# headless/automated runs. Under Agg, ``plt.show()`` is a no-op, so the
# plotting code path is still fully exercised. Must run before any test
# module imports pyplot.
matplotlib.use('Agg', force=True)

# Any dataset file whose absence should be reported as a skip. The
# exception message from h5netcdf/xarray carries the filename.
DATA_FILE_HINTS = ('Hydrolight',)


def l23_available():
    """Whether the Loisel+2023 dataset bing needs is on disk.

    Returns
    -------
    bool
        True when ocpy's L23 directory holds Hydrolight400.nc, the file
        ``bbNWModel.init_bbw`` reads.
    """
    try:
        from ocpy.hydrolight import loisel23
    except Exception:
        return False
    return os.path.isfile(os.path.join(loisel23.l23_path,
                                       'Hydrolight400.nc'))


# For tests that prefer to declare the dependency themselves
needs_l23 = pytest.mark.skipif(
    not l23_available(),
    reason='Loisel+2023 Hydrolight data not available ($OS_COLOR)')


def _module_importable(name):
    """Whether ``name`` can be imported, without raising."""
    try:
        __import__(name)
    except Exception:
        return False
    return True


# Modules that cannot even be imported without an optional dependency.
# An ImportError during collection cannot be converted to a skip, so drop
# them instead and say so.
collect_ignore = []
if not _module_importable('correct_atmosphere'):
    collect_ignore += ['test_evaluate.py', 'test_io.py',
                       'test_l23_fitting.py']


def _missing_data_reason(exc):
    """The skip reason for a missing-dataset error, else None.

    Parameters
    ----------
    exc : BaseException
        The exception raised by a test or its fixtures.

    Returns
    -------
    str or None
        A human-readable reason when this is the known
        "reference dataset is absent" failure, otherwise None.
    """
    if not isinstance(exc, OSError):      # FileNotFoundError included
        return None
    text = str(exc)
    for hint in DATA_FILE_HINTS:
        if hint in text:
            return (f'reference dataset not available ({hint}*.nc); '
                    'set $OS_COLOR to run this test')
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a missing reference dataset as a skip rather than a failure.

    Deliberately narrow: only ``OSError`` whose message names one of
    DATA_FILE_HINTS is converted, and only when the phase actually
    failed.
    """
    report = yield

    if report.failed and call.excinfo is not None:
        reason = _missing_data_reason(call.excinfo.value)
        if reason is not None:
            report.outcome = 'skipped'
            report.longrepr = (str(item.path), None, f'Skipped: {reason}')

    return report
