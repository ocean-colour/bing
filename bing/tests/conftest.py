"""Shared pytest configuration for the BING test suite.

Forces matplotlib onto the non-interactive ``Agg`` backend before any
test module imports ``pyplot``.  Several tests exercise the plotting
routines (e.g. ``bing.plotting.show_fits`` with ``show=True``), which
would otherwise pop up figure windows / block on a display during a test
run.  ``Agg`` renders to a buffer instead, so ``plt.show()`` becomes a
harmless no-op and the suite runs headless (and on CI).
"""

import matplotlib

# Select Agg up front; force=True overrides any backend a transitively
# imported module may have already requested.
matplotlib.use("Agg", force=True)
