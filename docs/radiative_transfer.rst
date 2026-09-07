.. _radiative_transfer:

====================
Radiative Transfer
====================

BING implements radiative transfer models for calculating remote sensing reflectance (Rrs) from inherent optical properties (IOPs). This module provides both standard and advanced formulations of the Gordon semi-analytical model, and — since the integration with the sibling ``retrieve-or-bust`` (``robust``) package — a pluggable **RT backend** that can replace the Gordon forward model with ``robust.rt``'s more physically complete radiative transfer (see :ref:`rt-backends`).

Overview
--------

The radiative transfer module (``bing.rt``) converts optical properties into observable remote sensing reflectance:

.. math::

    R_{rs}(\lambda) = f(\lambda) \cdot \frac{b_b(\lambda)}{a(\lambda) + b_b(\lambda)}

where the function :math:`f(\lambda)` captures the geometry of light propagation through the water column and depends on solar zenith angle, viewing geometry, and inherent optical properties.

Gordon Semi-Analytical Model
-----------------------------

The Gordon (1988) model provides a semi-analytical relationship between subsurface remote sensing reflectance and IOPs:

.. math::

    r_{rs}(\lambda) = G_1(\lambda) \cdot u(\lambda) + G_2(\lambda) \cdot u^2(\lambda)

where:
    - :math:`u(\lambda) = \frac{b_b(\lambda)}{a(\lambda) + b_b(\lambda)}` is the ratio of backscattering to total absorption+backscattering
    - :math:`G_1, G_2` are Gordon coefficients that account for bidirectional light field geometry
    - :math:`r_{rs}` is subsurface remote sensing reflectance

The subsurface reflectance :math:`r_{rs}` is related to above-surface :math:`R_{rs}` through:

.. math::

    R_{rs} = \frac{A \cdot r_{rs}}{1 - B \cdot r_{rs}}

where :math:`A = 0.52` and :math:`B = 1.7` are standard conversion coefficients.

Standard Gordon Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classical Gordon model uses constant coefficients:

.. math::

    G_1 = 0.0949, \quad G_2 = 0.0794

These values were derived from radiative transfer simulations for typical oceanic conditions (clear sky, moderate sun angle).

.. code-block:: python

    from bing import rt as bing_rt
    import numpy as np

    # Define IOPs at measurement wavelengths
    wavelengths = np.array([410, 440, 490, 510, 555, 670])  # nm

    # Absorption coefficient [m^-1]
    a = np.array([0.050, 0.045, 0.035, 0.038, 0.052, 0.350])

    # Backscattering coefficient [m^-1]
    bb = np.array([0.0025, 0.0020, 0.0015, 0.0013, 0.0010, 0.0006])

    # Calculate Rrs with standard Gordon coefficients
    Rrs = bing_rt.calc_Rrs(a, bb)

    print(f"Rrs at 490 nm: {Rrs[2]:.6f} sr^-1")

Variable Gordon Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recent research (Loisel et al. 2023) shows that Gordon coefficients vary with wavelength due to changes in scattering phase functions and light field geometry. BING provides wavelength-dependent Gordon coefficients derived from Hydrolight radiative transfer simulations.

The variable Gordon formulation uses interpolated coefficients:

.. math::

    G_1(\lambda) \text{ and } G_2(\lambda) \text{ are wavelength-dependent}

Key characteristics of variable Gordon coefficients:

1. **G1 wavelength dependence**:
   - Ranges from ~0.100-0.102 at blue wavelengths (350-450 nm)
   - Decreases to ~0.085-0.097 at red wavelengths (600-750 nm)
   - Generally positive across the spectrum

2. **G2 wavelength dependence**:
   - Positive at blue wavelengths (350-490 nm): ~0.05-0.07
   - Transitions to negative at green-red wavelengths (>490 nm)
   - Reaches large negative values at far-red (>700 nm): -0.5 to -2.0
   - **Negative G2 values are physically valid** and represent changes in light field structure

.. note::
   The transition of G2 from positive to negative around 490-500 nm reflects changes in water absorption and the relative importance of single vs. multiple scattering events.

Loading Variable Gordon Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.rt import rrs
    import numpy as np

    # Define model wavelengths
    wavelengths = np.arange(400, 701, 5)  # 400-700 nm, 5 nm resolution

    # Load wavelength-dependent Gordon coefficients
    G1, G2 = rrs.wave_dependent_gordon(wavelengths)

    # Calculate Rrs with variable Gordon coefficients
    Rrs = bing_rt.calc_Rrs(a, bb, in_G1=G1, in_G2=G2)

    # Examine coefficient variation
    print(f"G1 range: {G1.min():.4f} - {G1.max():.4f}")
    print(f"G2 range: {G2.min():.4f} - {G2.max():.4f}")
    print(f"G2 transitions to negative at: {wavelengths[G2 < 0][0]:.0f} nm")

Gordon Coefficient Data Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variable Gordon coefficients are stored in ``bing/data/RT/gordon_coefficients.csv`` and include:

- **wavelength**: 350-750 nm at 5 nm resolution
- **G1, G2**: Fitted Gordon coefficients
- **G1_err, G2_err**: Uncertainty estimates
- **rRMS, RMS**: Fit quality metrics

Data derived from Loisel et al. (2023) Hydrolight synthetic dataset with ~20,000 bio-optical profiles representing diverse ocean conditions.

When to Use Variable Gordon Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use standard Gordon (constant G1, G2) when:**

- Working with legacy algorithms or historical comparisons
- Processing multispectral data (MODIS, SeaWiFS) with limited spectral information
- Computational speed is critical
- Results need to match standard products

**Use variable Gordon (wavelength-dependent) when:**

- Processing hyperspectral data (PACE, SBG)
- Maximum accuracy is required, especially at red/NIR wavelengths
- Modeling oligotrophic (clear) waters where G2 effects are strongest
- Developing new algorithms or validating models

Comparison of Standard vs. Variable Gordon
-------------------------------------------

Spectral Differences
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    # Compare standard and variable Gordon
    wavelengths = np.arange(400, 751, 1)

    # Variable Gordon coefficients
    G1_var, G2_var = rrs.wave_dependent_gordon(wavelengths)

    # Standard Gordon coefficients
    G1_std = rrs.G1_STANDARD * np.ones_like(wavelengths)
    G2_std = rrs.G2_STANDARD * np.ones_like(wavelengths)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # G1 comparison
    ax1.plot(wavelengths, G1_std, 'k--', label='Standard G1 (0.0949)', linewidth=2)
    ax1.plot(wavelengths, G1_var, 'b-', label='Variable G1', linewidth=2)
    ax1.set_ylabel('G1')
    ax1.set_title('Gordon Coefficient G1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # G2 comparison
    ax2.plot(wavelengths, G2_std, 'k--', label='Standard G2 (0.0794)', linewidth=2)
    ax2.plot(wavelengths, G2_var, 'r-', label='Variable G2', linewidth=2)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('G2')
    ax2.set_title('Gordon Coefficient G2 (note negative values at λ > 490 nm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gordon_coefficients_comparison.png', dpi=150)

Impact on Retrieved Rrs
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate Rrs difference using both formulations
    def compare_gordon_formulations(a, bb, wavelengths):
        """Compare Rrs calculated with standard vs. variable Gordon."""

        # Standard Gordon
        Rrs_std = bing_rt.calc_Rrs(a, bb)

        # Variable Gordon
        G1, G2 = rrs.wave_dependent_gordon(wavelengths)
        Rrs_var = bing_rt.calc_Rrs(a, bb, in_G1=G1, in_G2=G2)

        # Calculate differences
        abs_diff = Rrs_var - Rrs_std
        rel_diff = 100 * (Rrs_var - Rrs_std) / Rrs_std

        return Rrs_std, Rrs_var, abs_diff, rel_diff

    # Example: oligotrophic water
    wavelengths = np.arange(400, 701, 10)
    a_oligo = np.array([0.03, 0.025, 0.02, 0.02, 0.025, 0.035, 0.05,
                        0.08, 0.12, 0.18, 0.25, 0.30, 0.35, 0.40,
                        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                        0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10,
                        1.15, 1.20, 1.25])
    bb_oligo = 0.001 * np.ones_like(a_oligo)

    Rrs_std, Rrs_var, abs_diff, rel_diff = compare_gordon_formulations(
        a_oligo, bb_oligo, wavelengths)

    # Differences typically largest at red/NIR wavelengths
    print(f"Maximum relative difference: {np.max(np.abs(rel_diff)):.2f}%")
    print(f"Wavelength of max difference: {wavelengths[np.argmax(np.abs(rel_diff))]:.0f} nm")

Typical relative differences:

- Blue wavelengths (400-500 nm): 0.5-2%
- Green wavelengths (500-600 nm): 1-5%
- Red wavelengths (600-700 nm): 3-15%

Differences are largest in oligotrophic waters and at longer wavelengths.

Raman Scattering Correction
---------------------------

BING supports Raman scattering corrections to account for the inelastic scattering
contribution to Rrs. This is particularly important for clear (oligotrophic) waters
at wavelengths > 500 nm.

Integrated Raman Correction in calc_Rrs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main ``calc_Rrs`` function accepts optional Raman parameters:

.. code-block:: python

    from bing.rt import rrs, raman
    import numpy as np

    # Define wavelengths
    wave = np.arange(400, 701, 5)

    # IOPs at emission (model) wavelengths
    a = ...   # Total absorption [m^-1]
    bb = ...  # Total backscattering [m^-1]

    # Compute excitation wavelengths (~3400 cm^-1 Raman shift)
    wave_ex = raman.emission_to_excitation_wavelength(wave)

    # IOPs at excitation wavelengths
    a_ex = ...   # Absorption at excitation wavelengths
    bb_ex = ...  # Backscattering at excitation wavelengths

    # Raman backscattering coefficient
    bb_R = raman.raman_backscattering_coeff(wave_ex)

    # Calculate Rrs with Raman correction
    Rrs = rrs.calc_Rrs(a, bb, a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

The Raman correction factor is computed using Sathyendranath & Platt (1998):

.. code-block:: python

    # Calculate correction factor separately
    corr = rrs.calc_raman_correction_factor(a, bb, a_ex, bb_ex, bb_R)
    print(f"Raman correction range: {corr.min():.3f} - {corr.max():.3f}")

Raman Correction in Fitting Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When fitting with BING, enable Raman correction in the parameter configuration:

.. code-block:: python

    from bing.parameters import standard

    params = standard.expb_pow(
        satellite='PACE',
        include_Raman=True,      # Enable Raman correction
        variable_Gordon=True,    # Recommended for best accuracy
        nsteps=40000
    )

The fitting module automatically:

1. Initializes excitation wavelengths for each model
2. Computes IOPs at both emission and excitation wavelengths
3. Applies the Raman correction factor during forward modeling

.. note::
   The correction depends to first order on the solar-spectrum ratio
   :math:`E_d(\lambda')/E_d(\lambda)`. Set it on the absorption model
   with ``a_model.set_raman_Ed(wave_Ed, Ed)`` (the L23 fitting pipeline
   does this automatically from ``correct_atmosphere``); the Ed grid
   must extend ~50 nm blueward of the model grid to cover the Raman
   excitation wavelengths. Without it, the correction falls back to a
   flat spectrum (ratio = 1) and emits a ``RuntimeWarning`` — the flat
   fallback distorts the spectral shape of the correction (validated
   against the Loisel et al. 2023 database: ~+60 % increment error at
   490 nm, −15 % and worse in the red).

.. note::
   For detailed Raman scattering physics and additional functions, see
   :ref:`raman`.

Chlorophyll Fluorescence
------------------------

BING also supports a chlorophyll fluorescence inelastic-emission contribution
to Rrs. The top-level method :func:`bing.rt.calc_Rrs_fluorescence` integrates
the fluorescence reflectance from a band of excitation wavelengths (370-690
nm) to emission wavelengths near the 685 nm primary peak (and optionally the
730 nm secondary peak), following Gordon (1979) and Sathyendranath & Platt
(1998).

The fluorescence calculation depends on a downwelling irradiance spectrum at
the excitation wavelengths. This is provided by the companion package
``correct_atmosphere``:

* Repository: https://github.com/ocean-colour/correct-atmosphere
* Install: ``pip install git+https://github.com/ocean-colour/correct-atmosphere.git``
* Import: ``from correct_atmosphere import downwelling``

.. code-block:: python

    import numpy as np
    from bing.rt import calc_Rrs_fluorescence
    from correct_atmosphere import downwelling

    wave    = np.linspace(650, 750, 101)        # Emission wavelengths
    wave_ex = np.arange(400.0, 681.0, 1.0)      # Excitation wavelengths

    Ed_ex = downwelling.solar_irradiance(wave_ex)
    Ed_em = downwelling.solar_irradiance(685.)

    Rrs_fl = calc_Rrs_fluorescence(
        wave, a_em, bb_em,
        a_ex, bb_ex, aph_ex,
        wave_ex, Ed_ex, Ed_em,
        phi_C=0.02,
        double_gaussian=True,
    )

For the full set of low-level fluorescence routines (emission line shapes,
quantum yields, FLH, ...) see :doc:`chlorophyll_fluorescence`.

.. _rt-dict-from-p:

The Radiative-Transfer Dictionary (``rt_dict_from_p``)
------------------------------------------------------

The fitting routines (:mod:`bing.fitting.chisq_fit`,
:mod:`bing.fitting.inference`, :mod:`bing.fitting.l23`) receive their
radiative-transfer options through a small dictionary, the **rt dict**.
The convenience method :func:`bing.rt.defs.rt_dict_from_p` builds this
dictionary from a parameter named-tuple (see :ref:`parameters`):

.. code-block:: python

    from bing.parameters import standard
    from bing.rt import defs as rt_defs

    # Build a parameter named-tuple (e.g. via standard.expb_pow(...))
    p = standard.expb_pow(satellite='PACE')

    # Or generate one explicitly with the RT options of interest
    from bing.parameters import p_ntuple
    p = p_ntuple.gen(
        model_names=['ExpBricaud', 'Pow'],
        variable_Gordon=True,
        include_Raman=True,
        include_Chl_fl=True,
        phi_C=0.02,
        double_gaussian=True,
    )

    # Convert to the radiative-transfer dictionary
    rt_dict = rt_defs.rt_dict_from_p(p)
    # rt_dict == {
    #     'variable_Gordon':  True,
    #     'include_Raman':    True,
    #     'include_Chl_fl':   True,
    #     'phi_C':            0.02,
    #     'double_gaussian':  True,
    #     'rt_backend':       'gordon',   # default; see RT Backends below
    #     'fit_Bp':           False,
    #     'Bp_value':         0.01,
    #     'include_CDOM_fl':  False,      # see CDOM Fluorescence below
    #     'cdom_fraction':    0.8,
    # }

    # Pass through to a fit
    from bing.fitting import chisq_fit
    ans, cov, idx = chisq_fit.fit(items[0], models, rt_dict, ...)

.. py:function:: bing.rt.defs.rt_dict_from_p(p)

   Build a radiative-transfer options dictionary from a BING parameter
   named-tuple.

   :param p: Parameter named-tuple (typically produced by
       :func:`bing.parameters.p_ntuple.gen` or one of the ``standard.*``
       helpers).
   :returns: ``dict`` with keys ``variable_Gordon``, ``include_Raman``,
       ``include_Chl_fl``, ``phi_C``, ``double_gaussian`` (any attribute
       that is missing on ``p`` is set to ``None``), plus the RT-backend
       keys ``rt_backend``, ``fit_Bp``, ``Bp_value`` and the
       CDOM-fluorescence keys ``include_CDOM_fl``, ``cdom_fraction``.
       Unlike the others, these five get real defaults — ``'gordon'`` /
       ``False`` / ``0.01`` / ``False`` / ``0.8`` — so a legacy ``p``
       (and any rt dict saved before the backend integration) yields a
       fully valid, Gordon-backend dict.
   :rtype: dict

   The returned dictionary is the canonical way to pass radiative-transfer
   options into the forward model and the fitting routines. The
   ``rt_backend``, ``fit_Bp``, ``Bp_value``, ``include_CDOM_fl`` and
   ``cdom_fraction`` keys are documented in :ref:`rt-backends` below;
   :func:`bing.rt.defs.validate_rt_dict` checks the combination once, at
   fit setup.


.. _rt-backends:

RT Backends: Gordon and ``robust.rt``
-------------------------------------

Since the integration with the sibling `retrieve-or-bust
<https://github.com/ocean-colour/retrieve-or-bust>`_ package (the
``robust`` import), the forward model that turns :math:`(a, b_b)` into
:math:`R_{rs}` is selectable through ``rt_dict['rt_backend']``. The four
valid values live in ``bing.rt.defs.RT_BACKENDS``:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Backend
     - Physics
     - Inelastic (Raman / Chl fluorescence / CDOM fluorescence)
   * - ``'gordon'`` (default)
     - BING's own Gordon (1988) elastic model (``bing.rt.rrs``),
       everything documented above on this page
     - Raman and Chl fluorescence — BING's own ``bing.rt.raman`` /
       ``bing.rt.chl_fl`` physics. **No** CDOM fluorescence:
       ``include_CDOM_fl=True`` raises ``ValueError``
   * - ``'robust_ztt'``
     - ``robust.rt``'s analytic (Zaneveld/Twardowski/Tassan-style)
       forward model
     - All three — via ``robust.rt.inelastic`` and
       ``robust.rt.cdom_fl``
   * - ``'robust_hybrid'``
     - ``robust_ztt`` plus a learned emulator correction; valid only
       for wavelengths inside 350–750 nm (its L23/HydroLight training
       domain)
     - All three — via ``robust.rt.inelastic`` and
       ``robust.rt.cdom_fl``
   * - ``'robust_baseline'``
     - ``robust``'s Gordon-compatible refit
     - **None** — ``include_Raman`` / ``include_Chl_fl`` /
       ``include_CDOM_fl`` all raise ``ValueError``

When unset, ``rt_backend`` defaults to ``'gordon'`` and the fit is
byte-identical to pre-integration BING — the integration is
non-breaking for existing configurations and saved rt dicts.

Choosing a backend
~~~~~~~~~~~~~~~~~~

The project has adopted ``robust`` — elastic and inelastic alike — as
the recommended path going forward, while keeping the measured
differences from BING's own Gordon physics clearly documented. In
brief (all numbers measured on real Loisel et al. 2023 data;
``nb/RT/rob_rt_coding_6.ipynb`` is the full four-backend comparison):

* ``'gordon'`` is the original, by far the fastest, and its inelastic
  terms are BING's own fully self-consistent legacy physics — the
  reference the other backends are measured against. It is kept for
  backward compatibility and remains the default, but it is not the
  recommended starting point for new work.
* ``'robust_baseline'`` is a **parity check**: its elastic Rrs agrees
  with ``'gordon'`` to ~2.3e-7 relative (float32-vs-float64 scale).
  Since it cannot carry inelastic terms and reproduces physics
  ``'gordon'`` already gets right, more slowly, reach for it when you
  specifically want an elastic-only ``robust``-side sanity check —
  not as a destination in itself.
* ``'robust_ztt'`` / ``'robust_hybrid'`` compute **genuinely
  different, more physically complete elastic RT** than Gordon's:
  a few percent higher Rrs on real L23 data (``robust_ztt``: 5.6 %
  max / 1.7 % mean; ``robust_hybrid``: 7.0 % max / 1.8 % mean at the
  L23 idx=170 anchor spectrum), propagating into materially different
  retrieved :math:`b_{b,nw}` in particular. This is a deliberate
  physical improvement, not a bug. Prefer ``robust_hybrid`` when the
  emulator correction's added fidelity is worth staying inside its
  350–750 nm domain; ``robust_ztt`` otherwise.

.. warning::

   **Inelastic accuracy caveat.** ``robust``'s Raman and fluorescence
   terms diverge from BING's own Gordon+Raman/fluorescence physics by
   several percent on real L23 data: Raman max 11.4 % / mean 5.6 %
   (worst at 400 nm); fluorescence max 9.2 % / mean 5.8 %
   (mismatched-``Ed`` case) to max 18.5 % / mean 7.2 %
   (``Ed``-matched case). That is orders of magnitude outside the
   project's usual cross-backend tolerance (rtol ≤ 5e-4) and is a
   known, understood physics-composition difference — ``robust``
   interpolates/clamps the emission-grid IOP spectrum to stand in for
   the excitation-wavelength IOPs, rather than re-evaluating the
   parametric models at the true excitation grid — not numerical
   noise (the *elastic* agreement is ~2.3e-7). The gap is documented,
   deliberate to live with, and not yet closed. Elastic-only use of
   the robust backends is unaffected.

Geometry is required (``ObsGeometry``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The robust backends require the observation geometry, which the Gordon
model never needed. It is carried by the frozen dataclass
:class:`bing.rt.geometry.ObsGeometry` — fixed, per-pixel, never-fit
scene metadata:

.. code-block:: python

    from bing.rt.geometry import ObsGeometry

    geom = ObsGeometry(
        theta_s=30.,   # solar zenith angle (deg) -- REQUIRED, no default
        theta_v=0.,    # sensor zenith angle (deg); defaults to nadir
        dphi=0.,       # sensor-sun relative azimuth (deg); default 0
    )

``theta_s`` is **never silently defaulted**: selecting a robust backend
without supplying ``geom`` raises ``ValueError`` at fit setup
(:func:`bing.rt.defs.validate_rt_dict`), before any forward-model call.
The fitting routines thread ``geom`` through as an optional fifth
element of the observation tuple — see :ref:`fitting-robust-backend`
in :doc:`fitting`.

Free or fixed :math:`B_p` (``fit_Bp``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The robust forward models take a phase-function parameter :math:`B_p`
(the particulate backscattering ratio :math:`b_{b,p}/b_p`) that Gordon's
model has no analog of. Two rt-dict keys control it:

* ``fit_Bp`` (default ``False``): when ``True``, :math:`B_p` becomes an
  extra free MCMC/least-squares parameter, appended after the model
  parameters. It is sampled **linearly** (not log10) under a uniform
  prior over ``[BP_PRIOR_PMIN, BP_PRIOR_PMAX] = [0.004, 0.05]``
  (module constants in ``bing.rt.defs``). ``fit_Bp=True`` with
  ``rt_backend='gordon'`` raises — Gordon has no phase-function input.
* ``Bp_value`` (default ``0.01``): the fixed value when
  ``fit_Bp=False``, and the walker-ball seed when ``fit_Bp=True``.

.. _rt-cdom-fluorescence:

CDOM fluorescence (``include_CDOM_fl``, ``cdom_fraction``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The robust backends can add a **third inelastic term** beside Raman
and chlorophyll fluorescence: CDOM (yellow-matter) fluorescence, the
analytic Hawes et al. (1992) FA7 kernel implemented in
``robust.rt.cdom_fl``. Two rt-dict keys control it:

* ``include_CDOM_fl`` (default ``False``): switch the term on. Robust
  backends only — ``'gordon'`` has no CDOM-fluorescence physics at all,
  and ``'robust_baseline'`` is elastic-only; both raise
  ``ValueError`` at fit setup.
* ``cdom_fraction`` (default ``0.8``, ``bing.rt.defs.CDOM_FRACTION_DEFAULT``):
  the CDOM fraction of the a-model's combined dissolved+detrital
  absorption — see the proxy note below. Consulted only when
  ``include_CDOM_fl`` is ``True``.

The kernel's emission source term is the **pure CDOM** absorption
spectrum (:math:`b_{bY} = \tfrac12 a_{cdom}`). BING has no such
spectrum: every a-model with a separable exponential term lumps CDOM
and detritus into a single :math:`a_{dg}`. The adapter therefore
supplies a **fixed-fraction proxy**

.. math::

    a_{cdom}(\lambda) = f_{cdom} \times a_{dg}(\lambda),
    \qquad f_{cdom} = 0.8 \ \text{by default}

where :math:`a_{dg}` comes from the a-model's separable component
(``a_model.eval_a_dg``, i.e. ``eval_anw(..., retsub_comps=True)[0]``).

.. warning::

   :math:`f_{cdom} = 0.8` is a **project decision**, not a measured or
   fitted quantity (JXP, 2026-09-05; ``claude_prompts/rt_tests.md``
   Q32). It exists because BING's :math:`a_{dg}` is CDOM *plus*
   detritus while the Hawes kernel expects CDOM alone, and no BING
   a-model splits the two. Vary it per fit through
   ``p.cdom_fraction`` / ``rt_dict['cdom_fraction']``; the Rrs
   increment is essentially linear in it.

The kernel amplitude ``robust.rt.CDOMFl.scale`` is held **fixed at
1.0** (``bing.evaluate.CDOM_FL_SCALE``) — the reference kernel exactly
as published. It is a differentiable leaf on robust's side (the
``phi_C`` analogue for a future inversion), but BING does not fit it:
``cdom_fraction`` is the only CDOM knob BING exposes.

Requirements and caveats:

* The a-model must expose a separable :math:`a_{dg}`
  (``has_a_dg = True``): ``ExpBricaud``, ``ExpBricaudFix``,
  ``ExpBricaudFree``, ``GIOP``, ``GSM``, ``ExpNMF``. Models with a
  single lumped :math:`a_{nw}` (``Exp``, ``ExpFix``, ``Cst``,
  ``Bricaud``, ``Chase2017``, ``Every``) raise at fit setup, naming
  the offending class.
* Emission is broad and featureless across the blue–green (roughly
  465–570 nm for the kernel's fixed 350–490 nm excitation band) — no
  685 nm-style line.
* robust's kernel integrates excitation from 350 nm, but *clamps*
  (constant extrapolation) any excitation wavelength below the model
  wavelength grid. A fit starting at 400 nm therefore feeds
  :math:`a_{cdom}(400)` to the whole 350–400 nm excitation band.
* robust flags its own CDOM-fluorescence term as analytic-only and
  **not yet validated** against HydroLight truth (its M5): the
  :math:`\delta_C` learned-correction head is defined but untrained.

.. code-block:: python

    from bing.parameters import p_ntuple
    from bing.rt import defs as rt_defs

    p = p_ntuple.gen(model_names=['ExpBricaud', 'Pow'])
    rt_dict = dict(rt_defs.rt_dict_from_p(p),
                   rt_backend='robust_ztt',
                   include_Raman=True,
                   include_Chl_fl=True,
                   include_CDOM_fl=True,
                   cdom_fraction=0.8)

Validation and error behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`bing.rt.defs.validate_rt_dict` is called once per fit, at setup,
and raises ``ValueError`` on any illegal combination:

* ``rt_backend`` not one of ``RT_BACKENDS``;
* ``fit_Bp=True`` with ``rt_backend='gordon'``;
* a robust backend with no ``geom`` (``theta_s`` never defaulted);
* ``rt_backend='robust_hybrid'`` with any model wavelength outside
  [350, 750] nm (the emulator's training range);
* ``include_CDOM_fl=True`` with ``rt_backend='gordon'`` (no such
  physics) or ``'robust_baseline'`` (elastic-only);
* ``include_CDOM_fl=True`` with an a-model that has no separable
  :math:`a_{dg}` component (``models[0].has_a_dg`` is ``False``) —
  the error names the offending model class.

Additionally, ``rt_backend='robust_baseline'`` combined with
``include_Raman=True``, ``include_Chl_fl=True`` or
``include_CDOM_fl=True`` raises in
:func:`bing.evaluate.calc_Rrs_from_models_robust` — the baseline is
elastic-only by construction.

Backend throughput
~~~~~~~~~~~~~~~~~~

Measured warm-path ``log_prob`` throughput at MCMC-realistic call
shapes (single-walker, 5-parameter ``ExpBricaud``+``Pow``, PACE-like
grid; ``dev/rob_rt/benchmark_backends.py``, which documents the
methodology and re-produces these numbers):

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 30

   * - Backend
     - Warm calls/s
     - vs ``gordon``
     - First-call (JIT) cost
   * - ``gordon``
     - 38936
     - 1.000x
     - ~0.0001 s (no JIT)
   * - ``robust_ztt``
     - 5991
     - 0.154x
     - 0.182 s
   * - ``robust_hybrid``
     - 5667
     - 0.146x
     - 0.181 s
   * - ``robust_baseline``
     - 7274
     - 0.187x
     - 0.019 s

The robust backends run through JAX and pay a one-time JIT-compile
cost per process. In :func:`bing.fitting.inference.fit_batch`, every
``ProcessPoolExecutor`` worker compiles its own cache, so an
:math:`n`-worker robust batch pays roughly :math:`n\times` the
single-process compile cost before any fitting happens.

API Reference
-------------

Core Functions
~~~~~~~~~~~~~~

.. py:function:: calc_Rrs(a, bb, in_G1=None, in_G2=None, a_ex=None, bb_ex=None, bb_R=None)

   Calculate remote sensing reflectance from IOPs using Gordon model,
   with optional Raman scattering correction.

   :param a: Total absorption coefficient at emission wavelength [m^-1]
   :type a: float or ndarray
   :param bb: Total backscattering coefficient at emission wavelength [m^-1]
   :type bb: float or ndarray
   :param in_G1: Gordon coefficient G1 (default: 0.0949)
   :type in_G1: float or ndarray, optional
   :param in_G2: Gordon coefficient G2 (default: 0.0794)
   :type in_G2: float or ndarray, optional
   :param a_ex: Absorption at Raman excitation wavelength [m^-1]. Required for Raman correction.
   :type a_ex: float or ndarray, optional
   :param bb_ex: Backscattering at Raman excitation wavelength [m^-1]. Required for Raman correction.
   :type bb_ex: float or ndarray, optional
   :param bb_R: Raman backscattering coefficient [m^-1]. Required for Raman correction.
   :type bb_R: float or ndarray, optional
   :return: Remote sensing reflectance [sr^-1]
   :rtype: float or ndarray
   :raises IOError: If a_ex is provided but bb_ex or bb_R are not.

   **Examples:**

   .. code-block:: python

       # Elastic-only (standard Gordon)
       Rrs = calc_Rrs(a=0.05, bb=0.002)

       # Variable Gordon
       wavelengths = np.arange(400, 701, 5)
       G1, G2 = wave_dependent_gordon(wavelengths)
       Rrs = calc_Rrs(a, bb, in_G1=G1, in_G2=G2)

       # With Raman correction
       from bing.rt import raman
       wave_ex = raman.emission_to_excitation_wavelength(wavelengths)
       bb_R = raman.raman_backscattering_coeff(wave_ex)
       Rrs = calc_Rrs(a, bb, in_G1=G1, in_G2=G2,
                      a_ex=a_ex, bb_ex=bb_ex, bb_R=bb_R)

.. py:function:: calc_Rrs_fluorescence(wavelength, a_em, bb_em, a_ex, bb_ex, aph_ex, wavelength_ex, Ed_ex, Ed_em, mu_d=None, mu_f=None, phi_C=0.02, double_gaussian=True)
   :no-index:

   Calculate the Rrs contribution from chlorophyll fluorescence, integrating
   over excitation wavelengths.

   :param wavelength: Emission wavelength(s) [nm]. Typically 650-750 nm.
   :type wavelength: float or ndarray
   :param a_em: Total absorption at emission wavelength(s) [m^-1]
   :type a_em: float or ndarray
   :param bb_em: Total backscattering at emission wavelength(s) [m^-1]
   :type bb_em: float or ndarray
   :param a_ex: Total absorption at excitation wavelengths [m^-1]
   :type a_ex: ndarray
   :param bb_ex: Total backscattering at excitation wavelengths [m^-1]
   :type bb_ex: ndarray
   :param aph_ex: Phytoplankton absorption at excitation wavelengths [m^-1]
   :type aph_ex: ndarray
   :param wavelength_ex: Excitation wavelengths [nm]
   :type wavelength_ex: ndarray
   :param Ed_ex: Downwelling irradiance at excitation wavelengths
       (from ``correct_atmosphere.downwelling``)
   :type Ed_ex: ndarray
   :param Ed_em: Downwelling irradiance at the (peak) emission wavelength
   :type Ed_em: float or ndarray
   :param phi_C: Quantum yield (default 0.02)
   :type phi_C: float, optional
   :param double_gaussian: Use double-Gaussian emission line (default True)
   :type double_gaussian: bool, optional
   :return: Fluorescence contribution to Rrs [sr^-1]
   :rtype: ndarray

   See :doc:`chlorophyll_fluorescence` for full physics, low-level methods,
   and FLH calculations.

   .. note::

      The upwelling attenuation :math:`\kappa^F(\lambda) =
      (a(\lambda) + b_b(\lambda))/\mu_f` is evaluated at every emission
      wavelength, not pinned to the 685 nm peak. This is what allows the
      double-Gaussian model to reproduce the correct (suppressed) 730 nm
      secondary shoulder, where pure-water absorption is ~4x larger than
      at 685 nm.

.. py:function:: calc_raman_correction_factor(a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=1.0, include_second_order=True)

   Calculate the multiplicative correction factor for Raman scattering.

   :param a_em: Absorption at emission wavelength [m^-1]
   :type a_em: float or ndarray
   :param bb_em: Backscattering at emission wavelength [m^-1]
   :type bb_em: float or ndarray
   :param a_ex: Absorption at excitation wavelength [m^-1]
   :type a_ex: float or ndarray
   :param bb_ex: Backscattering at excitation wavelength [m^-1]
   :type bb_ex: float or ndarray
   :param bb_R: Raman backscattering coefficient [m^-1]
   :type bb_R: float or ndarray
   :param Ed_ratio: Ratio Ed(λ')/Ed(λ). Default is 1.0
   :type Ed_ratio: float, optional
   :param include_second_order: Include second-order Raman terms. Default is True
   :type include_second_order: bool, optional
   :return: Correction factor (R_elastic + R_raman) / R_elastic
   :rtype: float or ndarray

   Typical correction factors range from 1.0 to ~1.25, with largest
   corrections in clear (oligotrophic) waters at longer wavelengths.

.. py:function:: wave_dependent_gordon(wave, bounds_error=True)

   Load wavelength-dependent Gordon coefficients.

   Reads from ``bing/data/RT/gordon_coefficients.csv`` and interpolates
   to requested wavelengths using cubic splines.

   :param wave: Wavelength array [nm]
   :type wave: ndarray
   :param bounds_error: Raise error if wavelengths outside 350-750 nm range
   :type bounds_error: bool, optional
   :return: G1 and G2 arrays at requested wavelengths
   :rtype: tuple(ndarray, ndarray)

   **Examples:**

   .. code-block:: python

       # Get coefficients for PACE wavelengths
       pace_waves = np.arange(340, 701, 5)
       G1, G2 = wave_dependent_gordon(pace_waves, bounds_error=False)

       # Examine spectral structure
       import matplotlib.pyplot as plt
       plt.plot(pace_waves, G1, label='G1')
       plt.plot(pace_waves, G2, label='G2')
       plt.xlabel('Wavelength (nm)')
       plt.ylabel('Gordon Coefficient')
       plt.legend()

Conversion Functions
~~~~~~~~~~~~~~~~~~~~

.. py:function:: Rrs_to_rrs(Rrs, A=0.52, B=1.7)

   Convert above-surface Rrs to subsurface rrs.

   :param Rrs: Remote sensing reflectance (above surface) [sr^-1]
   :type Rrs: ndarray
   :param A: Conversion coefficient (default: 0.52)
   :type A: float, optional
   :param B: Conversion coefficient (default: 1.7)
   :type B: float, optional
   :return: Subsurface remote sensing reflectance [sr^-1]
   :rtype: ndarray

   .. math::

       r_{rs} = \frac{R_{rs}}{A + B \cdot R_{rs}}

.. py:function:: rrs_to_Rrs(rrs, A=0.52, B=1.7)

   Convert subsurface rrs to above-surface Rrs.

   :param rrs: Subsurface remote sensing reflectance [sr^-1]
   :type rrs: ndarray
   :param A: Conversion coefficient (default: 0.52)
   :type A: float, optional
   :param B: Conversion coefficient (default: 1.7)
   :type B: float, optional
   :return: Remote sensing reflectance (above surface) [sr^-1]
   :rtype: ndarray

   .. math::

       R_{rs} = \frac{A \cdot r_{rs}}{1 - B \cdot r_{rs}}

Constants
~~~~~~~~~

.. py:data:: G1_STANDARD
   :value: 0.0949

   Standard (constant) Gordon coefficient G1 from Gordon et al. (1988).

.. py:data:: G2_STANDARD
   :value: 0.0794

   Standard (constant) Gordon coefficient G2 from Gordon et al. (1988).

.. py:data:: A_Rrs
   :value: 0.52

   Conversion coefficient from subsurface to above-surface reflectance.

.. py:data:: B_Rrs
   :value: 1.7

   Conversion coefficient from subsurface to above-surface reflectance.

Complete Workflow Example
--------------------------

Forward Modeling with Variable Gordon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates a complete forward modeling workflow using the variable Gordon formulation:

.. code-block:: python

    import numpy as np
    from bing import rt as bing_rt
    from bing.models import utils as model_utils
    from bing.parameters import standard
    import matplotlib.pyplot as plt

    # ===== 1. Setup =====
    # Define hyperspectral wavelengths (PACE-like)
    wavelengths = np.arange(400, 701, 5)

    # Initialize bio-optical models
    params = standard.expb_pow(satellite='PACE', variable_Gordon=True)
    models = model_utils.init(params.model_names, wavelengths)

    # ===== 2. Set bio-optical parameters =====
    # Example: oligotrophic water
    Chl = 0.1  # mg/m³
    Sdg = 0.015  # CDOM spectral slope [nm^-1]
    Adg_440 = 0.01  # CDOM absorption at 440 nm [m^-1]
    Bnw_ref = 0.0008  # Particle backscatter at 600 nm [m^-1]
    beta = 1.5  # Spectral slope for backscatter

    # ===== 3. Configure models =====
    # Set chlorophyll for phytoplankton absorption
    models[0].set_aph(Chl=Chl)

    # Model parameters (in log10 space for amplitudes)
    model_params = np.array([
        np.log10(Adg_440),  # log10(A_dg)
        Sdg,                 # S_dg
        np.log10(0.05582 * Chl),  # log10(A_ph) from Bricaud
        np.log10(Bnw_ref),   # log10(B_nw)
        beta                 # spectral slope
    ])

    # ===== 4. Evaluate IOPs =====
    # Non-water absorption
    anw = models[0].eval_anw(model_params[:3])

    # Non-water backscattering
    bbnw = models[1].eval_bbnw(model_params[3:])

    # Add water components
    a_total = models[0].a_w + anw[0]  # Total absorption
    bb_total = models[1].bb_w + bbnw[0]  # Total backscattering

    # ===== 5. Calculate Rrs with variable Gordon =====
    G1, G2 = bing_rt.rrs.wave_dependent_gordon(wavelengths)
    Rrs = bing_rt.calc_Rrs(a_total, bb_total, in_G1=G1, in_G2=G2)

    # ===== 6. Compare with standard Gordon =====
    Rrs_standard = bing_rt.calc_Rrs(a_total, bb_total)

    # ===== 7. Visualize results =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot IOPs
    axes[0, 0].semilogy(wavelengths, a_total, 'b-', label='Total absorption', linewidth=2)
    axes[0, 0].semilogy(wavelengths, models[0].a_w, 'b--', label='Water absorption', alpha=0.5)
    axes[0, 0].set_ylabel('Absorption [m$^{-1}$]')
    axes[0, 0].set_title('Absorption Coefficient')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(wavelengths, bb_total, 'r-', label='Total backscatter', linewidth=2)
    axes[0, 1].semilogy(wavelengths, models[1].bb_w, 'r--', label='Water backscatter', alpha=0.5)
    axes[0, 1].set_ylabel('Backscattering [m$^{-1}$]')
    axes[0, 1].set_title('Backscattering Coefficient')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot Rrs comparison
    axes[1, 0].plot(wavelengths, Rrs_standard, 'k--', label='Standard Gordon', linewidth=2)
    axes[1, 0].plot(wavelengths, Rrs, 'g-', label='Variable Gordon', linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('R$_{rs}$ [sr$^{-1}$]')
    axes[1, 0].set_title('Remote Sensing Reflectance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot relative difference
    rel_diff = 100 * (Rrs - Rrs_standard) / Rrs_standard
    axes[1, 1].plot(wavelengths, rel_diff, 'purple', linewidth=2)
    axes[1, 1].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[1, 1].set_xlabel('Wavelength (nm)')
    axes[1, 1].set_ylabel('Relative Difference (%)')
    axes[1, 1].set_title('Variable - Standard Gordon')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forward_model_variable_gordon.png', dpi=150)

    # Print summary statistics
    print(f"Chlorophyll: {Chl:.2f} mg/m³")
    print(f"CDOM a_dg(440): {Adg_440:.4f} m⁻¹")
    print(f"Mean Rrs difference: {np.mean(np.abs(rel_diff)):.2f}%")
    print(f"Max Rrs difference: {np.max(np.abs(rel_diff)):.2f}% at {wavelengths[np.argmax(np.abs(rel_diff))]} nm")

Best Practices
--------------

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

When using variable Gordon coefficients in fitting workflows:

.. code-block:: python

    from bing.parameters import standard

    # Enable variable Gordon in parameter configuration
    params = standard.expb_pow(
        satellite='PACE',
        variable_Gordon=True,  # Use wavelength-dependent coefficients
        add_noise=True,
        nsteps=40000
    )

    # The fitting module will automatically use G1(λ), G2(λ)

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

    from bing.rt import rrs

    # Handle wavelengths outside valid range
    wavelengths = np.arange(300, 801, 5)  # Extends beyond 350-750 nm range

    try:
        G1, G2 = rrs.wave_dependent_gordon(wavelengths, bounds_error=True)
    except ValueError as e:
        print(f"Wavelength out of range: {e}")
        # Use bounds_error=False to extrapolate
        G1, G2 = rrs.wave_dependent_gordon(wavelengths, bounds_error=False)
        print("Warning: Extrapolating Gordon coefficients beyond valid range")

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

Gordon coefficients must satisfy physical constraints:

- G1 > 0 (always positive)
- ``|G2|`` < 1 (bounded, can be negative)
- G1 >> ``|G2|`` at short wavelengths
- Rrs >= 0 for physical solutions

.. code-block:: python

    # Verify physical constraints
    def check_gordon_physics(G1, G2, wavelengths):
        """Validate Gordon coefficients."""

        checks = {
            'G1_positive': np.all(G1 > 0),
            'G2_bounded': np.all(np.abs(G2) < 1.0),
            'G1_magnitude': np.all(G1 > np.abs(G2))
        }

        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}: {passed}")

        return all(checks.values())

    # Run validation
    wavelengths = np.arange(400, 701, 5)
    G1, G2 = rrs.wave_dependent_gordon(wavelengths)
    is_valid = check_gordon_physics(G1, G2, wavelengths)

Performance Considerations
--------------------------

Computation Speed
~~~~~~~~~~~~~~~~~

Loading variable Gordon coefficients requires:

1. Reading CSV file (~100 rows)
2. Cubic spline interpolation to model wavelengths

This adds negligible overhead (~1 ms) and should be done once per model initialization.

.. code-block:: python

    import time

    # Benchmark
    wavelengths = np.arange(400, 701, 1)  # 301 wavelengths

    # Standard Gordon (constant)
    t0 = time.time()
    for _ in range(1000):
        Rrs = bing_rt.calc_Rrs(a, bb)
    t_standard = time.time() - t0

    # Variable Gordon (precomputed)
    G1, G2 = rrs.wave_dependent_gordon(wavelengths)
    t0 = time.time()
    for _ in range(1000):
        Rrs = bing_rt.calc_Rrs(a, bb, in_G1=G1, in_G2=G2)
    t_variable = time.time() - t0

    print(f"Standard Gordon: {t_standard:.3f} s")
    print(f"Variable Gordon: {t_variable:.3f} s")
    print(f"Overhead: {100*(t_variable-t_standard)/t_standard:.1f}%")

Typical overhead: <1% when G1, G2 are precomputed.

Memory Usage
~~~~~~~~~~~~

Variable Gordon coefficients add:

- 2 arrays × n_wavelengths × 8 bytes
- Example: 301 wavelengths = ~5 KB

Negligible compared to MCMC chains or model data.

References
----------

**Gordon Semi-Analytical Model:**

- Gordon, H.R. et al. (1988). "A semianalytic radiance model of ocean color,"
  *Journal of Geophysical Research* 93(D9), 10909-10924.
  https://doi.org/10.1029/JD093iD09p10909

**Variable Gordon Coefficients:**

- Loisel, H., Vantrepotte, V., et al. (2023). "Ocean color remote sensing of
  coastal waters: Are we ready for hyperspectral satellites?" *Journal of
  Geophysical Research: Oceans* (in preparation).

**Radiative Transfer Theory:**

- Mobley, C.D. (1994). *Light and Water: Radiative Transfer in Natural Waters*.
  Academic Press.

- Morel, A. and Gentili, B. (1996). "Diffuse reflectance of oceanic waters. III.
  Implication of bidirectionality for the remote-sensing problem,"
  *Applied Optics* 35(24), 4850-4862.

**Raman Scattering:**

- Sathyendranath, S. and Platt, T. (1998). "Ocean-colour model incorporating
  transspectral processes," *Applied Optics* 37, 2216-2227.

- Bartlett, J.S. et al. (1998). "Raman scattering by pure water and seawater,"
  *Applied Optics* 37, 3324-3332.

**Related BING Documentation:**

- :ref:`models` - Bio-optical model formulations
- :ref:`fitting` - Parameter estimation algorithms
- :ref:`data_processing` - End-to-end workflow examples

See Also
--------

* :py:mod:`bing.rt.raman` - Raman scattering corrections (inelastic)
* :py:mod:`bing.rt.chl_fl` - Chlorophyll fluorescence (inelastic)
* :py:mod:`bing.rt.geometry` - ``ObsGeometry`` for the robust backends
* :py:mod:`bing.models` - Bio-optical model components
* :py:mod:`bing.fitting` - Inverse retrieval algorithms
