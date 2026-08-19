"""
L23-anchored regression tests for BING's inelastic RT terms.

These pin BING's Raman correction factor and additive chlorophyll
fluorescence term to HydroLight truth from the Loisel et al. (2023)
database (fixture: 40 scenes at solar zenith 30 deg; see
files/gen_l23_inelastic_fixture.py). The L23 inelastic runs used
HydroLight defaults (Mobley 2012 Raman settings, phi_C = 0.02), which
are also BING's defaults, so the comparisons isolate the formulation.

Truth signals:
- Raman:        Rrs_X2 / Rrs_X1  (multiplicative, exactly how BING
                applies its correction factor)
- fluorescence: Rrs_X4 - Rrs_X2  (additive)

These tests guard the two 2026-08 fixes:
- the 1/pi irradiance-to-rrs conversion in calc_Rrs_fluorescence
  (without it the term is ~3x too large), and
- the true Ed(lambda')/Ed(lambda) ratio in the Raman correction
  (the flat-Ed fallback distorts its spectral shape).
"""

import os

import numpy as np
from scipy.interpolate import interp1d

from bing.rt import raman
from bing.rt import rrs as bing_rrs

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'files', 'l23_inelastic_fixture.npz')


def _load():
    d = np.load(FIXTURE)
    return {k: d[k].astype(float) for k in d.files}


def test_raman_correction_matches_l23():
    """True-Ed Raman correction tracks the L23 X2/X1 ratio in the
    green-red (median increment error < 15% over 550-700 nm)."""
    d = _load()
    wave = d['wave']

    # Emission grid: excitation must stay inside the 350 nm grid edge
    keep = wave >= 400.
    wave_em = wave[keep]
    wave_ex = raman.emission_to_excitation_wavelength(wave_em)

    a_em, bb_em = d['a'][:, keep], d['bb'][:, keep]
    f_a = interp1d(wave, d['a'], axis=-1, kind='linear')
    f_bb = interp1d(wave, d['bb'], axis=-1, kind='linear')
    a_ex, bb_ex = f_a(wave_ex), f_bb(wave_ex)
    bb_R = raman.raman_backscattering_coeff(wave_ex)

    f_Ed = interp1d(wave, d['Ed'], kind='linear')
    Ed_ratio = f_Ed(wave_ex) / f_Ed(wave_em)

    corr = bing_rrs.calc_raman_correction_factor(
        a_em, bb_em, a_ex, bb_ex, bb_R, Ed_ratio=Ed_ratio)
    truth = d['Rrs2'][:, keep] / d['Rrs1'][:, keep]

    band = (wave_em >= 550.) & (wave_em <= 700.)
    incr_err = (corr[:, band] - 1.) / (truth[:, band] - 1.) - 1.
    med = np.median(incr_err)
    assert abs(med) < 0.15, f'median Raman increment error {med:+.2f}'

    # And the flat-Ed fallback must NOT silently become the default
    # again: it is markedly worse in the blue (increment ~+60% at 490).
    corr_flat = bing_rrs.calc_raman_correction_factor(
        a_em, bb_em, a_ex, bb_ex, bb_R)
    i490 = int(np.argmin(np.abs(wave_em - 490.)))
    flat_err = np.median(
        (corr_flat[:, i490] - 1.) / (truth[:, i490] - 1.) - 1.)
    true_err = np.median(
        (corr[:, i490] - 1.) / (truth[:, i490] - 1.) - 1.)
    assert abs(true_err) < abs(flat_err)


def test_fluorescence_matches_l23():
    """The fluorescence term matches the L23 X4-X2 difference at the
    685 nm peak to +/-15% (median over scenes)."""
    d = _load()
    wave = d['wave']

    em = (wave >= 650.) & (wave <= 750.)
    ex = (wave >= 370.) & (wave <= 690.)

    # Single Gaussian: L23/HydroLight used the single 685 nm line.
    Rrs_fl = bing_rrs.calc_Rrs_fluorescence(
        wave[em], d['a'][:, em], d['bb'][:, em],
        d['a'][:, ex], d['bb'][:, ex], d['aph'][:, ex],
        wave[ex], d['Ed'][ex], d['Ed'][em],
        phi_C=0.02, double_gaussian=False)

    truth = d['Rrs4'][:, em] - d['Rrs2'][:, em]

    i685 = int(np.argmin(np.abs(wave[em] - 685.)))
    ratio = np.median(Rrs_fl[:, i685] / truth[:, i685])
    assert 0.85 < ratio < 1.15, f'685 nm model/truth ratio {ratio:.2f}'

    # Spectral shape: the peak must sit at the 685 nm bin.
    ipk = int(np.argmax(np.median(Rrs_fl, axis=0)))
    assert abs(wave[em][ipk] - 685.) <= 5.
