"""Investigation: double_gaussian vs single_gaussian in calc_R_fluorescence_integrated.

Goal
----
The prompt notes that, when using ``double_gaussian=True``, the secondary
fluorescence peak near 730 nm looks offset and too strong relative to the
primary peak at 685 nm.

This module reproduces the calculation, isolates the cause, and proposes a
fix.  It is exploratory — see the Logs section in
``prompts/chl_fl.md`` for the resulting recommendations.

Run from the bing repo root::

    conda run -n ocean14 python dev/ChlFl/double_gaussian.py

Two figures are written next to this file:

- ``double_gaussian_emission_shape.png`` — the bare emission line shapes
  ``h_C(λ)`` for the single vs double Gaussian.
- ``double_gaussian_Rrs_fl.png`` — Rrs_fl(λ) computed two ways:
  (a) the current ``bing.rt.rrs.calc_Rrs_fluorescence`` which pins
  κ_F to the value at 685 nm; (b) a per-wavelength κ_F reference
  implementation written here for comparison.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ocpy.water import absorption as water_abs

from bing.rt import chl_fl
from bing.rt import rrs as bing_rrs


# ---------------------------------------------------------------------------
# Reference implementation: integrate fluorescence with per-emission-wavelength
# upwelling attenuation κ_F(λ_em).  This is what calc_Rrs_fluorescence in
# bing.rt.rrs *should* do for emission wavelengths far from 685 nm.
# ---------------------------------------------------------------------------

def calc_Rrs_fluorescence_per_lambda(
    wavelength, a_em, bb_em,
    a_ex, bb_ex, aph_ex,
    wavelength_ex, Ed_ex, Ed_em,
    phi_C=0.02, mu_d=0.9, mu_f=0.5,
    double_gaussian=False,
):
    """Rrs_fl(λ_em) with κ_F evaluated at each emission wavelength.

    Same formulation as bing.rt.rrs.calc_Rrs_fluorescence except κ_F(λ_em)
    is *not* frozen at the value at 685 nm.  This is the correction the
    investigation argues for.
    """
    wavelength = np.atleast_1d(wavelength)

    # Emission line shape at each emission wavelength
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    # κ_F is a vector of length n_em
    kappa_F_em = (a_em + bb_em) / mu_f

    K_ex = (a_ex + bb_ex) / mu_d
    bb_F = chl_fl.fluorescence_backscattering_coeff(aph_ex, phi_C)

    R_F = np.zeros_like(wavelength, dtype=float)
    for i, lam_em in enumerate(wavelength):
        # λ' / λ_em — proper energy conversion factor per emission wavelength
        lambda_ratio = wavelength_ex / lam_em
        integrand = Ed_ex * lambda_ratio * (bb_F / mu_d) / (K_ex + kappa_F_em[i])
        R_F[i] = np.trapezoid(integrand, x=wavelength_ex)

    R_F /= Ed_em

    # Convert subsurface reflectance to Rrs and apply the emission shape
    Rrs_fl = h_C * bing_rrs.A_Rrs * R_F / (1 - bing_rrs.B_Rrs * R_F)
    return Rrs_fl


# ---------------------------------------------------------------------------
# Build a flat-ish test scene: a_nw, bb_nw constant; water IOPs from ocpy
# ---------------------------------------------------------------------------

def build_scene():
    """Return a dict with the inputs needed by both calculators.

    Uses a constant non-water absorption / backscattering so the only
    spectral feature in κ_F(λ_em) comes from pure-water absorption — that
    isolates the bug.
    """
    wave_em = np.arange(650.0, 760.0, 1.0)
    wave_ex = np.arange(400.0, 685.0, 5.0)

    # Pure water IOPs
    a_w_em = water_abs.a_water(wave_em)
    a_w_ex = water_abs.a_water(wave_ex)
    # bb_w from a simple Smith & Baker-style scaling is overkill here; use a
    # tiny constant to keep bb_em ≪ a_em in the red, which is realistic.
    bb_w_em = 4.0e-4 * (500.0 / wave_em) ** 4.32
    bb_w_ex = 4.0e-4 * (500.0 / wave_ex) ** 4.32

    # Non-water IOPs (constant — they aren't what the bug depends on)
    a_nw = 0.05
    bb_nw = 2.0e-3
    aph = 0.02

    a_em = a_w_em + a_nw
    bb_em = bb_w_em + bb_nw
    a_ex = a_w_ex + a_nw
    bb_ex = bb_w_ex + bb_nw
    aph_ex = aph * np.ones_like(wave_ex)

    # Flat downwelling irradiance — keeps the comparison clean
    Ed_ex = np.ones_like(wave_ex)
    Ed_em = 1.0

    return dict(
        wave_em=wave_em, wave_ex=wave_ex,
        a_em=a_em, bb_em=bb_em,
        a_ex=a_ex, bb_ex=bb_ex, aph_ex=aph_ex,
        Ed_ex=Ed_ex, Ed_em=Ed_em,
        a_w_em=a_w_em,
    )


# ---------------------------------------------------------------------------
# Drivers + plots
# ---------------------------------------------------------------------------

def plot_emission_shapes(outdir: Path):
    wave = np.linspace(640.0, 800.0, 401)
    h_single = chl_fl.emission_line_single_gaussian(wave)
    h_double = chl_fl.emission_line_double_gaussian(wave)

    # Decompose the double Gaussian into its primary and secondary pieces so
    # the reader can see where the 25 % weight on the wider secondary peak
    # actually lands.
    sigma1 = chl_fl.SIGMA_FL_PRIMARY
    sigma2 = chl_fl.SIGMA_FL_SECONDARY
    g1 = (1.0 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((wave - chl_fl.LAMBDA_FL_PRIMARY) / sigma1) ** 2)
    g2 = (1.0 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((wave - chl_fl.LAMBDA_FL_SECONDARY) / sigma2) ** 2)
    primary = chl_fl.WEIGHT_PRIMARY * g1
    secondary = chl_fl.WEIGHT_SECONDARY * g2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(wave, h_single, 'C0-', lw=2, label='single Gaussian')
    ax.plot(wave, h_double, 'C3-', lw=2, label='double Gaussian (total)')
    ax.plot(wave, primary, 'C3--', lw=1, label='0.75 × G(685, 10.6)')
    ax.plot(wave, secondary, 'C3:', lw=1, label='0.25 × G(730, 21.2)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$h_C(\lambda)$  [nm$^{-1}$]')
    ax.set_title('Fluorescence emission line shapes')
    ax.legend()

    # Sanity check normalizations (printed, not just plotted)
    int_single = np.trapezoid(h_single, wave)
    int_double = np.trapezoid(h_double, wave)
    ax.text(0.02, 0.95,
            f'∫ single dλ = {int_single:.4f}\n∫ double dλ = {int_double:.4f}',
            transform=ax.transAxes, va='top', fontsize=9)

    fig.tight_layout()
    path = outdir / 'double_gaussian_emission_shape.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}')

    print(f'  emission-shape integrals: single={int_single:.4f}, '
          f'double={int_double:.4f}')
    return wave, h_single, h_double


def run_and_plot_rrs(scene: dict, outdir: Path):
    """Compute Rrs_fl with the current code and the per-λ reference."""
    args = (
        scene['wave_em'], scene['a_em'], scene['bb_em'],
        scene['a_ex'], scene['bb_ex'], scene['aph_ex'],
        scene['wave_ex'], scene['Ed_ex'], scene['Ed_em'],
    )

    # Current implementation (κ_F frozen at 685 nm)
    Rrs_cur_single = bing_rrs.calc_Rrs_fluorescence(
        *args, phi_C=0.02, double_gaussian=False)
    Rrs_cur_double = bing_rrs.calc_Rrs_fluorescence(
        *args, phi_C=0.02, double_gaussian=True)

    # Reference: per-emission-wavelength κ_F
    Rrs_ref_single = calc_Rrs_fluorescence_per_lambda(
        *args, phi_C=0.02, double_gaussian=False)
    Rrs_ref_double = calc_Rrs_fluorescence_per_lambda(
        *args, phi_C=0.02, double_gaussian=True)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax = axes[0]
    ax.plot(scene['wave_em'], Rrs_cur_single, 'C0-', lw=2,
            label='current code, single Gaussian')
    ax.plot(scene['wave_em'], Rrs_cur_double, 'C3-', lw=2,
            label='current code, double Gaussian')
    ax.plot(scene['wave_em'], Rrs_ref_single, 'C0--', lw=2,
            label=r'per-$\lambda$ $\kappa_F$, single Gaussian')
    ax.plot(scene['wave_em'], Rrs_ref_double, 'C3--', lw=2,
            label=r'per-$\lambda$ $\kappa_F$, double Gaussian')
    ax.set_ylabel(r'$R_{rs}^{fl}(\lambda)$  [sr$^{-1}$]')
    ax.set_title('Fluorescence Rrs: current code vs per-λ κ_F')
    ax.legend(fontsize=9)

    # Also overlay the water absorption to make the cause obvious
    ax2 = axes[1]
    ax2.plot(scene['wave_em'], scene['a_w_em'], 'k-', lw=2,
             label=r'$a_w(\lambda_{em})$')
    ax2.axvline(685.0, color='C3', ls=':', lw=1, alpha=0.6)
    ax2.axvline(730.0, color='C3', ls=':', lw=1, alpha=0.6)
    ax2.set_xlabel('Emission wavelength (nm)')
    ax2.set_ylabel(r'$a_w$  [m$^{-1}$]')
    ax2.set_yscale('log')
    ax2.legend()

    fig.tight_layout()
    path = outdir / 'double_gaussian_Rrs_fl.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}')

    # ---- Quantitative summary at the two peaks ----
    i685 = int(np.argmin(np.abs(scene['wave_em'] - 685.0)))
    i730 = int(np.argmin(np.abs(scene['wave_em'] - 730.0)))

    def ratio(R):
        return R[i730] / R[i685]

    print('\nPeak values (sr^-1):')
    print(f"  current   single: Rrs[685]={Rrs_cur_single[i685]:.3e}, "
          f"Rrs[730]={Rrs_cur_single[i730]:.3e},  730/685={ratio(Rrs_cur_single):.3f}")
    print(f"  current   double: Rrs[685]={Rrs_cur_double[i685]:.3e}, "
          f"Rrs[730]={Rrs_cur_double[i730]:.3e},  730/685={ratio(Rrs_cur_double):.3f}")
    print(f"  per-λ κ_F single: Rrs[685]={Rrs_ref_single[i685]:.3e}, "
          f"Rrs[730]={Rrs_ref_single[i730]:.3e},  730/685={ratio(Rrs_ref_single):.3f}")
    print(f"  per-λ κ_F double: Rrs[685]={Rrs_ref_double[i685]:.3e}, "
          f"Rrs[730]={Rrs_ref_double[i730]:.3e},  730/685={ratio(Rrs_ref_double):.3f}")

    print('\nWater absorption at peaks:')
    print(f"  a_w(685) = {scene['a_w_em'][i685]:.3f} m^-1")
    print(f"  a_w(730) = {scene['a_w_em'][i730]:.3f} m^-1")
    print(f"  ratio a_w(730)/a_w(685) = "
          f"{scene['a_w_em'][i730] / scene['a_w_em'][i685]:.2f}")

    return dict(
        Rrs_cur_single=Rrs_cur_single,
        Rrs_cur_double=Rrs_cur_double,
        Rrs_ref_single=Rrs_ref_single,
        Rrs_ref_double=Rrs_ref_double,
    )


def main():
    outdir = Path(__file__).parent
    print('=== Step 1: emission line shapes ===')
    plot_emission_shapes(outdir)

    print('\n=== Step 2: build scene + compute Rrs_fl ===')
    scene = build_scene()
    run_and_plot_rrs(scene, outdir)


if __name__ == '__main__':
    main()
