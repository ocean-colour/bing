""" Tests for raman """""
import os

import numpy as np

import pytest
import matplotlib.pyplot as plt

from ocpy.utils import plotting
from bing.rt import raman 


from IPython import embed


def test_raman_seawater():

    print("Raman Scattering in Seawater")
    print("=" * 50)

    # Example 1: Scattering coefficient at various wavelengths
    wavelengths = np.array([350, 400, 450, 488, 500, 550, 600])
    print("\nRaman Scattering Coefficient b_R(λ'):")
    print("-" * 40)
    for lam in wavelengths:
        b_R = raman.raman_scattering_coeff(lam)
        lam_em = raman.excitation_to_emission_wavelength(lam)
        print(f"  λ' = {lam:3.0f} nm → λ ≈ {lam_em:5.1f} nm : "
              f"b_R = {b_R:.3e} m⁻¹")

    # Example 2: Summary at 488 nm
    print("\n" + "=" * 50)
    print("Summary at λ' = 488 nm (Argon laser line):")
    print("-" * 40)
    summary = raman.summary_at_wavelength(488)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4e}" if value < 0.01 else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Example 3: Backscattering coefficient
    print("\n" + "=" * 50)
    print("Raman Backscattering Coefficient b_bR(λ'):")
    print("-" * 40)
    for lam in wavelengths:
        b_bR = raman.raman_backscattering_coeff(lam)
        b_R = raman.raman_scattering_coeff(lam)
        print(f"  λ' = {lam:3.0f} nm : b_bR = {b_bR:.3e} m⁻¹  "
              f"(ratio = {b_bR/b_R:.3f})")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Scattering coefficient vs wavelength
    ax1 = axes[0, 0]
    lam = np.linspace(300, 700, 200)
    b_R = raman.raman_scattering_coeff(lam)
    ax1.semilogy(lam, b_R * 1e4, 'b-', linewidth=2)
    ax1.set_xlabel(r'Excitation Wavelength $\lambda$\' (nm)')
    ax1.set_ylabel(r'$b_R \, \rm (×10^{-4} m^{-1})$')
    ax1.set_title('Raman Scattering Coefficient')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=2.6, color='r', linestyle='--', label='b_R(488nm) = 2.6×10⁻⁴')
    ax1.legend()

    # Plot 2: Wavenumber distribution function
    ax2 = axes[0, 1]
    delta_nu = np.linspace(2800, 4000, 300)
    g = raman.wavenumber_distribution(delta_nu)
    ax2.plot(delta_nu, g, 'b-', linewidth=2)
    ax2.set_xlabel(r'Wavenumber Shift $\Delta \nu$ (cm⁻¹)')
    ax2.set_ylabel(r'$g(\Delta \nu$) (cm)')
    ax2.set_title('Wavenumber Distribution (Walrafen 1967)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Emission spectra for different excitation wavelengths
    ax3 = axes[1, 0]
    for lam_ex in [400, 450, 500, 550]:
        lam_em, intensity = raman.get_emission_spectrum(lam_ex)
        # Normalize for plotting
        intensity_norm = intensity / intensity.max()
        ax3.plot(lam_em, intensity_norm, label=f'λ\' = {lam_ex} nm', linewidth=2)
    ax3.set_xlabel('Emission Wavelength λ (nm)')
    ax3.set_ylabel('Relative Intensity')
    ax3.set_title('Raman Emission Spectra')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Phase function
    ax4 = axes[1, 1]
    psi = np.linspace(0, np.pi, 180)
    psi_deg = np.degrees(psi)
    phase = raman.raman_phase_function(psi)
    ax4.plot(psi_deg, phase, 'b-', linewidth=2)
    ax4.set_xlabel('Scattering Angle ψ (degrees)')
    ax4.set_ylabel('β_R(ψ) (sr⁻¹)')
    ax4.set_title('Raman Phase Function')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=90, color='r', linestyle='--', alpha=0.5, label='Forward/Backward boundary')
    ax4.axvline(x=180, color='g', linestyle='--', alpha=0.5, label='Backscatter (180°)')
    ax4.legend()

    # 
    for ax in [ax1, ax2, ax3, ax4]:
        plotting.set_fontsize(ax, 15)

    plt.tight_layout()
    plt.savefig('raman_seawater_plots.png', dpi=150)
    plt.close()

    print("\n" + "=" * 50)
    print("Plots saved to: raman_seawater_plots.png")

#if __name__ == "__main__":
#    raman_seawater()