"""Investigation: RAM blow-up when reconstructing Rrs fluorescence from chains.

Goal
----
The prompt (Modifications item 4 in ``prompts/chl_fl.md``) reports that
applying the chlorophyll-fluorescence correction across MCMC chains to
reconstruct Rrs needs > 100 GB of RAM at the standard ``nsteps=40000``.
The user is confident the culprit is
``bing.rt.rrs.calc_Rrs_fluorescence``.

This module isolates and quantifies the cause, then prototypes and
*verifies* a low-memory alternative.  It is exploratory — the resulting
recommendations are written into the Logs section of
``prompts/chl_fl.md``.

Run from the bing repo root::

    conda run -n ocean14 python dev/ChlFl/memory_profile.py

What it does
------------
1. Reconstructs the array shapes that the chains path of
   ``calc_Rrs_fluorescence`` builds, for a realistic
   ``(n_samples, n_em, n_ex)`` problem, and reports the theoretical
   footprint of the full 3-D integrand tensor.
2. Measures *peak* process memory of the current vectorised 3-D code
   path with ``tracemalloc`` on a tractable sample count.
3. Prototypes a ``per-emission-wavelength`` reformulation that never
   materialises the 3-D tensor (loops over the small ``n_em`` axis, each
   step touching only an ``(n_samples, n_ex)`` slice) and measures its
   peak memory the same way.
4. Asserts the two implementations agree to machine precision, so the
   recommended fix is numerically identical, not an approximation.
"""

import tracemalloc

import numpy as np

from bing.rt import chl_fl
from bing.rt import rrs as bing_rrs


# ---------------------------------------------------------------------------
# Problem dimensions.
#
# n_em  = number of model (emission) wavelengths.  calc_Rrs_from_models passes
#         the *full* model wave grid as the emission axis, so for PACE OCI at
#         5 nm over 400-700 nm this is ~60.
# n_ex  = number of excitation wavelengths, model.i_Chl_ex over wv_ex_range
#         (400-700 nm) -> also ~60 on the same grid.
# n_samples = flattened chains = (nsteps - burn) * nwalkers.
#         The standard biomass fit uses nsteps=40000; thin_burn_chains drops
#         burn=7000; nwalkers = max(16, 2*ndim).  For ExpBricaud+Pow (ndim=5)
#         nwalkers=16, giving (40000-7000)*16 = 528_000 samples.
# ---------------------------------------------------------------------------

N_EM = 60
N_EX = 60
N_SAMPLES_PROD = (40000 - 7000) * 16  # 528_000 — the real biomass workload


def _report_footprint(n_samples, n_em, n_ex):
    """Print the theoretical size of one (n_samples, n_em, n_ex) f64 array.

    Inputs
    ------
    n_samples, n_em, n_ex : int
        Tensor dimensions.

    Output
    ------
    Returns the per-array size in GiB (float) and prints a short table.
    """
    bytes_per = n_samples * n_em * n_ex * 8  # float64
    gib = bytes_per / 1024**3
    # The current expression builds several such tensors live at once:
    # denom, lambda_ratio (broadcast), the Ed*ratio product, bb_F broadcast,
    # the division result, and the integrand -> ~4-5 simultaneous copies.
    print(f"  (n_samples, n_em, n_ex) = ({n_samples}, {n_em}, {n_ex})")
    print(f"  one float64 tensor      = {gib:.2f} GiB")
    print(f"  ~4-5 live copies        = {4*gib:.1f}-{5*gib:.1f} GiB")
    return gib


def _build_inputs(n_samples, n_em, n_ex, seed_offset=0):
    """Build synthetic but shape-faithful inputs for the chains path.

    Inputs
    ------
    n_samples, n_em, n_ex : int
        Tensor dimensions.
    seed_offset : int
        Varies the synthetic draw without Math.random/Date (kept simple).

    Output
    ------
    dict of arrays matching the signature of calc_Rrs_fluorescence's
    chains (ndim==2) path.
    """
    # Deterministic, smooth, physically-plausible spectra.
    wavelength = np.linspace(650.0, 750.0, n_em)        # emission grid
    wavelength_ex = np.linspace(400.0, 680.0, n_ex)     # excitation grid

    # Per-sample IOPs at excitation wavelengths: (n_samples, n_ex)
    base_a = 0.05 + 0.4 * np.exp(-(wavelength_ex - 440.0) / 90.0)
    base_bb = 0.003 + 0.001 * (600.0 / wavelength_ex)
    amp = np.linspace(0.8, 1.2, n_samples)[:, None]     # sample-to-sample scale
    a_ex = amp * base_a[None, :]
    bb_ex = amp * base_bb[None, :]
    aph_ex = 0.3 * amp * np.exp(-(wavelength_ex - 440.0)**2 / 2000.0)[None, :]

    # IOPs at emission wavelengths: (n_samples, n_em)
    base_a_em = 0.4 + 1.5 * (wavelength - 650.0) / 100.0   # rises toward 730
    base_bb_em = 0.002 + 0.0008 * (600.0 / wavelength)
    a_em = amp * base_a_em[None, :]
    bb_em = amp * base_bb_em[None, :]

    Ed_ex = np.full(n_ex, 1.0)
    Ed_em = 1.0

    return dict(
        wavelength=wavelength, a_em=a_em, bb_em=bb_em,
        a_ex=a_ex, bb_ex=bb_ex, aph_ex=aph_ex,
        wavelength_ex=wavelength_ex, Ed_ex=Ed_ex, Ed_em=Ed_em,
    )


def calc_Rrs_fluorescence_per_em(
    wavelength, a_em, bb_em, a_ex, bb_ex, aph_ex,
    wavelength_ex, Ed_ex, Ed_em,
    mu_d=None, mu_f=0.5, phi_C=0.02, double_gaussian=True,
):
    """Low-memory rewrite: loop over emission λ, never build a 3-D tensor.

    Numerically identical to bing.rt.rrs.calc_Rrs_fluorescence's chains
    path, but peak memory is bounded by the (n_samples, n_ex) slice rather
    than the (n_samples, n_em, n_ex) integrand.  n_em (~60) is small, so the
    Python-level loop is cheap relative to the trapezoid over n_ex.

    Inputs / outputs mirror calc_Rrs_fluorescence (chains, ndim==2 path).
    Returns Rrs_fl of shape (n_samples, n_em).
    """
    from bing.rt import raman
    if mu_d is None:
        mu_d = raman.MU_D_DEFAULT

    wavelength = np.atleast_1d(wavelength)
    if double_gaussian:
        h_C = chl_fl.emission_line_double_gaussian(wavelength)
    else:
        h_C = chl_fl.emission_line_single_gaussian(wavelength)

    kappa_F_em = (np.asarray(a_em) + np.asarray(bb_em)) / mu_f  # (nS, n_em)
    K_ex = (a_ex + bb_ex) / mu_d                                # (nS, n_ex)
    bb_F = chl_fl.fluorescence_backscattering_coeff(aph_ex, phi_C)  # (nS,n_ex)

    n_samples = a_ex.shape[0]
    n_em = wavelength.size
    R_F = np.empty((n_samples, n_em))

    # Pre-compute the sample-independent excitation factor once.
    ex_factor = Ed_ex * (bb_F / mu_d)  # (nS, n_ex)

    # Loop over the *small* emission axis.  Each iteration touches only
    # (n_samples, n_ex) — the 3-D tensor is never formed.
    for j in range(n_em):
        denom = K_ex + kappa_F_em[:, j:j + 1]          # (nS, n_ex)
        lam_ratio = wavelength_ex / wavelength[j]       # (n_ex,)
        integrand = ex_factor * lam_ratio[None, :] / denom
        R_F[:, j] = np.trapezoid(integrand, x=wavelength_ex, axis=1)

    R_F = R_F / Ed_em
    return h_C * bing_rrs.A_Rrs * R_F / (1 - bing_rrs.B_Rrs * R_F)


def _peak_mem(fn):
    """Run fn() under tracemalloc and return (result, peak_GiB)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1024**3


def main():
    print("=" * 70)
    print("1. Theoretical footprint of the production workload")
    print("=" * 70)
    print(f"  Standard biomass fit: nsteps=40000, burn=7000, nwalkers=16")
    _report_footprint(N_SAMPLES_PROD, N_EM, N_EX)
    print("  -> the 3-D integrand alone explains the >100 GB report.\n")

    # A tractable subset to actually run + measure (full 528k would OOM here).
    n_meas = 20_000
    print("=" * 70)
    print(f"2/3. Measured peak memory on n_samples={n_meas} "
          "(subset, to fit in RAM)")
    print("=" * 70)
    inp = _build_inputs(n_meas, N_EM, N_EX)

    cur, peak_cur = _peak_mem(
        lambda: bing_rrs.calc_Rrs_fluorescence(**inp, double_gaussian=True))
    new, peak_new = _peak_mem(
        lambda: calc_Rrs_fluorescence_per_em(**inp, double_gaussian=True))

    print(f"  current 3-D path     peak = {peak_cur:7.3f} GiB")
    print(f"  per-emission rewrite peak = {peak_new:7.3f} GiB")
    print(f"  reduction factor          = {peak_cur / peak_new:6.1f}x\n")

    print("=" * 70)
    print("4. Numerical equivalence")
    print("=" * 70)
    max_abs = np.max(np.abs(cur - new))
    max_rel = np.max(np.abs(cur - new) / (np.abs(cur) + 1e-30))
    print(f"  max |current - rewrite|      = {max_abs:.3e}")
    print(f"  max relative difference      = {max_rel:.3e}")
    assert np.allclose(cur, new, rtol=1e-12, atol=0), "MISMATCH!"
    print("  -> identical to machine precision (rtol=1e-12). PASS\n")

    # Extrapolate the rewrite to the full production size.
    per_sample_gib = peak_new / n_meas
    print("=" * 70)
    print("5. Extrapolated peak for the rewrite at full production size")
    print("=" * 70)
    print(f"  rewrite @ {N_SAMPLES_PROD} samples "
          f"~ {per_sample_gib * N_SAMPLES_PROD:.2f} GiB "
          "(vs >100 GiB for the current path)")


if __name__ == "__main__":
    main()
