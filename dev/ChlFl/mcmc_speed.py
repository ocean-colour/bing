"""Investigation: why the per-emission loop slowed MCMC fitting ~7x.

Goal
----
Modifications item 6 in ``prompts/chl_fl.md``: the memory fix (looping over
the emission axis instead of building a 3-D tensor) cut RAM from >100 GB to
~2 GiB, but MCMC fitting is now ~7x slower.

Diagnosis (this module verifies it)
-----------------------------------
``inference.log_prob`` evaluates a *single* parameter vector per call, but
``models.eval_a`` returns shape ``(1, nwave)``.  So the excitation IOPs reach
``calc_Rrs_fluorescence`` as 2-D ``(1, n_ex)`` and take the *chains* branch
with ``n_samples == 1``.  The old chains branch was one vectorised 3-D op
over a tiny ``(1, n_em, n_ex)`` tensor — microseconds.  The new branch runs a
Python loop of ``n_em`` (~60) iterations, each allocating arrays and calling
``np.trapezoid``.  MCMC calls log_prob ~nsteps*nwalkers times (40000*16 ≈
6.4e5), so the per-call Python overhead dominates and the whole fit slows ~7x.

The memory blow-up only ever mattered for *large* ``n_samples``
(reconstruct_from_chains).  For ``n_samples == 1`` the 3-D tensor is trivially
small, so the loop buys nothing and costs a lot.

Proposed fix (benchmarked here)
-------------------------------
Chunk the chains branch over samples: process blocks sized so the 3-D tensor
stays within a memory budget, each block done with the fast vectorised 3-D op.
For ``n_samples == 1`` that is a single vectorised block (fast, like the old
code); for 528k samples it is many bounded-memory blocks (low RAM, like the
loop).  Best of both.

Run from the bing repo root::

    conda run -n ocean14 python dev/ChlFl/mcmc_speed.py
"""

import time

import numpy as np

from bing.rt import chl_fl
from bing.rt import rrs as bing_rrs


N_EM = 60
N_EX = 60


def _single_spectrum_inputs():
    """Build (1, n_ex)/(1, n_em) inputs mimicking one log_prob call.

    Output
    ------
    dict of arrays splattable into calc_Rrs_fluorescence, with n_samples=1.
    """
    wavelength = np.linspace(650.0, 750.0, N_EM)
    wavelength_ex = np.linspace(400.0, 680.0, N_EX)
    a_ex = (0.05 + 0.4 * np.exp(-(wavelength_ex - 440.0) / 90.0))[None, :]
    bb_ex = (0.003 + 0.001 * (600.0 / wavelength_ex))[None, :]
    aph_ex = (0.3 * np.exp(-(wavelength_ex - 440.0)**2 / 2000.0))[None, :]
    a_em = (0.4 + 1.5 * (wavelength - 650.0) / 100.0)[None, :]
    bb_em = (0.002 + 0.0008 * (600.0 / wavelength))[None, :]
    return dict(
        wavelength=wavelength, a_em=a_em, bb_em=bb_em,
        a_ex=a_ex, bb_ex=bb_ex, aph_ex=aph_ex,
        wavelength_ex=wavelength_ex, Ed_ex=np.full(N_EX, 1.0), Ed_em=1.0,
    )


def _loop_impl(inp, mu_d=0.9, mu_f=0.5, phi_C=0.02):
    """Current (post-memory-fix) per-emission loop, standalone copy."""
    wavelength = inp['wavelength']
    h_C = chl_fl.emission_line_double_gaussian(wavelength)
    kappa_F_em = (inp['a_em'] + inp['bb_em']) / mu_f
    K_ex = (inp['a_ex'] + inp['bb_ex']) / mu_d
    bb_F = chl_fl.fluorescence_backscattering_coeff(inp['aph_ex'], phi_C)
    n_samples, n_em = K_ex.shape[0], wavelength.size
    R_F = np.empty((n_samples, n_em))
    ex_factor = inp['Ed_ex'][None, :] * (bb_F / mu_d)
    for j in range(n_em):
        denom = K_ex + kappa_F_em[:, j:j + 1]
        lam_ratio = inp['wavelength_ex'] / wavelength[j]
        integrand = ex_factor * lam_ratio[None, :] / denom
        R_F[:, j] = np.trapezoid(integrand, x=inp['wavelength_ex'], axis=1)
    R_F = R_F / inp['Ed_em']
    return h_C * bing_rrs.A_Rrs * R_F / (1 - bing_rrs.B_Rrs * R_F)


def _vec3d_impl(inp, mu_d=0.9, mu_f=0.5, phi_C=0.02):
    """Old fully-vectorised 3-D op (fast for small n_samples), standalone."""
    wavelength = inp['wavelength']
    h_C = chl_fl.emission_line_double_gaussian(wavelength)
    kappa_F_em = (inp['a_em'] + inp['bb_em']) / mu_f
    K_ex = (inp['a_ex'] + inp['bb_ex']) / mu_d
    bb_F = chl_fl.fluorescence_backscattering_coeff(inp['aph_ex'], phi_C)
    denom = K_ex[:, None, :] + kappa_F_em[:, :, None]
    lam_ratio = inp['wavelength_ex'][None, None, :] / wavelength[None, :, None]
    integrand = (inp['Ed_ex'][None, None, :] * lam_ratio
                 * (bb_F[:, None, :] / mu_d) / denom)
    R_F = np.trapezoid(integrand, x=inp['wavelength_ex'], axis=2)
    R_F = R_F / inp['Ed_em']
    return h_C * bing_rrs.A_Rrs * R_F / (1 - bing_rrs.B_Rrs * R_F)


def _time(fn, n_iter):
    """Return mean seconds/call over n_iter calls (no Date/perf needed)."""
    t0 = time.process_time()
    for _ in range(n_iter):
        fn()
    return (time.process_time() - t0) / n_iter


def main():
    inp = _single_spectrum_inputs()
    n_iter = 5000

    # Warm up
    _loop_impl(inp); _vec3d_impl(inp)

    # Production function (now chunked) on the same single-spectrum inputs.
    bing_rrs.calc_Rrs_fluorescence(**inp, double_gaussian=True)  # warm up
    t_prod = _time(
        lambda: bing_rrs.calc_Rrs_fluorescence(**inp, double_gaussian=True),
        n_iter)

    t_loop = _time(lambda: _loop_impl(inp), n_iter)
    t_vec = _time(lambda: _vec3d_impl(inp), n_iter)

    print("=" * 66)
    print(f"Single-spectrum (n_samples=1) timing, mean of {n_iter} calls")
    print("=" * 66)
    print(f"  per-emission loop (regressed): {t_loop*1e6:8.1f} us/call")
    print(f"  vectorised 3-D (old)         : {t_vec*1e6:8.1f} us/call")
    print(f"  production (chunked, fixed)  : {t_prod*1e6:8.1f} us/call")
    print(f"  loop / vec slowdown          : {t_loop/t_vec:8.1f}x")
    print(f"  production / vec overhead    : {t_prod/t_vec:8.1f}x")
    print()
    print("  -> log_prob hot path: the chunked fix restores ~vectorised speed")
    print("     (single block for n_samples=1), removing the MCMC slowdown.")

    # Equivalence sanity check
    got = bing_rrs.calc_Rrs_fluorescence(**inp, double_gaussian=True)
    assert np.allclose(_loop_impl(inp), _vec3d_impl(inp), rtol=1e-12)
    assert np.allclose(got, _vec3d_impl(inp), rtol=1e-12)
    print("\n  (loop, vec, and production all agree to rtol=1e-12)")


if __name__ == "__main__":
    main()
