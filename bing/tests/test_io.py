"""Runnable demonstration of bing.io.save_fit / load_fit.

This is intentionally written as a *script* (not a pytest test yet — that
comes in a later development step).  It runs a short MCMC fit on a single
Loisel et al. (2023) spectrum, saves the result via :func:`bing.io.save_fit`,
loads it back with :func:`bing.io.load_fit`, prints a short summary, and
deletes the temporary files.

Run from the repo root (or anywhere) with the ``ocean14`` conda env active::

    python bing/tests/test_io.py
"""

import os
import tempfile

import numpy as np

from bing import io as bing_io
from bing.fitting import l23 as fit_l23
from bing.parameters import standard


def main():
    # ----- Build a small, fast MCMC fit --------------------------------
    # nsteps must be > evaluate.thin_burn_chains' default burn of 7000,
    # otherwise reconstruct_from_chains / calc_stats see an empty array.
    p = standard.expb_pow(satellite="PACE", add_noise=False,
                          nsteps=8000, nburn=200, variable_Gordon=False)
    idx = 170

    # fit_one returns (chains, models, prep_dict, idx, extras)
    chains, models, prep_dict, idx, extras = fit_l23.fit_one(p, idx)

    # ----- Save and reload via bing.io ---------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        outroot = os.path.join(tmpdir, "test_io")

        npz_path, json_path = bing_io.save_fit(
            outroot, p, models, chains,
            p0=prep_dict["p0"],
            Rrs=prep_dict["model_Rrs"],
            varRrs=prep_dict["model_varRrs"],
        )
        print(f"Saved:\n  {npz_path}\n  {json_path}")

        # Confirm both files exist before reloading.
        assert os.path.exists(npz_path), npz_path
        assert os.path.exists(json_path), json_path

        loaded = bing_io.load_fit(outroot)

        # ----- Quick summary so the round-trip is visible ----------------
        print("\n=== Loaded fit summary ===")
        print(f"BING version    : {loaded['bing_version']}")
        print(f"Model names     : {loaded['model_names']}")
        print(f"Parameter names : {loaded['pnames']}")
        print(f"Wavelengths     : {loaded['wave'].shape}, "
              f"{loaded['wave'].min():.1f}–{loaded['wave'].max():.1f} nm")
        print(f"Chains shape    : {loaded['chains'].shape}")
        print(f"Rrs shape       : {loaded['Rrs'].shape}")
        print(f"a_recon shape   : {loaded['a'].shape}")
        print(f"Stats medians   : {np.array(loaded['stats']['med'])}")
        print(f"Percentiles     : stats={loaded['stats_perc']} "
              f"recon={loaded['recon_perc']}")
        print(f"rt_dict         : {loaded['rt_dict']}")
        print(f"Reconstructed p : {loaded['p'].model_names}")

        # Sanity check: arrays round-trip exactly.
        np.testing.assert_array_equal(loaded["chains"], chains)
        np.testing.assert_array_equal(loaded["wave"], models[0].wave)
        print("\nRound-trip OK — chains and wavelengths match.")


if __name__ == "__main__":
    main()
