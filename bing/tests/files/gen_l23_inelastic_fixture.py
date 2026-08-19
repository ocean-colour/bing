"""
Generate the L23 inelastic-RT fixture used by tests/test_l23_inelastic.py.

Extracts a small subset of scenes from the Loisel et al. (2023) HydroLight
database (solar zenith 30 deg): the paired inelastic scenarios

- X1 : elastic only
- X2 : + Raman scattering by water
- X4 : + Raman and chlorophyll-a fluorescence (phi_C = 0.02)

The scenario differences provide exact truth for BING's Raman correction
factor (Rrs_X2 / Rrs_X1) and additive fluorescence term (Rrs_X4 - Rrs_X2).
The input IOPs are identical across scenarios; Ed(0+) is a sky property,
identical across scenes.

Run once (ocean14) on a machine with the L23 store:

    python gen_l23_inelastic_fixture.py

Writes l23_inelastic_fixture.npz next to this script (~150 kB; committed).
"""

import os

import numpy as np
import xarray as xr

L23_PATH = os.path.join(
    os.environ.get('OS_COLOR_DATA', '/mnt/tank/Oceanography/data/Color'),
    'Loisel2023')

ZEN = '30'          # solar zenith [deg]
N_SCENES = 40       # subset size (evenly strided over the 3320 scenes)


def main():
    out = {}
    idx = None
    for x in (1, 2, 4):
        ds = xr.open_dataset(
            os.path.join(L23_PATH, f'Hydrolight{x}{ZEN}.nc'))
        if idx is None:
            n = ds['Rrs'].shape[0]
            idx = np.arange(0, n, n // N_SCENES)[:N_SCENES]
            out['wave'] = ds['Lambda'].values.astype(np.float32)
            out['idx'] = idx.astype(np.int32)
            out['zenith'] = np.float32(float(ZEN))
            # Input IOPs are identical across X scenarios; store once.
            for key in ('a', 'bb', 'aph'):
                out[key] = ds[key].values[idx].astype(np.float32)
            # Ed(0+) is scene-independent (sky property); store the mean.
            Ed = ds['Ed_0+'].values
            assert (Ed.std(axis=0) / Ed.mean(axis=0)).max() < 1e-3
            out['Ed'] = Ed.mean(axis=0).astype(np.float32)
        else:
            # Verify the IOPs really are shared across scenarios.
            assert np.allclose(ds['a'].values[idx], out['a'], rtol=1e-4)
        out[f'Rrs{x}'] = ds['Rrs'].values[idx].astype(np.float32)
        ds.close()

    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'l23_inelastic_fixture.npz')
    np.savez_compressed(fname, **out)
    print(f'Wrote {fname} '
          f'({os.path.getsize(fname)/1024:.0f} kB, {N_SCENES} scenes)')


if __name__ == '__main__':
    main()
