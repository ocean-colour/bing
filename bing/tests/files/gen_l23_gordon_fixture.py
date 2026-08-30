"""
Generate the Gordon-path regression fixture used by
tests/test_evaluate.py (test_calc_Rrs_from_models_matches_gordon_fixture_*).

Pins the literal output of ``bing.evaluate.calc_Rrs_from_models`` -- the
untouched Gordon (1988) forward path -- on parameters derived from a
reference Loisel et al. (2023) spectrum (idx=170, the same index
``test_evaluate.py``'s end-to-end fixtures fit), so that "the Gordon path
is untouched" is a test, not a claim (rob_rt integration, M1 task 4;
``docs/design/rob_rt_design.md`` §6, the ``RT_correction`` deletion).

Captured 2026-08-30 (branch ``rob_rt``) with the deprecated
``RT_correction`` block still present in ``calc_Rrs_from_models`` but
inert: no rt_dict construction in ``bing/`` ever set the key
(``rt_dict_from_p`` never emitted it), so this snapshot is exactly "what
the function outputs today, dead code and all" and must keep matching
bit-for-bit after the block is deleted.

Three Gordon-path configurations are pinned, bracketing the deleted block
on both sides of the function body:

* elastic, constant Gordon coefficients (1-D params and a 5-sample batch,
  via ``full_return=True`` -> Rrs, a, bb);
* elastic + Raman correction (batch) -- the code *before* the block;
* elastic + chlorophyll fluorescence (batch) -- the code *after* it.

Generation needs the L23 store (``$OS_COLOR`` data) to derive realistic
parameters; the *test* needs only the .npz. Everything required to
re-evaluate is stored (wavelength grid, Chl, the exact parameter arrays,
RT flags), and the models are rebuilt from the public API alone
(``model_utils.init`` + ``set_aph`` [+ ``set_raman_Ed`` /
``init_Chl_fluorescence``]), with no L23 access at test time.

Run once (ocean14) on a machine with the L23 store:

    python gen_l23_gordon_fixture.py

Writes l23_gordon_fixture.npz next to this script (committed).
"""

import os

import numpy as np

from correct_atmosphere import downwelling

from bing import evaluate
from bing.fitting import l23 as fit_l23
from bing.models import utils as model_utils
from bing.parameters import standard

# Reference L23 spectrum and inelastic settings pinned by the fixture.
L23_IDX = 170
PHI_C = 0.02
DOUBLE_GAUSSIAN = True
NBATCH = 5


def build_models(wave, Chl):
    """Rebuild the ExpBricaud + Pow model pair exactly as the test will.

    Only public API calls that need no external data: this is the recipe
    the regression test replays, so generation and test evaluate an
    identical model state.
    """
    models = model_utils.init(['ExpBricaud', 'Pow'], wave)
    models[0].set_aph(np.array([Chl]))
    return models


def main():
    """Derive L23-realistic inputs, evaluate, and write the fixture."""
    # --- L23-derived reference inputs (needs the L23 store) ---
    p = standard.expb_pow(satellite='PACE', add_noise=False,
                          variable_Gordon=False, include_Raman=False,
                          include_Chl_fl=False)
    prep = fit_l23.prep_one_l23(p, idx=L23_IDX)

    wave = np.asarray(prep['models'][0].wave, dtype=np.float64)
    Chl = float(np.squeeze(prep['odict']['Chl']))
    p0 = np.asarray(prep['p0'], dtype=np.float64)
    nparam_a = prep['models'][0].nparam

    a_params_1d = p0[:nparam_a]
    bb_params_1d = p0[nparam_a:]

    # Deterministic small batch around the L23 initial guess. The realized
    # arrays are stored in the fixture, so RNG reproducibility never
    # matters at test time. +/-2 percent keeps every sample physical
    # (Sdg > 0, beta > 0).
    rng = np.random.default_rng(20260830)
    a_params_batch = a_params_1d * (
        1.0 + 0.02 * rng.standard_normal((NBATCH, a_params_1d.size)))
    bb_params_batch = bb_params_1d * (
        1.0 + 0.02 * rng.standard_normal((NBATCH, bb_params_1d.size)))

    out = dict(l23_idx=np.int32(L23_IDX), wave=wave, Chl=np.float64(Chl),
               phi_C=np.float64(PHI_C),
               double_gaussian=np.bool_(DOUBLE_GAUSSIAN),
               a_params_1d=a_params_1d, bb_params_1d=bb_params_1d,
               a_params_batch=a_params_batch, bb_params_batch=bb_params_batch)

    # --- Config 1: elastic, constant Gordon (1-D and batch, full_return) ---
    models = build_models(wave, Chl)
    rt_elastic = {'variable_Gordon': False, 'include_Raman': False,
                  'include_Chl_fl': False}
    (out['Rrs_elastic_1d'], out['a_elastic_1d'],
     out['bb_elastic_1d']) = evaluate.calc_Rrs_from_models(
        models[0], a_params_1d, models[1], bb_params_1d,
        rt_elastic, full_return=True)
    (out['Rrs_elastic_batch'], out['a_elastic_batch'],
     out['bb_elastic_batch']) = evaluate.calc_Rrs_from_models(
        models[0], a_params_batch, models[1], bb_params_batch,
        rt_elastic, full_return=True)

    # --- Config 2: + Raman (batch) ---
    models_r = build_models(wave, Chl)
    wv_Ed = np.arange(np.floor(models_r[0].wave_ex.min()) - 5.,
                      wave.max() + 5.1, 1.)
    Ed_full = downwelling.downwelling_irradiance(wv_Ed, 0.)
    models_r[0].set_raman_Ed(wv_Ed, Ed_full)
    rt_raman = {'variable_Gordon': False, 'include_Raman': True,
                'include_Chl_fl': False}
    out['Rrs_raman_batch'] = evaluate.calc_Rrs_from_models(
        models_r[0], a_params_batch, models_r[1], bb_params_batch, rt_raman)

    # --- Config 3: + chlorophyll fluorescence (batch) ---
    models_f = build_models(wave, Chl)
    Ed = downwelling.downwelling_irradiance(wave, 0.)
    models_f[0].init_Chl_fluorescence(Ed=Ed)
    rt_fl = {'variable_Gordon': False, 'include_Raman': False,
             'include_Chl_fl': True, 'phi_C': PHI_C,
             'double_gaussian': DOUBLE_GAUSSIAN}
    out['Rrs_chlfl_batch'] = evaluate.calc_Rrs_from_models(
        models_f[0], a_params_batch, models_f[1], bb_params_batch, rt_fl)

    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'l23_gordon_fixture.npz')
    np.savez_compressed(fname, **out)
    print(f'Wrote {fname} ({os.path.getsize(fname)/1024:.0f} kB, '
          f'L23 idx={L23_IDX}, nwave={wave.size}, nbatch={NBATCH})')


if __name__ == '__main__':
    main()
