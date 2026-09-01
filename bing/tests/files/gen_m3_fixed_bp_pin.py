"""
Generate the **provisional** fixed-B_p fitter-level regression pin used by
tests/test_evaluate_robust.py (test_fit_Bp_false_matches_provisional_pin_*).

M3 Gate item 6 reads "``fit_Bp=False`` results identical to an M2-pinned
value" -- but M2 never created a fitter-level pin (M2 Q5, still OPEN: the
only pre-existing pins are M1's *forward-model* fixture and M2's
tuple-arg-identity tests).  Per the fallback the prompt-4 Context section
lays out, this fixture is M3 task 3's own stand-in: the fixed-``B_p``
(``fit_Bp`` False/absent) fitter outputs captured **live under
task-3-complete code**, whose dispatch behavior was verified correct at
capture time (the M3 task 1 byte-identity tests --
``test_fit_Bp_false_or_absent_byte_identical_to_m2`` -- were green, so
these values *are* the M2 dispatch values; they were just never frozen to
disk before).  It is explicitly **provisional**: if/when M2 Q5 is resolved
with a true pre-M3 pin, that pin supersedes this one.

Pinned, for both the 'gordon' and 'robust_ztt' backends on the cheap
synthetic ExpBricaud+Pow recipe test_evaluate_robust.py already uses
(``_THREAD_TRUTH``/``_THREAD_P0``, 61-band 400-700 nm grid, noiseless
observation, 2% assumed error):

* ``chisq_fit.fit`` best-fit parameters (curve_fit, deterministic);
* ``inference.fit_one`` chains (nsteps=30, nburn=10, nwalkers=16),
  seeded with ``np.random.seed(20260831)`` immediately before the fit --
  emcee draws from the global legacy ``np.random`` state, so the chain is
  reproducible.

Portability caveat: MCMC chains are bit-reproducible on one machine/env
but can drift across BLAS/emcee versions (floating-point noise amplifies
step to step).  The test compares with a small rtol; if the pin ever
fails on a *new environment* while the M3 task-1 byte-identity tests stay
green, regenerate here rather than suspecting a dispatch regression.

Run once (ocean14):

    python gen_m3_fixed_bp_pin.py

Writes m3_fixed_bp_pin.npz next to this script (committed).

Regenerated 2026-09-01 (PR #27): chisq_fit.fit now widens the numerical-
Jacobian step for robust backends (float32 -- scipy's ~1.5e-8 default
produced an exactly-zero Jacobian, so the pinned ``chisq_robust_ztt`` had
frozen an optimizer that never moved off p0). Only ``chisq_robust_ztt``
changed in the regeneration -- ``chisq_gordon``/both MCMC chains were
verified byte-identical to the previous pin, so the Gordon path and the
dispatch behavior this fixture exists to freeze are untouched.
"""

import os

import numpy as np

from bing.models import utils as model_utils
from bing.parameters import standard
from bing.fitting import inference as bing_inf
from bing.fitting import chisq_fit
from bing.rt import defs as rt_defs
from bing.rt.geometry import ObsGeometry

# Same recipe as test_evaluate_robust.py's threading_setup fixture.
TRUTH = np.array([-1.0, 0.015, -0.7, -2.0, 1.0])
P0 = np.array([-0.9, 0.014, -0.6, -1.9, 0.9])
SEED = 20260831
NSTEPS, NBURN = 30, 10


def build_setup():
    """The threading_setup recipe: models, rt_dict, a noiseless Gordon
    observation and its 2% variance."""
    p = standard.expb_pow(wv_min=400., wv_max=700., variable_Gordon=False)
    wave = np.arange(400., 705., 5.)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    models[0].set_aph(np.array([1.0]))
    rt_dict = rt_defs.rt_dict_from_p(p)   # gordon, fit_Bp False, Bp 0.01
    Rrs = chisq_fit.fit_func(wave, *TRUTH, models=models, rt_dict=rt_dict)
    return models, rt_dict, Rrs, (0.02 * Rrs)**2


def main():
    models, rt_dict, Rrs, varRrs = build_setup()
    geom = ObsGeometry(theta_s=30.)
    payload = dict(truth=TRUTH, p0=P0, Rrs=Rrs, varRrs=varRrs,
                   seed=SEED, nsteps=NSTEPS, nburn=NBURN)

    for backend in ('gordon', 'robust_ztt'):
        rt = dict(rt_dict, rt_backend=backend)
        items4 = (Rrs, varRrs, P0.copy(), 0)
        items5 = (Rrs, varRrs, P0.copy(), 0, geom)
        items = items4 if backend == 'gordon' else items5

        # chi-squared best fit (fit_Bp False by rt_dict_from_p default)
        ans, _, _ = chisq_fit.fit(items, models, rt)
        payload[f'chisq_{backend}'] = ans

        # seeded MCMC chain
        pdict = bing_inf.init_mcmc(models, nsteps=NSTEPS, nburn=NBURN,
                                   rt_dict=rt)
        pdict['Chl'] = np.array([1.0])
        pdict['Y'] = None
        np.random.seed(SEED)
        chains, _ = bing_inf.fit_one(items, models=models, pdict=pdict,
                                     chains_only=True, rt_dict=rt)
        payload[f'chain_{backend}'] = chains

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'm3_fixed_bp_pin.npz')
    np.savez(out, **payload)
    print(f'Wrote {out}')
    for backend in ('gordon', 'robust_ztt'):
        print(backend, 'chisq:', payload[f'chisq_{backend}'])
        print(backend, 'chain shape:', payload[f'chain_{backend}'].shape)


if __name__ == '__main__':
    main()
