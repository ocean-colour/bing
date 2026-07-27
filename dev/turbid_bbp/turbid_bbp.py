"""Benchmark: do the two-component turbid backscattering models help?

Goal
----
Three questions, in order of how much they matter:

1. **Does a second backscattering component actually buy anything?**
   Fit a synthetic spectrum whose true ``bb_nw`` is a genuine *sum* of a
   steep organic term and a near-flat mineral term, with noise, using
   ``Pow``, ``PowFlex``, ``Pow2Flat`` and ``Pow2``.
2. **Is the 4-parameter model identifiable?** Four backscattering plus
   three absorption parameters against tightly-quoted in-situ noise may
   be degenerate. Compared here: chi-squared with the tight noise, MCMC
   with the tight noise, MCMC with an inflated noise floor, and the
   3-parameter ``Pow2Flat`` (fixed ``eta_min``).
3. **Does it cost anything on open ocean water?** Fit a handful of L23
   spectra with ``expb_pow`` vs the two-component models and compare
   reduced chi-squared and IOP accuracy against L23 truth.

The truth is *deliberately balanced* -- the organic term dominates the
blue, the mineral term the red -- so the log-log slope of ``bb_nw``
changes across the band. A truth dominated by one term is effectively a
single power law and would make ``PowFlex`` look sufficient (see the
Prompt-3 log in ``prompts/turbid_waters.md``).

Run from the bing repo root::

    python dev/turbid_bbp/turbid_bbp.py

Four figures are written next to this file:

- ``turbid_bbnw_recovery.png``  -- bb_nw(lambda), truth vs each model
- ``turbid_rrs_panels.png``     -- Rrs + residuals, turbid and clear
- ``turbid_identifiability.png``-- parameter precision and correlations
- ``turbid_l23_regression.png`` -- L23 chi2_nu and IOP accuracy

Per Q10 in the prompt doc, the MCMC runs here use nwalkers=48 locally;
``init_mcmc``'s default (max(16, 2*ndim) = 16 for 7 parameters) is left
alone.
"""
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bing.models import utils as model_utils
from bing.parameters import standard
from bing.fitting import chisq_fit
from bing.fitting import inference as bing_inf
from bing.rt import defs as rt_defs

# --- configuration -----------------------------------------------------
WAVE = np.arange(400., 755., 5.)        # turbid work extends to 750 nm

# Adg, Sdg, Aph | Bmin, eta_min, Borg, eta_org
TRUTH = np.array([-0.3, 0.015, -0.4,
                  np.log10(0.05), 0.0, np.log10(0.10), 1.8])

# GLORIA-like measured error: ~1.5e-4 sr^-1, i.e. well under 1% of a
# turbid Rrs.  This tightness is what makes identifiability a question.
SIG_MEASURED = 1.5e-4
INFLATE_FRAC = 0.05                     # the approved error floor

NREAL = 24                              # noise realisations
NWALKERS = 48                           # Q10: local bump, not in init_mcmc
NSTEPS, NBURN = 6000, 1000

# L23 has 3320 spectra
L23_IDX = [170, 500, 1000, 1500, 2000, 2500, 3000, 3200]

# How strongly two-component must the truth be before one component is
# statistically rejected?  Sweep the mineral slope across its prior.
ETA_MIN_SWEEP = [-0.5, -0.25, 0.0, 0.25, 0.5]
NREAL_SWEEP = 8

MODELS = [('Pow', 'expb_pow'), ('PowFlex', 'expb_powflex'),
          ('Pow2Flat', 'expb_pow2flat'), ('Pow2', 'expb_pow2')]
CLR = {'Pow': '#0072B2', 'PowFlex': '#E69F00', 'Pow2Flat': '#CC79A7',
       'Pow2': '#009E73', 'truth': 'k'}

# Starting guesses, per bb model (a-model part is shared)
P0 = {'Pow': np.array([-.5, .015, -.5, -1., .5]),
      'PowFlex': np.array([-.5, .015, -.5, -1., .5]),
      'Pow2Flat': np.array([-.5, .015, -.5, -1., -2., 1.]),
      'Pow2': np.array([-.5, .015, -.5, -1., .05, -2., 1.])}


def build(combo, wave=WAVE, **kwargs):
    """Set up one model combination for fitting.

    Args:
        combo (str): Name of a factory in bing.parameters.standard.
        wave (np.ndarray): Wavelengths.
        **kwargs: Overrides passed to the factory.

    Returns:
        tuple: (p, models, rt_dict, (low, high) bounds).
    """
    p = getattr(standard, combo)(wv_min=wave[0], wv_max=wave[-1],
                                 variable_Gordon=False, **kwargs)
    models = model_utils.init(p.model_names, wave, (p.apriors, p.bpriors))
    if models[0].uses_Chl:
        models[0].set_aph(np.array([2.0]))
    rt_dict = rt_defs.rt_dict_from_p(p)
    low = np.array([d['pmin'] for d in p.apriors] +
                   [d['pmin'] for d in p.bpriors], dtype=float)
    high = np.array([d['pmax'] for d in p.apriors] +
                    [d['pmax'] for d in p.bpriors], dtype=float)
    return p, models, rt_dict, (low, high)


def synth_truth():
    """The noiseless turbid synthetic spectrum and its true IOPs.

    Returns:
        dict: wave, Rrs, bb_nw and the models used to generate them.
    """
    _, models, rt_dict, _ = build('expb_pow2')
    Rrs = chisq_fit.fit_func(WAVE, *TRUTH, models=models,
                             rt_dict=rt_dict)
    return dict(wave=WAVE, Rrs=Rrs, models=models, rt_dict=rt_dict,
                bb_nw=models[1].eval_bbnw(TRUTH[3:])[0])


# ---------------------------------------------------------------------
# 1. Synthetic recovery
# ---------------------------------------------------------------------

def run_recovery(scene, rng):
    """Fit noisy realisations of the turbid spectrum with every model.

    Args:
        scene (dict): Output of synth_truth().
        rng (np.random.Generator): Source of the noise draws.

    Returns:
        dict: Per-model lists of fitted parameters and misfits.
    """
    out = {label: dict(ans=[], misfit=[], chi2nu=[], bb=[])
           for label, _ in MODELS}
    sigma = np.full(scene['Rrs'].size, SIG_MEASURED)

    for _ in range(NREAL):
        obs = scene['Rrs'] + rng.normal(0., sigma)
        for label, combo in MODELS:
            _, models, rt_dict, bounds = build(combo)
            try:
                ans, _, _ = chisq_fit.fit((obs, sigma**2, P0[label], 0),
                                          models, rt_dict, bounds=bounds,
                                          maxfev=40000)
            except RuntimeError:
                continue
            pred = chisq_fit.fit_func(WAVE, *ans, models=models,
                                      rt_dict=rt_dict)
            chi2 = np.sum((pred - obs)**2/sigma**2)
            out[label]['ans'].append(ans)
            out[label]['misfit'].append(
                np.median(np.abs(pred - scene['Rrs'])/scene['Rrs']))
            out[label]['chi2nu'].append(chi2/(obs.size - ans.size))
            out[label]['bb'].append(models[1].eval_bbnw(ans[3:])[0])
    for label in out:
        for key in out[label]:
            out[label][key] = np.array(out[label][key])
    return out


def report_recovery(rec):
    """Print the recovery table."""
    print('\n' + '='*70)
    print('1. SYNTHETIC RECOVERY  (turbid two-component truth, '
          f'{NREAL} noise realisations,')
    print(f'   sigma = {SIG_MEASURED:.1e} sr^-1 absolute)')
    print('='*70)
    print(f'{"model":10s} {"n_bb":>5s} {"converged":>10s} '
          f'{"median misfit":>14s} {"median chi2_nu":>15s}')
    for label, _ in MODELS:
        r = rec[label]
        n_bb = r['ans'].shape[1] - 3 if len(r['ans']) else 0
        print(f'{label:10s} {n_bb:5d} {len(r["ans"]):7d}/{NREAL:<3d}'
              f'{np.median(r["misfit"]):14.2e} '
              f'{np.median(r["chi2nu"]):15.3f}')
    print('\n   "misfit" is vs the NOISELESS truth, so it measures model'
          ' adequacy,\n   not how well the fit chased the noise.')

    print('\n   Pow2 parameter recovery (median over realisations):')
    pn = ['Adg', 'Sdg', 'Aph', 'Bmin', 'eta_min', 'Borg', 'eta_org']
    ans = rec['Pow2']['ans']
    for kk, nm in enumerate(pn):
        print(f'     {nm:8s} truth {TRUTH[kk]:+8.4f}   '
              f'fit {np.median(ans[:, kk]):+8.4f} '
              f'+/- {np.std(ans[:, kk]):.4f}')


def run_sweep(rng):
    """When is one component statistically rejected?

    Sweeps the mineral slope ``eta_min`` across its prior range, refits
    with ``Pow`` and ``Pow2``, and records reduced chi-squared. A single
    power law is only *rejected* where its chi2_nu rises well above 1.

    Args:
        rng (np.random.Generator): Noise source.

    Returns:
        dict: eta_min -> per-model median chi2_nu and misfit.
    """
    out = {}
    for eta_min in ETA_MIN_SWEEP:
        truth = TRUTH.copy()
        truth[4] = eta_min
        _, gen_models, gen_rt, _ = build('expb_pow2')
        clean = chisq_fit.fit_func(WAVE, *truth, models=gen_models,
                                   rt_dict=gen_rt)
        sigma = np.full(clean.size, SIG_MEASURED)
        rec = {label: dict(chi2nu=[], misfit=[])
               for label in ('Pow', 'Pow2')}
        for _ in range(NREAL_SWEEP):
            obs = clean + rng.normal(0., sigma)
            for label, combo in [('Pow', 'expb_pow'),
                                 ('Pow2', 'expb_pow2')]:
                _, models, rt_dict, bounds = build(combo)
                try:
                    ans, _, _ = chisq_fit.fit(
                        (obs, sigma**2, P0[label], 0), models, rt_dict,
                        bounds=bounds, maxfev=40000)
                except RuntimeError:
                    continue
                pred = chisq_fit.fit_func(WAVE, *ans, models=models,
                                          rt_dict=rt_dict)
                rec[label]['chi2nu'].append(
                    np.sum((pred - obs)**2/sigma**2)/(obs.size-ans.size))
                rec[label]['misfit'].append(
                    np.median(np.abs(pred - clean)/clean))
        # Structural residual: fit Pow to the NOISELESS spectrum.  If the
        # model is wrong by r(lambda) and the noise is sigma, then
        # E[chi2_nu] ~ 1 + mean(r^2)/sigma^2, so the noise level at which
        # the single power law starts to be rejected (chi2_nu ~ 2) is
        # sigma_crit = rms(r).
        _, models, rt_dict, bounds = build('expb_pow')
        try:
            ans, _, _ = chisq_fit.fit((clean, sigma**2, P0['Pow'], 0),
                                      models, rt_dict, bounds=bounds,
                                      maxfev=40000)
            pred = chisq_fit.fit_func(WAVE, *ans, models=models,
                                      rt_dict=rt_dict)
            sig_crit = float(np.sqrt(np.mean((pred - clean)**2)))
        except RuntimeError:
            sig_crit = np.nan

        out[eta_min] = {lb: {k: float(np.median(v)) if len(v) else np.nan
                             for k, v in rec[lb].items()} for lb in rec}
        out[eta_min]['sig_crit'] = sig_crit
    return out


def report_sweep(sweep):
    """Print the "when do you need two components" table."""
    print('\n' + '='*70)
    print('1b. WHEN IS ONE COMPONENT REJECTED?  (mineral slope sweep,')
    print(f'    {NREAL_SWEEP} realisations each, sigma = '
          f'{SIG_MEASURED:.1e} sr^-1)')
    print('='*70)
    print(f'{"eta_min":>8s} {"Pow chi2_nu":>12s} {"Pow2 chi2_nu":>13s} '
          f'{"sigma_crit":>11s} {"vs measured":>12s} {"verdict":>14s}')
    for eta_min, r in sweep.items():
        pow_chi2 = r['Pow']['chi2nu']
        verdict = 'Pow rejected' if pow_chi2 > 2. else 'Pow adequate'
        ratio = SIG_MEASURED/r['sig_crit'] if r['sig_crit'] else np.nan
        print(f'{eta_min:8.2f} {pow_chi2:12.2f} '
              f'{r["Pow2"]["chi2nu"]:13.2f} {r["sig_crit"]:11.2e} '
              f'{ratio:11.1f}x {verdict:>14s}')
    print('\n   chi2_nu ~ 1 means the single power law is statistically')
    print('   indistinguishable from the two-component truth AT THIS')
    print('   NOISE LEVEL -- the extra parameters are only justified')
    print('   where it climbs well above 1.')
    print('\n   sigma_crit is the noise level at which a single power law')
    print('   WOULD start to be rejected (rms of its structural residual')
    print('   against the noiseless truth).  "vs measured" is how many')
    print('   times tighter than the GLORIA-like error that would need')
    print('   to be.')


# ---------------------------------------------------------------------
# 2. Identifiability
# ---------------------------------------------------------------------

def fit_mcmc(combo, obs, varRrs, p0, label):
    """Run MCMC on one spectrum and return the flattened chains.

    Args:
        combo (str): standard.py factory name.
        obs (np.ndarray): Observed Rrs.
        varRrs (np.ndarray): Variance on Rrs.
        p0 (np.ndarray): Starting parameters.
        label (str): For progress output.

    Returns:
        tuple: (flat chains, parameter names, acceptance fraction).
    """
    _, models, rt_dict, _ = build(combo)
    print(f'   ... MCMC {label} ({NWALKERS} walkers, {NSTEPS} steps)')
    np.random.seed(1234)
    sampler = bing_inf.run_emcee(models, obs, varRrs, rt_dict,
                                 nwalkers=NWALKERS, nburn=NBURN,
                                 nsteps=NSTEPS, skip_check=True,
                                 p0=p0.copy())
    chains = sampler.get_chain()
    pnames = list(models[0].pnames) + list(models[1].pnames)
    return (chains.reshape(-1, chains.shape[-1]), pnames,
            float(np.mean(sampler.acceptance_fraction)))


def run_identifiability(scene, rng):
    """Compare four ways of constraining the turbid model.

    Args:
        scene (dict): Output of synth_truth().
        rng (np.random.Generator): Noise source.

    Returns:
        dict: One entry per strategy.
    """
    sigma = np.full(scene['Rrs'].size, SIG_MEASURED)
    obs = scene['Rrs'] + rng.normal(0., sigma)
    sig_inflated = np.maximum(sigma, INFLATE_FRAC*np.abs(obs))

    res = {}

    # (a) chi-squared with the tight measured noise: read the degeneracy
    # straight off the covariance matrix
    _, models, rt_dict, bounds = build('expb_pow2')
    ans, cov, _ = chisq_fit.fit((obs, sigma**2, P0['Pow2'], 0), models,
                                rt_dict, bounds=bounds, maxfev=40000)
    sig_par = np.sqrt(np.diag(cov))
    corr = cov/np.outer(sig_par, sig_par)
    res['chisq tight'] = dict(par=ans, sig=sig_par, corr=corr,
                              pnames=list(models[0].pnames) +
                              list(models[1].pnames),
                              cond=float(np.linalg.cond(corr)))

    # (b) MCMC, tight noise
    flat, pnames, acc = fit_mcmc('expb_pow2', obs, sigma**2, P0['Pow2'],
                                 'Pow2 / tight noise')
    res['MCMC tight'] = dict(par=np.median(flat, axis=0),
                             sig=flat.std(axis=0),
                             corr=np.corrcoef(flat, rowvar=False),
                             pnames=pnames, acc=acc, flat=flat)

    # (c) MCMC with the inflated error floor
    flat, pnames, acc = fit_mcmc('expb_pow2', obs, sig_inflated**2,
                                 P0['Pow2'], 'Pow2 / inflated noise')
    res['MCMC inflated'] = dict(par=np.median(flat, axis=0),
                                sig=flat.std(axis=0),
                                corr=np.corrcoef(flat, rowvar=False),
                                pnames=pnames, acc=acc, flat=flat)

    # (d) MCMC on the 3-parameter model (eta_min fixed at 0)
    flat, pnames, acc = fit_mcmc('expb_pow2flat', obs, sigma**2,
                                 P0['Pow2Flat'], 'Pow2Flat / tight noise')
    res['MCMC Pow2Flat'] = dict(par=np.median(flat, axis=0),
                                sig=flat.std(axis=0),
                                corr=np.corrcoef(flat, rowvar=False),
                                pnames=pnames, acc=acc, flat=flat)
    return res, obs, sigma, sig_inflated


def report_identifiability(res):
    """Print the identifiability comparison."""
    print('\n' + '='*70)
    print('2. IDENTIFIABILITY  (one turbid spectrum, 4 strategies)')
    print('='*70)

    for key, r in res.items():
        print(f'\n   {key}'
              + (f'   [acceptance {r["acc"]:.2f}]' if 'acc' in r else ''))
        for kk, nm in enumerate(r['pnames']):
            print(f'     {nm:9s} {r["par"][kk]:+8.4f} '
                  f'+/- {r["sig"][kk]:7.4f}')

    print('\n   Key parameter correlations (the degeneracy to watch):')
    for key, r in res.items():
        pn = r['pnames']
        pairs = [('Bmin', 'Borg'), ('eta_min', 'eta_org'),
                 ('Bmin', 'eta_min'), ('Aph', 'Borg')]
        bits = []
        for a, b in pairs:
            if a in pn and b in pn:
                cc = r['corr'][pn.index(a), pn.index(b)]
                bits.append(f'{a}-{b} {cc:+.2f}')
        print(f'     {key:15s} ' + '   '.join(bits))

    print(f'\n   chi-squared correlation-matrix condition number: '
          f'{res["chisq tight"]["cond"]:.3g}')
    print('   (>~100 means the parameters trade off strongly)')

    # Accuracy AND precision together: a tight posterior in the wrong
    # place is worse than a wide one in the right place.
    print('\n   Recovery of the backscattering parameters '
          '(|bias| in units of the reported sigma):')
    tnames = ['Adg', 'Sdg', 'Aph', 'Bmin', 'eta_min', 'Borg', 'eta_org']
    for key, r in res.items():
        bits = []
        for nm in ['Bmin', 'eta_min', 'Borg', 'eta_org']:
            if nm not in r['pnames']:
                bits.append(f'{nm} fixed')
                continue
            kk = r['pnames'].index(nm)
            truth = TRUTH[tnames.index(nm)]
            bias = abs(r['par'][kk] - truth)
            bits.append(f'{nm} {bias/max(r["sig"][kk], 1e-9):.1f}s')
        print(f'     {key:15s} ' + '   '.join(bits))


# ---------------------------------------------------------------------
# 3. L23 no-regression
# ---------------------------------------------------------------------

def fit_l23_one(fit_l23, p, idx, maxfev=None):
    """Chi-squared fit of one L23 spectrum, mirroring l23.fit_with_LM.

    Re-implemented here only so the evaluation budget can be raised;
    ``fit_with_LM`` does not expose ``maxfev``.

    Args:
        fit_l23 (module): bing.fitting.l23.
        p (namedtuple): BING parameter tuple.
        idx (int): L23 spectrum index.
        maxfev (int, optional): Evaluation budget for curve_fit.

    Returns:
        tuple: (ans, models, prep_dict, rt_dict).
    """
    prep = fit_l23.prep_one_l23(p, idx)
    models = prep['models']
    low = np.array([d['pmin'] for d in p.apriors] +
                   [d['pmin'] for d in p.bpriors], dtype=float)
    high = np.array([d['pmax'] for d in p.apriors] +
                    [d['pmax'] for d in p.bpriors], dtype=float)
    rt_dict = rt_defs.rt_dict_from_p(p)
    items = (prep['model_Rrs'], prep['model_varRrs'], prep['p0'], idx)
    ans, _, _ = chisq_fit.fit(items, models, rt_dict, bounds=(low, high),
                              maxfev=maxfev)
    return ans, models, prep, rt_dict


def run_l23(maxfev=40000):
    """Fit several clear L23 spectra with each model.

    Args:
        maxfev (int, optional): Evaluation budget handed to curve_fit.

    Returns:
        dict: Per-model chi2_nu, IOP errors and convergence record, or
            None when L23 is unavailable.
    """
    try:
        from bing.fitting import l23 as fit_l23
    except Exception as exc:                      # pragma: no cover
        print(f'\n   [L23 unavailable: {exc}]')
        return None

    out = {label: dict(chi2nu={}, a_err={}, bb_err={}, failed=[])
           for label, _ in MODELS}
    for idx in L23_IDX:
        for label, combo in MODELS:
            p = getattr(standard, combo)(satellite='PACE',
                                         add_noise=False,
                                         variable_Gordon=True)
            try:
                ans, models, prep, rt_dict = fit_l23_one(
                    fit_l23, p, idx, maxfev=maxfev)
            except RuntimeError:
                out[label]['failed'].append(idx)
                continue
            pred = chisq_fit.fit_func(models[0].wave, *ans, models=models,
                                      rt_dict=rt_dict)
            Rrs, var = prep['model_Rrs'], prep['model_varRrs']
            chi2 = float(np.sum((pred - Rrs)**2/var))

            # IOP accuracy vs L23 truth, interpolated to the fit grid
            od = prep['odict']
            wave = models[0].wave
            a_true = np.interp(wave, od['true_wave'], od['anw'])
            bb_true = np.interp(wave, od['true_wave'], od['bbnw'])
            na = models[0].nparam
            a_fit = models[0].eval_anw(ans[:na])[0]
            bb_fit = models[1].eval_bbnw(ans[na:])[0]

            out[label]['chi2nu'][idx] = chi2/(Rrs.size - ans.size)
            out[label]['a_err'][idx] = float(
                np.median(np.abs(a_fit - a_true)/a_true))
            out[label]['bb_err'][idx] = float(
                np.median(np.abs(bb_fit - bb_true)/bb_true))
    return out


def report_l23(l23, maxfev):
    """Print the L23 no-regression comparison.

    Only spectra that *every* model fitted are compared, otherwise the
    medians describe different samples and are not comparable.
    """
    print('\n' + '='*70)
    print(f'3. L23 NO-REGRESSION  ({len(L23_IDX)} open-ocean spectra, '
          f'maxfev={maxfev})')
    print('='*70)
    if l23 is None:
        return None

    common = [i for i in L23_IDX
              if all(i in l23[label]['chi2nu'] for label, _ in MODELS)]
    print(f'{"model":10s} {"converged":>10s} {"median chi2_nu":>15s} '
          f'{"median |da|/a":>14s} {"median |dbb|/bb":>16s}')
    for label, _ in MODELS:
        r = l23[label]
        n_ok = len(r['chi2nu'])
        if not common:
            continue
        print(f'{label:10s} {n_ok:7d}/{len(L23_IDX):<3d}'
              f'{np.median([r["chi2nu"][i] for i in common]):15.4f} '
              f'{np.median([r["a_err"][i] for i in common]):14.3f} '
              f'{np.median([r["bb_err"][i] for i in common]):16.3f}')
        if r['failed']:
            print(f'{"":10s} did not converge: {r["failed"]}')
    print(f'\n   Compared on the {len(common)} spectra every model '
          f'fitted: {common}')
    return common


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def fig_recovery(scene, rec, outdir):
    """bb_nw(lambda): truth vs what each model recovers."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axs[0]
    ax.plot(scene['wave'], scene['bb_nw'], 'o', ms=4, color=CLR['truth'],
            label='truth (2 components)')
    for label, _ in MODELS:
        if not len(rec[label]['bb']):
            continue
        med = np.median(rec[label]['bb'], axis=0)
        ax.plot(scene['wave'], med, color=CLR[label], label=label)
    ax.set_yscale('log')
    ax.set_ylabel(r'$b_{b,nw}$ [m$^{-1}$]')
    ax.set_title('Recovered backscattering')

    ax = axs[1]
    for label, _ in MODELS:
        if not len(rec[label]['bb']):
            continue
        med = np.median(rec[label]['bb'], axis=0)
        ax.plot(scene['wave'], 100*(med - scene['bb_nw'])/scene['bb_nw'],
                color=CLR[label], label=label)
    ax.axhline(0., color='k', lw=0.8)
    ax.set_ylabel(r'$b_{b,nw}$ error [%]')
    ax.set_title('One component cannot follow the shape')

    for ax in axs:
        ax.set_xlabel('wavelength [nm]')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir/'turbid_bbnw_recovery.png', dpi=150)
    plt.close(fig)


def fig_rrs(scene, rec, l23, outdir):
    """Rrs and residuals for the turbid synthetic and a clear L23 fit."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 7),
                            gridspec_kw=dict(height_ratios=[2, 1]))

    # -- turbid synthetic
    ax, axr = axs[0, 0], axs[1, 0]
    ax.plot(scene['wave'], scene['Rrs'], 'o', ms=4, color=CLR['truth'],
            label='truth')
    for label, combo in MODELS:
        if not len(rec[label]['ans']):
            continue
        _, models, rt_dict, _ = build(combo)
        ans = np.median(rec[label]['ans'], axis=0)
        pred = chisq_fit.fit_func(scene['wave'], *ans, models=models,
                                  rt_dict=rt_dict)
        ax.plot(scene['wave'], pred, color=CLR[label], label=label)
        axr.plot(scene['wave'],
                 100*(pred - scene['Rrs'])/scene['Rrs'],
                 color=CLR[label])
    ax.set_title(f'Turbid synthetic (Rrs peaks at '
                 f'{scene["wave"][np.argmax(scene["Rrs"])]:.0f} nm)')

    # -- clear L23
    ax2, axr2 = axs[0, 1], axs[1, 1]
    if l23 is not None:
        from bing.fitting import l23 as fit_l23
        idx = L23_IDX[0]
        for label, combo in MODELS:
            p = getattr(standard, combo)(satellite='PACE',
                                         add_noise=False,
                                         variable_Gordon=True)
            ans, _, models, prep, _ = fit_l23.fit_with_LM(p, idx)
            rt_dict = rt_defs.rt_dict_from_p(p)
            pred = chisq_fit.fit_func(models[0].wave, *ans, models=models,
                                      rt_dict=rt_dict)
            if label == 'Pow':
                ax2.plot(models[0].wave, prep['model_Rrs'], 'o', ms=4,
                         color=CLR['truth'], label='L23')
            ax2.plot(models[0].wave, pred, color=CLR[label], label=label)
            axr2.plot(models[0].wave,
                      100*(pred - prep['model_Rrs'])/prep['model_Rrs'],
                      color=CLR[label])
        ax2.set_yscale('log')
        ax2.set_title(f'Clear L23 idx {idx}: no regression')

    for ax in (axs[0, 0], axs[0, 1]):
        ax.set_ylabel(r'$R_{rs}$ [sr$^{-1}$]')
        ax.legend(fontsize=9)
    for ax in (axr, axr2):
        ax.axhline(0., color='k', lw=0.8)
        ax.set_xlabel('wavelength [nm]')
        ax.set_ylabel('residual [%]')
    for ax in axs.flatten():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir/'turbid_rrs_panels.png', dpi=150)
    plt.close(fig)


def fig_identifiability(res, outdir):
    """Parameter precision per strategy, and the correlation structure."""
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # -- per-parameter uncertainty, strategy by strategy
    ax = fig.add_subplot(gs[0])
    order = ['chisq tight', 'MCMC tight', 'MCMC inflated',
             'MCMC Pow2Flat']
    pn4 = res['MCMC tight']['pnames']
    x = np.arange(len(pn4))
    width = 0.2
    for jj, key in enumerate(order):
        r = res[key]
        vals = [r['sig'][r['pnames'].index(nm)] if nm in r['pnames']
                else np.nan for nm in pn4]
        ax.bar(x + (jj - 1.5)*width, vals, width, label=key)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(pn4, rotation=45, ha='right')
    ax.set_ylabel('parameter uncertainty (dex or linear)')
    ax.set_title('Precision by strategy')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # -- correlation matrices, tight vs inflated
    for jj, key in enumerate(['MCMC tight', 'MCMC inflated']):
        ax = fig.add_subplot(gs[jj + 1])
        r = res[key]
        im = ax.imshow(r['corr'], vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(r['pnames'])))
        ax.set_xticklabels(r['pnames'], rotation=90, fontsize=8)
        ax.set_yticks(range(len(r['pnames'])))
        ax.set_yticklabels(r['pnames'], fontsize=8)
        ax.set_title(f'posterior correlation\n{key}', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(outdir/'turbid_identifiability.png', dpi=150)
    plt.close(fig)


def fig_l23(l23, outdir):
    """L23 reduced chi-squared and IOP accuracy, model by model."""
    if l23 is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key, ylbl in zip(
            axs, ['chi2nu', 'a_err', 'bb_err'],
            [r'$\chi^2_\nu$', r'median $|\Delta a_{nw}|/a_{nw}$',
             r'median $|\Delta b_{b,nw}|/b_{b,nw}$']):
        for label, _ in MODELS:
            r = l23[label]
            if not len(r[key]):
                continue
            idxs = sorted(r[key])
            ax.plot(idxs, [r[key][i] for i in idxs], 'o-', ms=5,
                    color=CLR[label], label=label, alpha=0.8)
        ax.set_yscale('log')
        ax.set_xlabel('L23 index')
        ax.set_ylabel(ylbl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[0].set_title('Fit quality')
    axs[1].set_title('Absorption accuracy')
    axs[2].set_title('Backscattering accuracy')
    fig.tight_layout()
    fig.savefig(outdir/'turbid_l23_regression.png', dpi=150)
    plt.close(fig)


def main():
    outdir = Path(__file__).parent
    rng = np.random.default_rng(20260728)

    scene = synth_truth()
    print(f'Turbid synthetic: Rrs peaks at '
          f'{scene["wave"][np.argmax(scene["Rrs"])]:.0f} nm, '
          f'max Rrs = {scene["Rrs"].max():.4f} sr^-1')
    print(f'  bb_nw: {scene["bb_nw"][0]:.4f} m^-1 at 400 nm -> '
          f'{scene["bb_nw"][-1]:.4f} at 750 nm')

    rec = run_recovery(scene, rng)
    report_recovery(rec)

    sweep = run_sweep(rng)
    report_sweep(sweep)

    res, obs, sigma, sig_inflated = run_identifiability(scene, rng)
    report_identifiability(res)

    # Convergence with scipy's default budget vs a raised one.  This is
    # what the maxfev keyword on chisq_fit.fit exists for.
    l23_default = run_l23(maxfev=None)
    l23 = run_l23(maxfev=40000)
    report_l23(l23, maxfev=40000)
    if l23_default is not None:
        print('\n   Effect of the evaluation budget on convergence:')
        for label, _ in MODELS:
            n_def = len(l23_default[label]['chi2nu'])
            n_big = len(l23[label]['chi2nu'])
            print(f'     {label:10s} scipy default {n_def}/{len(L23_IDX)}'
                  f'   maxfev=40000 {n_big}/{len(L23_IDX)}')

    fig_recovery(scene, rec, outdir)
    fig_rrs(scene, rec, l23, outdir)
    fig_identifiability(res, outdir)
    fig_l23(l23, outdir)
    print(f'\nFigures written to {outdir}')


if __name__ == '__main__':
    main()
