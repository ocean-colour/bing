.. _fitting:

==================
Fitting Algorithms
==================

BING provides multiple fitting algorithms for parameter estimation in bio-optical models,
ranging from simple least-squares to sophisticated Bayesian inference methods.

Overview
--------

The fitting module includes:

* **Least-squares fitting** - Fast initial parameter estimation
* **MCMC sampling** - Bayesian inference with uncertainty quantification
* **L23 fitting** - Specialized fitting for Loisel et al. (2023) data
* **Chi-square minimization** - Weighted least-squares with error propagation

Least-Squares Fitting
---------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    import numpy as np

    from bing.fitting import chisq_fit
    from bing.models import utils as model_utils
    from bing.parameters import standard
    from bing.rt import defs as rt_defs

    # Configuration and models
    wavelengths = np.arange(400, 701, 5)
    p = standard.expb_pow(wv_min=400., wv_max=700.)
    models = model_utils.init(p.model_names, wavelengths,
                              (p.apriors, p.bpriors))
    rt_dict = rt_defs.rt_dict_from_p(p)

    # Data, initial guess and parameter bounds.  Bounds come from the
    # priors, as (lower_array, upper_array) -- not a list of pairs.
    items = (Rrs_measured, varRrs, p0, idx)
    low = np.array([d['pmin'] for d in p.apriors] +
                   [d['pmin'] for d in p.bpriors])
    high = np.array([d['pmax'] for d in p.apriors] +
                    [d['pmax'] for d in p.bpriors])

    ans, cov, idx = chisq_fit.fit(items, models, rt_dict,
                                  bounds=(low, high))

    # ans holds the best-fit parameters in fitting space (log10 for
    # amplitudes); sqrt(diag(cov)) gives 1-sigma uncertainties.
    pred = chisq_fit.fit_func(wavelengths, *ans, models=models,
                              rt_dict=rt_dict)
    chi2 = np.sum((pred - Rrs_measured)**2/varRrs)
    print(f"Reduced chi-squared: {chi2/(Rrs_measured.size - ans.size)}")

``items`` is the tuple ``(Rrs, varRrs, p0, idx)``, where ``idx`` is
echoed back in the return so batch callers can reassemble results.

Advanced Options
~~~~~~~~~~~~~~~~

The evaluation budget
^^^^^^^^^^^^^^^^^^^^^

``maxfev`` caps the number of forward-model evaluations the optimizer may
spend. The default (``None``) leaves scipy's own default in place.

.. code-block:: python

    ans, cov, idx = chisq_fit.fit(items, models, rt_dict,
                                  bounds=(low, high),
                                  maxfev=40000)

Two things to know about it:

* **It changes whether the fit returns, not how well the model can fit.**
  When the budget is exhausted, ``curve_fit`` raises ``RuntimeError``;
  raising the budget converts those failures into converged fits but does
  not improve the misfit of the ones that already converged. On turbid
  in-situ spectra a roughly 40x increase moved the convergence rate from
  12.5% to 37.5% with no change in the residuals.
* **Parameter-rich models need it.** Fitting the two-component
  backscattering models (``Pow2``, ``Pow2Flat``; see :doc:`models`)
  through least squares at scipy's default budget fails on a substantial
  fraction of spectra -- 5 of 8 and 6 of 8 respectively on a clear L23
  sample, versus 8 of 8 with ``maxfev=40000``.

``maxfev`` is the correct spelling for both of ``curve_fit``'s back ends:
with finite bounds it uses ``least_squares`` ('trf') and renames the
keyword to ``max_nfev`` internally, while the unbounded case passes it to
``leastsq`` ('lm').

MCMC Fitting
------------

MCMC (Markov Chain Monte Carlo) provides full posterior distributions for parameters.

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

    from bing.fitting import inference as bing_inf
    from bing.priors import priors as bing_priors
    
    # Initialize MCMC parameters
    pdict = bing_inf.init_mcmc(
        models,
        nsteps=10000,    # Number of MCMC steps
        nburn=1000,      # Burn-in period
        nwalkers=32,     # Number of walkers
        threads=4        # Parallel threads
    )
    
    # Set priors
    priors = bing_priors.set_standard_priors(models)
    pdict['priors'] = priors

Walker initialization
~~~~~~~~~~~~~~~~~~~~~

Walkers start as a ball around ``p0``, built by
:func:`bing.fitting.inference.init_walkers`: each walker is

.. math::

    p_0 + U(-1, 1) \cdot \max(|p_0| \cdot \mathrm{frac},\ \mathrm{floor})

per parameter, then clipped into the prior bounds (from
:func:`bing.fitting.inference.prior_bounds`) so every walker starts with
a finite log-probability.

Both knobs are exposed on ``run_emcee`` as ``perturb_frac`` (default
1e-2, the relative half-width) and ``perturb_floor`` (default 1e-3, an
absolute floor). Widen them for badly degenerate models.

.. warning::

    The floor matters. A purely *multiplicative* perturbation gives a
    parameter seeded at exactly 0 no spread at all, and because emcee's
    stretch move proposes along walker-to-walker vectors, such a
    dimension never moves for the entire run -- silently, with a healthy
    acceptance fraction and a zero-width credible interval. Linear
    parameters legitimately sit at zero (a flat backscattering exponent,
    for instance), so this is not a corner case.

Reproducibility: ``init_walkers`` draws from the legacy global
``np.random`` by default, so ``np.random.seed`` -- as used by
:func:`bing.fitting.l23.batch_fit` -- still governs the initialization.
Pass ``rng=`` to inject a ``Generator`` instead.

Running MCMC
~~~~~~~~~~~~

.. code-block:: python

    # Prepare input
    items = [(Rrs_measured, Rrs_uncertainty, p0, 0)]
    
    # Run MCMC
    chains, idx = bing_inf.fit_one(
        items[0], 
        models=models, 
        pdict=pdict,
        chains_only=True
    )
    
    # chains shape: (nsteps, nwalkers, n_params)
    print(f"Chain shape: {chains.shape}")

Analyzing Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing import evaluate
    import corner
    
    # Calculate statistics
    stats = evaluate.calc_stats(chains, models, wavelengths)
    
    # Extract percentiles
    median_params = np.median(chains.reshape(-1, chains.shape[-1]), axis=0)
    percentiles = np.percentile(
        chains.reshape(-1, chains.shape[-1]), 
        [16, 50, 84], 
        axis=0
    )
    
    # Corner plot
    corner.corner(
        chains.reshape(-1, chains.shape[-1]),
        labels=['A_ph', 'E_ph', 'S_dg', 'b_bp'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True
    )

L23 Fitting
-----------

Specialized fitting for Loisel et al. (2023) synthetic dataset, which contains
~3300 Hydrolight radiative transfer simulations spanning diverse ocean conditions.
This provides a valuable validation dataset with known "true" IOPs.

Single Profile
~~~~~~~~~~~~~~

.. code-block:: python

    from bing.fitting import l23
    from bing.parameters import standard

    # Define parameters
    params = standard.expb_pow(
        satellite='PACE',
        nsteps=40000,
        add_noise=True
    )

    # Fit single profile
    idx = 170  # Profile index (0-3319)
    chains, models, prep_dict, idx, extras = l23.fit_one(params, idx)

    # Save results
    outfile = l23.chain_filename(params, idx=idx, path='./fits/')
    l23.save_chains(chains, idx, outfile, extras=extras)

Fitting with Raman Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable Raman scattering correction for improved accuracy in clear waters:

.. code-block:: python

    from bing.fitting import l23
    from bing.parameters import standard

    # Enable Raman and variable Gordon coefficients
    params = standard.expb_pow(
        satellite='PACE',
        nsteps=40000,
        include_Raman=True,      # Enable Raman correction
        variable_Gordon=True,    # Wavelength-dependent Gordon coefficients
        add_noise=True
    )

    # Fit with Raman correction
    chains, models, prep_dict, idx, extras = l23.fit_one(params, idx=170)

    # The models now have Raman-related attributes
    print(f"Excitation wavelengths: {models[0].wave_ex[:5]}")
    print(f"Raman backscatter coeff: {models[1].bb_R[:5]}")

Least-Squares Fitting
~~~~~~~~~~~~~~~~~~~~~

For quick fits without full MCMC posterior estimation, use Levenberg-Marquardt:

.. code-block:: python

    from bing.fitting import l23

    # Least-squares fit (much faster than MCMC)
    ans, cov, models, prep_dict, idx = l23.fit_with_LM(params, idx=170)

    # ans: best-fit parameters
    # cov: covariance matrix
    print(f"Best-fit params: {ans}")
    print(f"Parameter uncertainties: {np.sqrt(np.diag(cov))}")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process multiple profiles
    indices = range(100, 200)
    
    for idx in indices:
        try:
            chains, models, prep_dict, _, extras = l23.fit_one(params, idx)
            outfile = l23.chain_filename(params, idx=idx)
            l23.save_chains(chains, idx, outfile, extras=extras)
            print(f"Completed profile {idx}")
        except Exception as e:
            print(f"Failed on profile {idx}: {e}")

Custom Fitting Functions
------------------------

Chi-square with Priors
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bing.fitting import chisq_fit
    
    def custom_chi2(params, models, wave, Rrs_obs, Rrs_err, priors):
        """Chi-square with prior penalties"""
        
        # Standard chi-square
        chi2 = chisq_fit.calc_chisq(
            params, models, wave, Rrs_obs, Rrs_err
        )
        
        # Add prior penalties
        for i, (param, prior) in enumerate(zip(params, priors)):
            if prior['type'] == 'gaussian':
                chi2 += ((param - prior['mean']) / prior['std'])**2
        
        return chi2

Bayesian Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scipy.special import logsumexp
    
    def calculate_evidence(chains):
        """Estimate model evidence using harmonic mean"""
        
        # Log-likelihoods
        log_likes = -0.5 * chains[:, :, -1]  # Last column is chi2
        
        # Harmonic mean estimator
        log_evidence = -logsumexp(-log_likes) + np.log(len(log_likes))
        
        return log_evidence
    
    # Compare models
    evidence_1 = calculate_evidence(chains_model1)
    evidence_2 = calculate_evidence(chains_model2)
    bayes_factor = np.exp(evidence_1 - evidence_2)

Fitting Strategies
------------------

Sequential Fitting
~~~~~~~~~~~~~~~~~~

Fit parameters sequentially for better convergence:

.. code-block:: python

    # First fit absorption only
    result_abs = chisq_fit.fit(
        [models[0]], wave, Rrs, Rrs_err,
        bounds=[(1e-4, 1), (0.3, 1.0), (0.01, 0.02)]
    )
    
    # Then fit backscattering with fixed absorption
    result_bb = chisq_fit.fit(
        [models[1]], wave, Rrs, Rrs_err,
        fixed_params={'anw': result_abs['x']},
        bounds=[(1e-4, 0.1)]
    )

Regularization
~~~~~~~~~~~~~~

Add regularization to prevent overfitting:

.. code-block:: python

    def regularized_cost(params, models, wave, Rrs, Rrs_err, lambda_reg=0.01):
        """Cost function with L2 regularization"""
        
        chi2 = chisq_fit.calc_chisq(params, models, wave, Rrs, Rrs_err)
        
        # L2 penalty
        reg_term = lambda_reg * np.sum(params**2)
        
        return chi2 + reg_term

Error Analysis
--------------

Uncertainty Propagation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from scipy import stats
    
    def propagate_errors(chains, models, wave):
        """Propagate parameter uncertainties to Rrs"""
        
        n_samples = 1000
        n_wave = len(wave)
        Rrs_samples = np.zeros((n_samples, n_wave))
        
        # Sample from posterior
        flat_chains = chains.reshape(-1, chains.shape[-1])
        indices = np.random.choice(len(flat_chains), n_samples)
        
        for i, idx in enumerate(indices):
            params = flat_chains[idx]
            Rrs_samples[i] = evaluate.forward_model(params, models, wave)
        
        # Calculate statistics
        Rrs_mean = np.mean(Rrs_samples, axis=0)
        Rrs_std = np.std(Rrs_samples, axis=0)
        Rrs_percentiles = np.percentile(Rrs_samples, [5, 16, 50, 84, 95], axis=0)
        
        return Rrs_mean, Rrs_std, Rrs_percentiles

Goodness of Fit
~~~~~~~~~~~~~~~

.. code-block:: python

    def assess_fit_quality(observed, modeled, uncertainty):
        """Calculate fit quality metrics"""
        
        residuals = observed - modeled
        weighted_residuals = residuals / uncertainty
        
        metrics = {
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'bias': np.mean(residuals),
            'r2': 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2),
            'chi2': np.sum(weighted_residuals**2),
            'reduced_chi2': np.sum(weighted_residuals**2) / (len(observed) - 4)
        }
        
        return metrics

Convergence Diagnostics
-----------------------

Gelman-Rubin Statistic
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def gelman_rubin(chains):
        """Calculate Gelman-Rubin convergence diagnostic"""
        
        n_steps, n_walkers, n_params = chains.shape
        
        # Split chains
        split_chains = chains.reshape(n_steps, n_walkers * 2, n_params // 2, 2)
        
        # Within-chain variance
        W = np.mean(np.var(split_chains, axis=0))
        
        # Between-chain variance
        chain_means = np.mean(split_chains, axis=0)
        B = np.var(chain_means) * n_steps
        
        # Potential scale reduction factor
        var_est = (1 - 1/n_steps) * W + B/n_steps
        R_hat = np.sqrt(var_est / W)
        
        return R_hat

Autocorrelation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from emcee import autocorr
    
    def analyze_autocorrelation(chains):
        """Analyze chain autocorrelation"""
        
        # Integrated autocorrelation time
        tau = autocorr.integrated_time(chains, quiet=True)
        
        # Effective sample size
        n_eff = chains.shape[0] * chains.shape[1] / np.max(tau)
        
        print(f"Autocorrelation time: {tau}")
        print(f"Effective samples: {n_eff:.0f}")
        
        return tau, n_eff

Best Practices
--------------

1. **Initial Guess**: Use least-squares for initial parameter estimates
2. **Burn-in**: Remove initial samples to ensure convergence
3. **Thinning**: Thin chains to reduce autocorrelation
4. **Multiple Chains**: Run multiple independent chains
5. **Convergence Checks**: Always verify chain convergence
6. **Prior Selection**: Use informative priors when available
7. **Model Comparison**: Use information criteria for model selection

Performance Tips
----------------

* Use parallel processing for MCMC (`threads` parameter)
* Vectorize likelihood calculations
* Profile code to identify bottlenecks
* Consider approximate methods for large datasets
* Cache expensive computations (e.g., water optical properties)
