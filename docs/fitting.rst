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

    from bing.fitting import chisq_fit
    from bing.models import utils as model_utils
    import numpy as np
    
    # Initialize models
    wavelengths = np.arange(400, 701, 5)
    models = model_utils.init(['ExpBricaud', 'PowerLaw'], wavelengths)
    
    # Prepare data
    Rrs_measured = np.array([...])  # Your measured Rrs
    Rrs_uncertainty = np.array([...])  # Uncertainties
    
    # Perform fit
    result = chisq_fit.fit(
        models, 
        wavelengths, 
        Rrs_measured, 
        Rrs_uncertainty,
        bounds=[(1e-4, 10)] * 4  # Parameter bounds
    )
    
    print(f"Best-fit parameters: {result['x']}")
    print(f"Chi-squared: {result['chisq']}")
    print(f"Reduced chi-squared: {result['rchisq']}")

Advanced Options
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Custom initial guess
    p0 = [0.01, 0.65, 0.015, 0.001]
    
    # Fit with constraints
    result = chisq_fit.fit(
        models, wavelengths, Rrs_measured, Rrs_uncertainty,
        p0=p0,
        method='trf',  # Trust Region Reflective algorithm
        bounds=[(1e-6, 1), (0.3, 1.0), (0.01, 0.02), (1e-4, 0.1)],
        max_nfev=1000  # Maximum function evaluations
    )

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

Specialized fitting for Loisel et al. (2023) synthetic dataset.

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
    idx = 170  # Profile index
    chains, models, prep_dict, idx, extras = l23.fit_one(params, idx)
    
    # Save results
    outfile = l23.chain_filename(params, idx=idx, path='./fits/')
    l23.save_chains(chains, idx, outfile, extras=extras)

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
