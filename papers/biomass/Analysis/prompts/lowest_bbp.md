# Assessing the sensitivity of PACE to bbp

## Goals

Here are our goals:

- Generate simulated PACE Rrs spectra from a range of bbp values with sensible a_ph and CDOM absorption values
- Fit the spectra with BING
- Assess the sensitivity of PACE as a function of bbp

## Data

We may use the simulated spectra provided by Loisel et al. (2023) for this.  Those data are located in $OS_COLOR/Loisel2023 and can be loaded with the ocpy.hydrolight.loisel23.load_ds() method.

## Code

### Writing

Here are guidelines for writing code:

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place any new code in the existing Analysis/py directory

## Development

1. Begin by generating a single synthetic spectrum from the Loisel et al. (2023) dataset.  Do the following:

- Choose the model with the lowest bbp value.
- Follow the code in the bing/fitting/l23.py module to load the spectrum and add noise
- Generate a PACE Rrs spectrum and error spectrum
- Generate a figure of the spectrum and error spectrum
- Indicate the bbp value in the figure
- Generate a new module named lowest_bbp.py in Analysis/py with a method to generate the synthetic spectrum. 
- Model that module after the fit_with_argo.py module.

## Code

1. Read this doc.  Proceed with the first item under Development