# Constrained fits to Argo bbp measurements

## Goals

Analyze fits to the PACE data constrained to the Argo bbp measurements.

## Data

BING fits that are both free and constrained are in the Argo_Constrained/ folder for select cruise/profile pairs.

## Code

### Writing

Here are guidelines for writing code:

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place imports at the top of the file

## Development

1. Examine parameter changes

Write a method to examine the parameter changes between the free and constrained fits.  It should:

- Load all of the free and constrained fits in the Argo_Constrained/ folder
- Generate a DataFrame with the parameters
- Add the cruise and profile values to the DataFrame
- Compare the parameters
- Use the bing.io.load_fit method to load the fit outputs

Add the method to the py/fit_with_argo.py module.

## Prompts

1. Read this doc.  Proceed with the first item under Development
