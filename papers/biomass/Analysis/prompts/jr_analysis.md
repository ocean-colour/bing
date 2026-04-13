# Fussing with the JR Rrs extraction

## Goals

Here are our goals:

- Build code to convert the table output of JR to Rrs spectra
- Examine the output
- Perform BING fits on the spectra
- Compare to fits on PACE provided Rrs

## Data

An example file of 15 Rrs spectra is provided in the papers/biomass/Analysis/Frouin/jr_test_matchup_L1B.csv file.  This file is a CSV file.  The spectra are stored in the columns.   The JR processed Rrs have a prefix of Rrs_mean for the mean Rrs and Rrs_std for the standard deviation.  

## Code

### Writing

Here are guidelines for writing code:

- Use Python
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place any new code in the existing Analysis/py directory

## Development

1. Generate a module named jr_utils.py in Analysis/py with a method to extract Rrs spectra from the CSV file.  It should:

- Load up the matched_argo_bgc_profiles_bbp_v3.csv file in Analysis/
- Take as input the cruise and profile values of the Argo float
- Match on lat, lon and time to the jr_test_matchup_L1B.csv file
- Extract the Rrs_mean and Rrs_std spectra
- Also provide the wavelengths as an array based on the column names

## Prompts

## Code

1. Read this doc.  Proceed with the first item under Development
