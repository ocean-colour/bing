# Fussing with the JR Rrs extractions

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

2. Create a module named jr_analysis.py in Analysis/py with a method to analyze the JR Rrs extractions.  It should:

- Load the JR Rrs extractions
- Load the PACE Rrs from $OS_COLOR/Biomass/Fits folder.  Use the load_fit_data method to load the fits.
- Generate a figure comparing the JR Rrs extractions to the PACE Rrs.
- Follow the style of the module named fit_with_argo.py.

3. Create a method in the jr_analysis.py module to fit individual JR Rrs extractions.  It should:

- Load the JR Rrs extractions
- Fit the JR Rrs extractions with BING.  See the fit_with_argo.py module for examples.
- Save the fits to the Frouin/ folder.
- Use the plot_fit() method in fitting.py to generate a figure of the fit.  Place this figure in the Frouin/ folder.

4. Create a method in the jr_analysis.py module that reads in the jr_test_matchup_L1B.csv file into a pandas DataFrame and prints the cruise and profile values for each spectrum.

## Modifications

1. Make these changes to the jr_utils.py and jr_analysis.py modules:

- When matching the Frouin analysis to an Argo crusie/profile pair, use the AOP_file name matched to the closest_file name in the matched CSV file, and the lat,lon and closest_time in the matched CSV file.
- Add a new method to jr_utils.py "match_jr_to_pace()" to perform this.  In the future, I will provide the PACE_lat, PACE_lon in the matched CSV file but use the lat,lon values for Argo for now.  
- Require the match in time to be nearly exact.  For the time, use the closest time in the CSV file.
- Refactor the methods in jr_utils.py and jr_analysis.py to use the new method.

## Prompts

## Code

1. Read this doc.  Proceed with the first item under Development
2. Read this doc.  Proceed with the 2nd item under Development
3. Read this doc.  Proceed with the 3rd item under Development
4. Read this doc.  Proceed with the 4th item under Development
5. Read this doc.  Perform the 1st set of modifications in the Modifications section above.
