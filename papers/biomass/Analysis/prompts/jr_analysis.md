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

2. Modify the list_jr_matchups() method in jr_analysis.py to print the lat,lon and UT time of the matched Argo profile when `verbose` is True.  Log your work in Logs.

## Prompts

## Code

1. Read this doc.  Proceed with the first item under Development
2. Read this doc.  Proceed with the 2nd item under Development
3. Read this doc.  Proceed with the 3rd item under Development
4. Read this doc.  Proceed with the 4th item under Development

5. Read this doc.  Perform the 1st set of modifications in the Modifications section above.
6. Read this doc.  Perform the 2nd set of modifications in the Modifications section above.

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-06-20 (verbose lat/lon/UT-time in list_jr_matchups)

Performed the 2nd set of modifications: extended `list_jr_matchups()` in
`Analysis/py/jr_analysis.py` to report the matched Argo profile's location and
time when `verbose=True`.

- The per-row matcher `jr_utils.match_jr_to_argo()` returns `argo_row` as the
  *index* into the Argo DataFrame (not the row itself), so I uncommented the
  `argo_df.loc[mdict['argo_row']]` lookup to pull `lat`, `lon`, and `time`.
- Appended three new columns to the returned DataFrame — `argo_lat`,
  `argo_lon`, and `argo_time` — alongside the existing `cruise`/`profile`/
  `match_dist_km`/`match_dt_hours`. Unmatched rows (missing PACE lat/lon/time)
  get NaN/NaT placeholders so the column dtypes stay clean.
- The verbose line now also prints `Argo lat=`, `lon=`, and `UT=` (ISO-8601 to
  seconds resolution via `Timestamp.strftime`).
- Updated the docstring Returns section to document the new columns.

Verified with `python jr_analysis.py 5`: all 9 matched JR rows print their Argo
lat/lon and UT time (e.g. row 6 → cruise 7902226 profile 4 at lat 27.479,
lon -46.221, UT 2025-02-18T20:26:59); the 6 rows lacking PACE geolocation still
report as unmatched. Note the matched Argo `time` equals the JR `time` here, so
all `dt` values are ~0 h, consistent with the near-exact time matching.