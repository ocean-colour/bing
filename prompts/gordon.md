# Develop wavelength dependent Gordon coefficients

## Goals

I wish to derive wavelength dependent Gordon coefficients for radiative transfer approximations from assumed IOPs.  


## Development 

1. Please generate a new module in dev/ named calc_gordon.py that includes a method to do this.  Base it on the method fig_u() in bing/papers/phytoplankton/Figures/py/figs_phyto.py which inputs Hydrolight data from Loisel23 and estimates G1 and G2 them at several wavelengths.  If you need to run Python, use the "ocean14" environment in conda.

2. Thanks!  Now generate a Jupyter Notebook in dev/ that runs the code.  Include a cell that writes the best G1 and G2 values to disk.  

3. I am finding that the variable Gordon coefficients are not working well at redder wavelengths for oligotrophic waters.  Please investigate this by generating a new Notebook in dev/Gordon named chk_Gordon_oligotrophic.ipynb that examines the Gordon coefficients as a function of bbp for oligotrophic waters.  Do the following:

- Use the methods in bing.rt.rrs to calculate the Gordon Rrs from the IOPs of the Loisel23 dataset.
- Plot the difference between the elastic Hydrolight Rrs and the Gordon Rrs as a function of bbp.  Do so at select wavelengths: 400nm, 500nm, 550nm, 600nm, 650nm, 700nm.

## Code

Here are guidelines for coding: 

- Use Python
- When possible use existing methods from the modules in fronts/properties/ and fronts/viz
- Add inline comments to explain the effort
- Reuse existing code when possible
- Use methods, not classes
- Place I/O methods in the fronts/properties/io.py module.
- Place import statements at the top of the file.
- Include a description of inputs/outputs in the doc string of all methods


## I/O

### Modify the Notebook calc_gordon.ipynb in bing/dev/Gordon to write the output to a CSV file instead of a Numpy save file.  Update the Notebook and also the code in calc_gordon.py.  If you need to run Python, use the "ocean13" environment in conda

# Further checks

1. Generate a new Notebook in dev/Gordon named chk_gordon_Loisel23.ipynb that reads in the Elastic outputs of Loisel23 and for a single index (idx=170) and comparse the Hydrolight calculation of Rrs from Loisel23 with our new estimate using the Gordon coefficients.

2. Modify the calc_gordon.ipynb Notebook to plot rrs vs. u for the Loisel23 dataset and the best fit for the Gordon coefficients at a select wavelength (e.g. 370nm).

3. Generate a new Notebook in dev/Gordon named chk_Gordon_bbp.ipynb that examines the Gordon coefficients as a function of bbp which is provided in the Loisel23 dataset.  Do the following:

- Use the methods in bing.rt.rrs to calculate the Gordon Rrs from the IOPs of the Loisel23 dataset.
- Plot the difference between the Hydrolight Rrs and the Gordon Rrs as a function of bbp.  Do so at select wavelengths: 400nm, 500nm, 550nm, 600nm, 650nm, 700nm.

## Prompts

1. Read this doc. Proceed with the 3rd item under Further checks.