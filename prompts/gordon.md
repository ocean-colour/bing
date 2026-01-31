# Develop wavelength dependent Gordon coefficients

## I wish to derive wavelength dependent Gordon coefficients for radiative transfer approximations from assumed IOPs.  Please generate a new module in dev/ named calc_gordon.py that includes a method to do this.  Base it on the method fig_u() in bing/papers/phytoplankton/Figures/py/figs_phyto.py which inputs Hydrolight data from Loisel23 and estimates G1 and G2 them at several wavelengths.  If you need to run Python, use the "ocean14" environment in conda.

## Thanks!  Now generate a Jupyter Notebook in dev/ that runs the code.  Include a cell that writes the best G1 and G2 values to disk.  

# I/O

## Modify the Notebook calc_gordon.ipynb in bing/dev/ to write the output to a CSV file instead of a Numpy save file.  Update the Notebook and also the code in calc_gordon.py.  If you need to run Python, use the "ocean13" environment in conda

# Further checks

## Generate a new Notebook in dev/ named chk_gordon_Loisel23.ipynb that reads in the Elastic outputs of Loisel23 and for a single index (idx=170) and comparse the Hydrolight calculation of Rrs from Loisel23 with our new estimate using the Gordon coefficients.

## Modify the calc_gordon.ipynb Notebook to plot rrs vs. u for the Loisel23 dataset and the best fit for the Gordon coefficients at a select wavelength (e.g. 370nm).