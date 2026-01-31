# Setting up a test

## The test test_l23_fitting.py in bing/tests/ is failing.  Please suggest a fix.  If you need to run it, use the "ocean14" environment in conda.

## Thanks.  Now add checks in the test_single_fit() method to test the output.  We will use this test as we develop parts of the RT module.

## ok, now add comparisons of the fitted values to the true values.  The true values are in l23_dict and need to be cut down to the analyzed wavelengths, provided in p_expb.

# Refactoring rrs.py

## In the bing.rt folder there is code related to Raman scattering in rrs.py and raman.py.  Please move nearly all of the code into raman.py (avoiding duplication).   Leave the calc_raman_correction_factor() function in rrs.py but have it call raman.py for Raman methods.  If you need to run Python, use the "ocean13" environment in conda.