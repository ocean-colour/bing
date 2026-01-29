# Setting up a test

## The test test_l23_fitting.py in bing/tests/ is failing.  Please suggest a fix.  If you need to run it, use the "ocean14" environment in conda.

## Thanks.  Now add checks in the test_single_fit() method to test the output.  We will use this test as we develop parts of the RT module.

## ok, now add comparisons of the fitted values to the true values.  The true values are in l23_dict and need to be cut down to the analyzed wavelengths, provided in p_expb.