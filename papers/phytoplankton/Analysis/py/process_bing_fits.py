""" Process BING's fits to the data. """
import sys, os
from collections import namedtuple

import numpy as np
import pandas



from ocpy.hydrolight import loisel23

from bing.models import utils as model_utils
from bing import evaluate

sys.path.append(os.path.abspath("../../bing_2.0/Analysis/py"))
import anly_utils_20

import single_bing_fits

from IPython import embed

# bb water
ds = loisel23.load_ds(4,0)
iwave = np.argmin(np.abs(ds.Lambda.data - 440))
bbw_440=ds.bb.data[0,iwave]-ds.bbnw.data[0,iwave]

def replace_one(p, outfile:str, idx:int):
    # Load originals
    df = pandas.read_csv(outfile)

    # Run
    standard, extras = process_one(idx, pdict=p._asdict(), verbose=True)
    for key in standard.keys():
        df.loc[idx, key] = standard[key]
    for key in extras.keys():
        df.loc[idx, key] = extras[key]

    # Write
    df.to_csv(outfile, index=False)
    print(f'Replaced: {idx} in {outfile}')


def main(flg):
    flg = int(flg)

    # Test
    if flg == 1:
        p = single_bing_fits.standard_expb_pow()
        standard, extras = process_one(170, pdict=p._asdict(), verbose=True)
        embed(header='119 of process')

    # Run em all for ExpBricaud
    if flg == 2:
        p = single_bing_fits.standard_expb_pow()
        process_all(p, 'BING_L23_results_ExpBricaudPow.csv')#, debug=True)

    # Run em all on GIOP
    if flg == 3:
        p = single_bing_fits.standard_giop()
        process_all(p, 'BING_L23_results_GIOPLee.csv')#, debug=True)

    # Replace one
    if flg == 4:
        idx = 2773
        p = single_bing_fits.standard_expb_pow()
        replace_one(p, 'BING_L23_results_ExpBricaudPow.csv', idx)#, debug=True)
        p = single_bing_fits.standard_giop()
        replace_one(p, 'BING_L23_results_GIOPLee.csv', idx)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg = 1 # -- Testing
        #flg = 2 # -- ExpBricaud Pow
        #flg = 3 # -- GIOP Lee

    else:
        flg = sys.argv[1]

    main(flg)
    
