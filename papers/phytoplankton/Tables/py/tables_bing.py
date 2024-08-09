"""  Module for Tables for the CNMF paper I """
# Imports
import os, sys


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy

from ocpy.satellites import modis as oc_modis
from ocpy.satellites import seawifs as oc_seawifs

from cnmf import io as cnmf_io

from IPython import embed


def mktab_error(dataset:str):

    # Grab the error
    if dataset == 'MODIS':
        err_dict = oc_modis.calc_errors()
        waves = oc_modis.modis_wave
        outfile='tab_modis.tex'
        caption = '\\caption{'+'MODIS Data \\label{tab:modis}}\n'
    elif dataset == 'SeaWiFS':
        err_dict = oc_seawifs.calc_errors()
        waves = oc_seawifs.seawifs_wave
        caption = '\\caption{'+'SeaWiFS Data \\label{tab:seawifs}}\n'
        outfile='tab_seawifs.tex'

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write(caption)
    tbfil.write('\\begin{tabular}{cc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Band & \sreflect \\\\ \n')
    tbfil.write('(nm) & (sr$^{-1}$) \n')
    tbfil.write('\\hline \n')

    for kk, wv in enumerate(waves):
        tbfil.write('{:d} & {:0.4f} \\\\ \n'.format(
            wv, err_dict[waves[kk]][0]))
        #tbfil.write('{:d} & {:0.4f} & {:0.1f} \\\\ \n'.format(
        #    wv, err_dict[waves[kk]][0], 100*err_dict[waves[kk]][1]))

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    tbfil.write('Notes: The error has assumed that 1/2 of the variance is due to the in the in-situ measurements. \\\\ \n')
    #tbfil.write('PD = absolute percent difference \\\\ \n')
    #tbfil.write('LL is the log-likelihood metric calculated from the \\ulmo\\ algorithm. \\\\ \n')
    #tbfil.write('$U_{0,\\rm all}, U_{1,\\rm all}$ are the UMAP values for the UMAP analysis on the full dataset. \\\\ \n')
    #tbfil.write('$U_0, U_1$ are the UMAP values for the UMAP analysis in the \\DT\\ bin for this cutout. \\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


    

# Command line execution
if __name__ == '__main__':

    #mktab_modis()
    mktab_error('MODIS')
    mktab_error('SeaWiFS')