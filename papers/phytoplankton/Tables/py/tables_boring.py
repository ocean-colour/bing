"""  Module for Tables for the CNMF paper I """
# Imports
import os, sys


# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import ssl_paper_analy

from oceancolor.satellites import modis as oc_modis
from oceancolor.satellites import seawifs as oc_seawifs

from cnmf import io as cnmf_io

from IPython import embed


def mktab_modis(outfile='tab_modis.tex'):

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{'+'MODIS Data \\label{tab:modis}}\n')
    tbfil.write('\\begin{tabular}{cc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Band & \sreflect \\\\ \n')
    tbfil.write('(nm) & (sr$^{-1}$) \\\\ \n')
    tbfil.write('\\hline \n')

    for kk, wv in enumerate(oc_modis.modis_wave):
        tbfil.write('{:d} & {:0.4f} \\\\ \n'.format(wv, oc_modis.modis_aqua_error[kk]))

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    #tbfil.write('Notes: The \\DT\\ value listed here is measured from the inner $40 \\times 40$\,pixel$^2$ region of the cutout. \\\\ \n')
    #tbfil.write('LL is the log-likelihood metric calculated from the \\ulmo\\ algorithm. \\\\ \n')
    #tbfil.write('$U_{0,\\rm all}, U_{1,\\rm all}$ are the UMAP values for the UMAP analysis on the full dataset. \\\\ \n')
    #tbfil.write('$U_0, U_1$ are the UMAP values for the UMAP analysis in the \\DT\\ bin for this cutout. \\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


def mktab_seawifs(outfile='tab_seawifs.tex'):

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{'+'SeaWiFS Data \\label{tab:seawifs}}\n')
    tbfil.write('\\begin{tabular}{cc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Band & \sreflect \\\\ \n')
    tbfil.write('(nm) & (sr$^{-1}$) \\\\ \n')
    tbfil.write('\\hline \n')

    for kk, wv in enumerate(oc_seawifs.seawifs_wave):
        tbfil.write('{:d} & {:0.4f} \\\\ \n'.format(
            wv, oc_seawifs.seawifs_error[kk]))

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    #tbfil.write('Notes: The \\DT\\ value listed here is measured from the inner $40 \\times 40$\,pixel$^2$ region of the cutout. \\\\ \n')
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
    mktab_seawifs()