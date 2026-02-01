"""
Radiation Transfer Module
=========================

This module contains radiative transfer functions for computing remote sensing
reflectance (Rrs) from inherent optical properties (IOPs).

Submodules
----------
rrs
    Standard Gordon (1988) elastic scattering model, Raman corrections,
    and chlorophyll fluorescence Rrs calculations
raman
    Raman scattering coefficients and redistribution functions
chl_fl
    Chlorophyll fluorescence inelastic emission model (low-level functions)
"""

from . import rrs
from . import raman
from . import chl_fl

# Re-export commonly used functions and constants at package level
from .rrs import (
    A_Rrs,
    B_Rrs,
    calc_Rrs,
    calc_elastic_Rrs,
    calc_Rrs_fluorescence,
    calc_Rrs_fluorescence_simple,
    calc_Rrs_with_fluorescence,
    calc_fluorescence_spectrum,
    calc_fluorescence_correction_factor,
    #calc_a_ph_bricaud,
    #calc_a_water,
    #calc_bb_water,
)
from .raman import (
    calc_Rrs_with_raman,  # Now in raman module
)
from .chl_fl import (
    calc_R_fluorescence,
    calc_fluorescence_line_height,
    fluorescence_backscattering_coeff,
)

__all__ = [
    'rrs',
    'raman',
    'chl_fl',
    # Constants
    'A_Rrs',
    'B_Rrs',
    # Rrs
    'calc_Rrs',
    'calc_elastic_Rrs',
    # Fluorescence Rrs
    'calc_Rrs_fluorescence',
    'calc_Rrs_fluorescence_simple',
    'calc_Rrs_with_fluorescence',
    'calc_fluorescence_spectrum',
    'calc_fluorescence_correction_factor',
    # IOP functions
    #'calc_a_ph_bricaud',
    #'calc_a_water',
    #'calc_bb_water',
    # Low-level fluorescence
    'calc_R_fluorescence',
    'calc_fluorescence_line_height',
    'fluorescence_backscattering_coeff',
]
