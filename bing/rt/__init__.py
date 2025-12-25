"""
Radiation Transfer Module
=========================

This module contains radiative transfer functions for computing remote sensing
reflectance (Rrs) from inherent optical properties (IOPs).

Submodules
----------
rrs
    Standard Gordon (1988) elastic scattering model and Raman corrections
raman
    Raman scattering coefficients and redistribution functions
chl_fl
    Chlorophyll fluorescence inelastic emission model
"""

from . import rrs
from . import raman
from . import chl_fl

# Re-export commonly used functions at package level
from .rrs import calc_Rrs, calc_Rrs_with_raman
from .chl_fl import (
    calc_R_fluorescence,
    calc_fluorescence_line_height,
    fluorescence_backscattering_coeff,
)

__all__ = [
    'rrs',
    'raman',
    'chl_fl',
    'calc_Rrs',
    'calc_Rrs_with_raman',
    'calc_R_fluorescence',
    'calc_fluorescence_line_height',
    'fluorescence_backscattering_coeff',
]
