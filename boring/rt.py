""" Radiation Transfer methods for BIG """

# Conversion from rrs to Rrs
A_Rrs, B_Rrs = 0.52, 1.7

# Gordon factors
G1, G2 = 0.0949, 0.0794  # Standard Gordon factors


def calc_Rrs(a, bb, in_G1=None, in_G2=None):
    """
    Calculates the remote sensing reflectance (Rrs) using the given absorption (a) and backscattering (bb) coefficients.

    Parameters:
        a (float or array-like): Absorption coefficient.
        bb (float or array-like): Backscattering coefficient.
        in_G1 (float or array-like, optional): G1 value. Default is None.
        in_G2 (float or array-like, optional): G2 value. Default is None.

    Returns:
        float or array-like: Remote Sensing Reflectance (Rrs) value.
    """
    # u
    u = bb / (a+bb)
    # rrs
    if in_G1 is not None:
        t1 = in_G1 * u
    else: 
        t1 = G1 * u
    if in_G2 is not None:
        t2 = in_G2 * u*u
    else:
        t2 = G2 * u*u
    rrs = t1 + t2
    # Done
    Rrs = A_Rrs*rrs / (1 - B_Rrs*rrs)
    return Rrs