import numpy as np

from ocpy.satellites import pace as sat_pace
from ocpy.satellites import modis as sat_modis
from ocpy.satellites import seawifs as sat_seawifs
from ocpy.satellites import sbg as sat_sbg

def add_noise(Rs, perc:int=None, abs_sig:float=None,
              wave:np.ndarray=None, correlate:bool=False):
    """
    Add random noise to the input array Rs.

    Parameters:
        Rs (np.ndarray): Input array.
        perc (int, optional): Percentage of noise to be added as a fraction of Rs. Default is None.
        abs_sig (float, str, optional): Absolute value of noise to be added. Default is None.
        correlate (bool, optional): Whether to correlate the noise. Default is False.

    Returns:
        ndarray: Array with noise added.
    """
    use_Rs = Rs.copy()

    # Random draws
    if correlate:
        npix = Rs.shape[1]
        # Genearte the covariance matrix
        vals = {0: 1., 1: 0.5, 2: 0.3, 3: 0.1}
        cov_m = np.zeros((npix,npix))
        for jj in range(npix):
            i0 = max(0, jj-3)
            i1 = min(jj+4, npix)
            for ii in range(i0, i1):
                diff = int(np.abs(ii-jj))
                cov_m[jj,ii] = vals[diff] 
        # Generate the noise
        r_sig = np.random.multivariate_normal(
            np.zeros(npix), cov_m, size=use_Rs.shape[0])
    else:
        r_sig = np.random.normal(size=Rs.shape)

    # Truncate to 3 sigma
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)

    if perc is not None:
        use_Rs += (perc/100.) * use_Rs * r_sig
    elif isinstance(abs_sig, (float,int,np.ndarray)):
        use_Rs += r_sig * abs_sig
    #elif abs_sig  == 'PACE':
    #    if wave is None:
    #        raise ValueError("Need wavelength array for PACE noise.")
    #    # Add it in
    #    pace_sig = calc_pace_sig(wave)
    #    use_Rs += r_sig * pace_sig
    #elif abs_sig  == 'PACE_CORR':
    #    if wave is None:
    #        raise ValueError("Need wavelength array for PACE noise.")
    #    # Add it in
    #    pace_sig = calc_pace_sig(wave)
    #    use_Rs += r_sig * pace_sig
    #elif abs_sig  == 'PACE_TRUNC':
    #    if wave is None:
    #        raise ValueError("Need wavelength array for PACE noise.")
    #    pace_sig = calc_pace_sig(wave)
    #    # Boost the noise at the edges
    #    ok_wv = (wave > 380.) & (wave < 700.)
    #    pace_sig[~ok_wv] *= 100.   
    #    # Add it in
    #    use_Rs += r_sig * pace_sig
    else:
        raise ValueError("Bad abs_sig")
    
    # Return
    return use_Rs

def scale_noise(scl_noise, 
                model_Rrs:np.ndarray, 
                model_wave:np.ndarray,
                reduce_by_in_situ:float=None):
    """
    Calculate the scaled noise for the given model Rrs and wave.

    Parameters:
    scl_noise (str or float): The type of noise scaling to be applied. Can be one of 'SeaWiFS', 'MODIS_Aqua', 'PACE', or a float value.
    model_Rrs (np.ndarray): The model Rrs values.
    model_wave (np.ndarray): The wave values corresponding to the model Rrs.

    Returns:
    np.ndarray: The scaled noise values.

    """
    if scl_noise == 'SeaWiFS':
        err_dict = sat_seawifs.calc_errors()
        model_varRrs = np.array([err_dict[wv][0] for wv in sat_seawifs.seawifs_wave])**2
    elif scl_noise == 'MODIS_Aqua':
        err_dict = sat_modis.calc_errors(reduce_by_in_situ=reduce_by_in_situ)
        model_varRrs = np.array([err_dict[wv][0] for wv in sat_modis.modis_wave])**2
    elif scl_noise == 'PACE':
        PACE_error = sat_pace.gen_noise_vector(model_wave)
        model_varRrs = PACE_error**2
    elif scl_noise == 'SBG':
        SBG_error = sat_sbg.gen_noise_vector(model_wave)
        model_varRrs = SBG_error**2
    else:
        model_varRrs = (scl_noise * model_Rrs)**2

    # Return
    return model_varRrs