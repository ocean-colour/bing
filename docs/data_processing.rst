.. _data_processing:

===============
Data Processing
===============

BING provides comprehensive tools for processing ocean color remote sensing data,
with special emphasis on NASA's PACE mission and integration with Argo BGC floats.

PACE Data Processing
--------------------

Loading PACE OCI Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ocpy.pace import io as pace_io
    import os
    
    # Load PACE L2 file
    pace_file = os.path.join(
        os.getenv('OS_COLOR'), 
        'PACE/L2_AOP/PACE_OCI.20240315T183000.L2.OC_AOP.V2.nc'
    )
    
    # Load with quality flags
    xds, flags = pace_io.load_oci_l2(pace_file)
    
    # Available variables
    print(xds.data_vars.keys())
    # Rrs, Rrs_unc, chlor_a, poc, ...

Extracting Rrs Spectra
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    
    def extract_pace_rrs(xds, lat, lon, window=3):
        """Extract Rrs spectrum at a location with spatial averaging
        
        Parameters
        ----------
        xds : xarray.Dataset
            PACE L2 dataset
        lat, lon : float
            Target coordinates
        window : int
            Spatial window size (pixels)
        
        Returns
        -------
        dict
            Wavelengths, Rrs, uncertainties
        """
        # Find nearest pixel
        idx_lat = np.argmin(np.abs(xds.latitude.values - lat))
        idx_lon = np.argmin(np.abs(xds.longitude.values - lon))
        
        # Define window
        lat_slice = slice(max(0, idx_lat-window), idx_lat+window+1)
        lon_slice = slice(max(0, idx_lon-window), idx_lon+window+1)
        
        # Extract and average
        rrs_window = xds.Rrs[lat_slice, lon_slice, :]
        rrs_mean = rrs_window.mean(dim=['latitude', 'longitude'])
        rrs_std = rrs_window.std(dim=['latitude', 'longitude'])
        
        # Get uncertainties
        unc_window = xds.Rrs_unc[lat_slice, lon_slice, :]
        unc_mean = np.sqrt((unc_window**2).mean(dim=['latitude', 'longitude']))
        
        return {
            'wavelength': xds.wavelength.values,
            'Rrs': rrs_mean.values,
            'Rrs_unc': unc_mean.values,
            'Rrs_std': rrs_std.values,
            'n_pixels': np.sum(~np.isnan(rrs_window.values))
        }

Quality Control
~~~~~~~~~~~~~~~

.. code-block:: python

    def quality_control_pace(xds, flags):
        """Apply quality control to PACE data
        
        Parameters
        ----------
        xds : xarray.Dataset
            PACE L2 dataset
        flags : dict
            Quality flags from pace_io.load_oci_l2
        
        Returns
        -------
        xarray.Dataset
            Masked dataset
        """
        # Define flag bits to check
        flag_bits = {
            'ATMFAIL': 1,      # Atmospheric correction failure
            'LAND': 2,         # Land
            'CLOUD': 4,        # Cloud
            'GLINT': 8,        # Sun glint
            'HISOLZEN': 256,   # High solar zenith
            'LOWLW': 512,      # Low water-leaving radiance
            'CHLFAIL': 1024    # Chlorophyll algorithm failure
        }
        
        # Create mask
        mask = np.zeros_like(flags['l2_flags'], dtype=bool)
        for flag_name, bit_val in flag_bits.items():
            mask |= (flags['l2_flags'] & bit_val) > 0
        
        # Apply mask
        xds_clean = xds.where(~mask)
        
        return xds_clean

Argo BGC Integration
--------------------

Matching Satellite and Float Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from datetime import timedelta
    
    def match_argo_pace(argo_profiles, pace_granules, 
                        max_time_diff=3, max_distance=25):
        """Match Argo profiles with PACE granules
        
        Parameters
        ----------
        argo_profiles : pd.DataFrame
            Argo profile locations and times
        pace_granules : list
            PACE granule metadata
        max_time_diff : float
            Maximum time difference in hours
        max_distance : float
            Maximum distance in km
        
        Returns
        -------
        pd.DataFrame
            Matched profiles with granule information
        """
        matches = []
        
        for idx, profile in argo_profiles.iterrows():
            for granule in pace_granules:
                # Time check
                time_diff = abs(profile['datetime'] - granule['datetime'])
                if time_diff > timedelta(hours=max_time_diff):
                    continue
                
                # Distance check
                distance = haversine_distance(
                    profile['lat'], profile['lon'],
                    granule['center_lat'], granule['center_lon']
                )
                
                if distance <= max_distance:
                    matches.append({
                        'profile_id': profile['id'],
                        'granule_id': granule['id'],
                        'time_diff': time_diff.total_seconds() / 3600,
                        'distance': distance,
                        'profile_lat': profile['lat'],
                        'profile_lon': profile['lon']
                    })
        
        return pd.DataFrame(matches)

Processing Argo Profiles
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def process_argo_bgc(profile_file):
        """Process Argo BGC profile data
        
        Parameters
        ----------
        profile_file : str
            Path to Argo profile netCDF file
        
        Returns
        -------
        dict
            Processed profile data
        """
        import xarray as xr
        
        # Load profile
        ds = xr.open_dataset(profile_file)
        
        # Extract BGC variables
        data = {
            'pressure': ds.PRES.values,
            'temperature': ds.TEMP.values,
            'salinity': ds.PSAL.values,
            'chlorophyll': ds.CHLA.values if 'CHLA' in ds else None,
            'bbp': ds.BBP700.values if 'BBP700' in ds else None,
            'oxygen': ds.DOXY.values if 'DOXY' in ds else None
        }
        
        # Quality control
        for var in ['chlorophyll', 'bbp', 'oxygen']:
            if data[var] is not None:
                qc_var = f'{var}_QC'
                if qc_var in ds:
                    # Keep only good data (QC flags 1, 2)
                    mask = ds[qc_var].values > 2
                    data[var][mask] = np.nan
        
        # Calculate derived variables
        if data['bbp'] is not None:
            # Estimate POC from bbp
            data['poc'] = 31600 * data['bbp'] - 1.18  # Stramski et al. 2008
        
        return data

Data Organization
-----------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

Recommended directory structure for BING projects:

.. code-block:: text

    project/
    ├── data/
    │   ├── pace/
    │   │   ├── L2_AOP/         # PACE Level 2 files
    │   │   └── matchups/       # Matched PACE-Argo data
    │   ├── argo/
    │   │   ├── profiles/       # Argo profile files
    │   │   └── metadata/       # Profile metadata
    │   └── auxiliary/
    │       ├── bathymetry/     # Bathymetry data
    │       └── climatology/    # Climatological data
    ├── results/
    │   ├── fits/              # Fitting results
    │   ├── figures/           # Generated plots
    │   └── statistics/        # Statistical analyses
    └── notebooks/             # Jupyter notebooks

Data Management
~~~~~~~~~~~~~~~

.. code-block:: python

    class BingDataManager:
        """Centralized data management for BING workflows"""
        
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.pace_dir = os.path.join(base_dir, 'data/pace')
            self.argo_dir = os.path.join(base_dir, 'data/argo')
            self.results_dir = os.path.join(base_dir, 'results')
            
            # Create directories if needed
            for dir_path in [self.pace_dir, self.argo_dir, self.results_dir]:
                os.makedirs(dir_path, exist_ok=True)
        
        def get_pace_file(self, date, product='L2_AOP'):
            """Get PACE file path for a given date"""
            pattern = f"PACE_OCI.{date:%Y%m%d}*.{product}.nc"
            files = glob.glob(os.path.join(self.pace_dir, product, pattern))
            return files[0] if files else None
        
        def save_results(self, results, identifier):
            """Save processing results with metadata"""
            timestamp = datetime.now().isoformat()
            results['metadata'] = {
                'identifier': identifier,
                'timestamp': timestamp,
                'version': bing.__version__
            }
            
            outfile = os.path.join(
                self.results_dir, 
                f"{identifier}_{timestamp[:10]}.npz"
            )
            np.savez(outfile, **results)
            return outfile

Batch Processing
----------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from joblib import Parallel, delayed
    import multiprocessing
    
    def process_pace_granule(granule_file, config):
        """Process single PACE granule"""
        xds, flags = pace_io.load_oci_l2(granule_file)
        # ... processing logic ...
        return results
    
    # Parallel processing
    n_cores = multiprocessing.cpu_count() - 1
    
    results = Parallel(n_jobs=n_cores)(
        delayed(process_pace_granule)(f, config) 
        for f in granule_files
    )

Pipeline Example
~~~~~~~~~~~~~~~~

.. code-block:: python

    def bing_processing_pipeline(pace_file, argo_file, output_dir):
        """Complete BING processing pipeline
        
        Parameters
        ----------
        pace_file : str
            PACE L2 file path
        argo_file : str
            Argo profile file path
        output_dir : str
            Output directory
        """
        # Step 1: Load and QC PACE data
        xds, flags = pace_io.load_oci_l2(pace_file)
        xds_clean = quality_control_pace(xds, flags)
        
        # Step 2: Load Argo profile
        argo_data = process_argo_bgc(argo_file)
        
        # Step 3: Extract matchup
        rrs_data = extract_pace_rrs(
            xds_clean, 
            argo_data['lat'], 
            argo_data['lon']
        )
        
        # Step 4: Fit models
        from bing.parameters import standard
        from bing.models import utils as model_utils
        from bing.fitting import chisq_fit
        
        params = standard.expb_pow()
        models = model_utils.init(
            params.model_names, 
            rrs_data['wavelength']
        )
        
        fit_result = chisq_fit.fit(
            models,
            rrs_data['wavelength'],
            rrs_data['Rrs'],
            rrs_data['Rrs_unc']
        )
        
        # Step 5: Save results
        output = {
            'pace_data': rrs_data,
            'argo_data': argo_data,
            'fit_result': fit_result,
            'metadata': {
                'pace_file': pace_file,
                'argo_file': argo_file,
                'processing_date': datetime.now().isoformat()
            }
        }
        
        outfile = os.path.join(output_dir, 'pipeline_results.npz')
        np.savez(outfile, **output)
        
        return output

Data Formats
------------

Input Formats
~~~~~~~~~~~~~

BING supports various input data formats:

* **NetCDF**: PACE L2, Argo profiles
* **CSV**: Matchup tables, metadata
* **NPZ**: Numpy compressed arrays
* **HDF5**: Large dataset storage
* **JSON**: Configuration and metadata

Output Formats
~~~~~~~~~~~~~~

Standard output formats:

.. code-block:: python

    # Fitting results format
    fit_results = {
        'wavelength': np.array([...]),      # Wavelengths
        'Rrs': np.array([...]),             # Measured Rrs
        'Rrs_unc': np.array([...]),         # Uncertainties
        'fitted_params': np.array([...]),   # Best-fit parameters
        'chains': np.array([...]),          # MCMC chains
        'statistics': {                     # Fit statistics
            'chi2': float,
            'rmse': float,
            'r2': float
        },
        'metadata': {                       # Processing metadata
            'timestamp': str,
            'version': str,
            'config': dict
        }
    }

Best Practices
--------------

1. **Quality Control**: Always apply quality flags to satellite data
2. **Spatial Averaging**: Use spatial windows to reduce noise
3. **Temporal Matching**: Consider time differences for matchups
4. **Data Organization**: Maintain consistent directory structure
5. **Metadata**: Always save processing metadata with results
6. **Version Control**: Track data and code versions
7. **Validation**: Compare with in-situ measurements when available
