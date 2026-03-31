import os
import numpy as np
import pandas as pd
import xarray as xr

VAR_BBP     = "Particle_backscattering_at_700_nm_adjusted_"
VAR_BBP_QC  = "Particle_backscattering_at_700_nm_adjusted__qc"
VAR_DEPTH   = "Pressure_adjusted_"
VAR_WMO     = "Platform_Number"
DIM_PROFILE = "N_STATIONS"


def process_profile(prof, min_surface=3, surface_depth=20, qc_threshold=50):
    try:
        good = prof[VAR_BBP_QC].data <= qc_threshold
        if not np.any(good):
            return None

        depth = prof[VAR_DEPTH].data[good]
        bbp   = prof[VAR_BBP].data[good]

        if np.sum(depth < surface_depth) < min_surface:
            return None

        mask = depth <= surface_depth
        vals = bbp[mask]
        if len(vals) == 0:
            return None

        wmo      = str(prof[VAR_WMO].data.astype(str)).strip()
        filename = f"{wmo}QC.nc"

        return {
            "file":                 filename,
            "bbp700_top20m_median": float(np.median(vals)),
            "n_values_used":        len(vals),
        }

    except Exception as e:
        return None


def process_file(path, **kwargs):
    rows = []
    print("Processing:", os.path.basename(path))

    ds = xr.open_dataset(path)
    n_profiles = ds.dims[DIM_PROFILE]

    for i in range(n_profiles):
        prof   = ds.isel(**{DIM_PROFILE: i})
        result = process_profile(prof, **kwargs)
        # Add profile number
        result["profile"] = i
        if result is not None:
            rows.append(result)

    ds.close()
    print(f"  -> {len(rows)} valid profiles found")
    return rows


def process_files(nc_files, out_path=None, **kwargs):
    all_rows = []
    for path in nc_files:
        all_rows.extend(process_file(path, **kwargs))

    df = pd.DataFrame(all_rows, columns=["file", "bbp700_top20m_median", "n_values_used"])

    if out_path:
        df.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")

    print(f"Total rows: {len(df)}")
    return df

# Command line interface
if __name__ == '__main__':
    nc_files = [
        os.path.join(base, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_GulfofMexico.nc"),
        os.path.join(base, "Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_Mediterranean.nc")
    ]

