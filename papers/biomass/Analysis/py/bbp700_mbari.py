import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr


def calc_bbp700_mbari(argo_dir:str=None, csv_path:str=None,
    out_path:str=None, surface_depth: float = 25.0) -> pd.DataFrame:

    if argo_dir is None:
        argo_dir = os.path.join(os.getenv('OS_DATA'), 'Argo',
            'SOCCOM_GO-BGC_LoResQC_LIAR_26Jun2025_netcdf')

    # -------------------------
    # load matchup CSV
    # -------------------------
    if csv_path is None:
        csv_path = "matched_argo_bgc_profiles_bbp_v3.csv"
    match = pd.read_csv(csv_path)

    # build list of (wmo, profile)
    targets = set()

    for _, row in match.iterrows():

        fname = str(row["filename"])
        profile = int(row["profile"])

        m = re.search(r"\d{7}", fname)
        if m:
            wmo = m.group(0)
            targets.add((wmo, profile))

    target_wmos = set(w for w, p in targets)

    print("Matched floats:", len(target_wmos))
    print("Matched profiles:", len(targets))

    # -------------------------
    # find Argo files
    # -------------------------
    files = sorted(set(glob.glob(os.path.join(argo_dir, "*QC.nc")) + 
        glob.glob(os.path.join(argo_dir, "*HRQC.nc"))))

    rows = []

    def get_wmo(fname):
        m = re.match(r"(\d+)", fname)
        return m.group(1) if m else None

    # -------------------------
    # process floats
    # -------------------------
    for f in files:

        wmo = get_wmo(f)

        if wmo not in target_wmos:
            continue

        try:

            ds = xr.open_dataset(f)

            if "b_bp700" not in ds.data_vars:
                ds.close()
                continue

            bbp = ds["b_bp700"]
            bbp_qc = ds["b_bp700_QF"]
            depth = ds["Depth"]

            prof_dim, lev_dim = bbp.dims[:2]

            for (twmo, p) in targets:

                # Double check
                if twmo != wmo:
                    raise ValueError(f"WMO mismatch: {twmo} != {wmo}")

                bbp_prof = bbp.isel({prof_dim: p}).values.astype(float)
                qc_prof = bbp_qc.isel({prof_dim: p}).values.astype(float)
                depth_prof = depth.isel({prof_dim: p}).values.astype(float)

                # QC
                good = qc_prof == b'0'
                if not np.any(good):
                    raise ValueError(f"No good values for profile {p}")

                # Depth
                mask = depth_prof <= surface_depth

                mask = mask & good
                if np.sum(mask) < 4:
                    raise ValueError(f"Not enough good values for profile {p}")

                # Calculate
                vals = bbp_prof[mask]
                vals = vals[np.isfinite(vals)]

                if len(vals) > 0:
                    med = float(np.median(vals))
                    n_used = len(vals)
                else:
                    med = np.nan
                    n_used = 0

                rows.append([wmo, p, f, med, n_used])

            ds.close()

        except Exception as e:
            print("Skipping", f, e)


    # -------------------------
    # save output
    # -------------------------
    df = pd.DataFrame(
        rows,
        columns=[
            "wmo",
            "profile",
            "file",
            "bbp700_top25m_median",
            "n_values_used"
        ]
    )

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print("Saved to:", out_path)

# Command line
# 
# 
if __name__ == "__main__":
    calc_bbp700_mbari()

