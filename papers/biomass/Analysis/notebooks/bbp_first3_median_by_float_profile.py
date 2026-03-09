import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr

files = sorted(set(glob.glob("*QC.nc") + glob.glob("*HRQC.nc")))

rows = []

def get_wmo(fname):
    m = re.match(r"(\d+)", fname)
    return m.group(1) if m else fname

for f in files:
    wmo = get_wmo(f)
    try:
        ds = xr.open_dataset(f)

        if "b_bp700" not in ds.data_vars:
            rows.append([wmo, None, f, np.nan, 0, "error: no b_bp700 var"])
            ds.close()
            continue

        bbp = ds["b_bp700"]  # dims: (N_PROF, N_LEVELS)

        prof_dim, lev_dim = bbp.dims[:2]
        nprof = bbp.sizes[prof_dim]

        for p in range(nprof):
            vals = bbp.isel({prof_dim: p, lev_dim: slice(0, 3)}).values.astype(float).ravel()
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                rows.append([wmo, p, f, np.nan, 0, "no finite vals"])
                continue

            med = float(np.median(vals))
            rows.append([wmo, p, f, med, len(vals), "ok"])

        ds.close()

    except Exception as e:
        rows.append([wmo, None, f, np.nan, 0, f"error: {e}"])

df = pd.DataFrame(rows, columns=["wmo","profile","file","bbp700_first3_median","n_used","status"])

out = os.path.expanduser("~/bbp_first3_by_float_profile_median.csv")
df.to_csv(out, index=False)
print("Saved to:", out)

