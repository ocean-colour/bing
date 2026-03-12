import os
import glob
import re
import numpy as np
import pandas as pd
import xarray as xr

# -------------------------
# load matchup CSV
# -------------------------
csv_path = os.path.expanduser("~/matched_argo_bgc_profiles_bbp_v2.csv")
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
files = sorted(set(glob.glob("*QC.nc") + glob.glob("*HRQC.nc")))

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
        depth = ds["Depth"]

        prof_dim, lev_dim = bbp.dims[:2]

        for (twmo, p) in targets:

            if twmo != wmo:
                continue

            bbp_prof = bbp.isel({prof_dim: p}).values.astype(float)
            depth_prof = depth.isel({prof_dim: p}).values.astype(float)

            mask = depth_prof <= 20

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
        "bbp700_top20m_median",
        "n_values_used"
    ]
)

out = os.path.expanduser("~/bbp700_top20m_matched_profiles.csv")
df.to_csv(out, index=False)

print("Saved to:", out)

