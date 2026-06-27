
import pandas
import numpy as np

import biomass_io

def table_for_robert(match_file:str, output_file:str='matched_L1B_for_robert.csv'):
    # Load up Argo profiles, already matched to PACE
    matched, _, _ = biomass_io.load_matched_data()

    # New table
    df = pandas.DataFrame()

    df['AOP_file'] = matched.closest_file

    L1B_files = []
    for ifile in df.AOP_file.values:
        # Modify
        l1b_file = ifile.replace('L2.OC_AOP', 'L1B')
        l1b_file = l1b_file.replace('V3_0', 'V3')
        # Append
        L1B_files.append(l1b_file)
    df['L1B_file'] = L1B_files    

    df['PACE_lat'] = matched.PACE_lat
    df['PACE_lon'] = matched.PACE_lon
    df['time'] = matched.time
    df['dist'] = matched.closest_dist_km

    # Cut
    good = np.isfinite(df.PACE_lat)

    # Write
    df[good].to_csv(output_file, index=False)
    print(f"Wrote: {output_file}")

# Command line
if __name__ == '__main__':
    matched_file = 'matched_argo_bgc_profiles_bbp.csv'

    # Robert
    table_for_robert(matched_file)