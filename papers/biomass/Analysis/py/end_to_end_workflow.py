# End to end workflow for the biomass paper

import os
import argo

import pandas

import grab_pace_granules

from IPython import embed

def slurp_argo():
    """ Slurp Argo data """

    # MBARI
    argo_path = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'SOCCOM_GO-BGC_LoResQC_LIAR_26Jun2025_netcdf')
    mbari = argo.scan_mbari_profiles(argo_path=argo_path)

    # Mediterranean
    argo_file = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'Med_Mexico',
                             'Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_Mediterranean.nc')
    med = argo.scan_ocean_bio_profiles(argo_file)

    # Gulf of Mexico
    argo_file = os.path.join(os.getenv('OS_DATA'), 
                             'Argo', 
                             'Med_Mexico',
                             'Ocean_Biogeochemistry_BGC-Argo_Global_Profiles_GulfofMexico.nc')
    gulf = argo.scan_ocean_bio_profiles(argo_file)


    # Combine
    df = pandas.concat([mbari, med, gulf], ignore_index=True)
    df.to_csv('argo_bgc_profiles_bbp.csv', index=False)
    print(f'Wrote {len(df)} profiles to argo_bgc_profiles_bbp.csv')


def main(flg):
    flg= int(flg)

    # Slurp all of the Argo profiles
    if flg == 1:
        slurp_argo()

    # Build the JSON file for PACE granules
    if flg == 2:
        granule_file = 'PACE_50clouds_v31.json'
        grab_pace_granules.build_json(outfile=granule_file, 
            cloud_cover=(0,50))
    
    # Match Argo to PACE
    if flg == 3:
        out_file='matched_argo_bgc_profiles_bbp_v2.csv'
        granule_file = 'PACE_50clouds_v31.json'
        argo.match_argo_to_pace(granule_file, out_file, dtime='1 day')
    
    # Grab PACE AOP granules
    if flg == 4:
        match_file='matched_argo_bgc_profiles_bbp_v2.csv'
        granule_file = 'PACE_50clouds_v31.json'
        grab_pace_granules.download_matched(
            match_file, granule_file, IOP=False, L1B=False)

    # Find closest PACE granules (with good Rrs)
    if flg == 5:
        match_file='matched_argo_bgc_profiles_bbp_v2.csv'
        granule_file = 'PACE_50clouds_v31.json'
        grab_pace_granules.find_closest(
            match_file, granule_file, iRrs=38, 
            debug=True)

    # Fit PACE

# Command line
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)
    