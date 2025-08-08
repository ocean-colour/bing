
import earthaccess

import pandas

from ocpy.utils import io as ocpy_io
from remote_sensing.download import earthaccess as rs_ea

def grab_em_all(outfile:str='PACE_50clouds.json', 
                cloud_cover=(0,50)):

    # Authorize
    auth = earthaccess.login(persist=True)

    # Grab em
    all_results = earthaccess.search_data(
        short_name="PACE_OCI_L2_AOP",
        cloud_cover=cloud_cover,
    )

    # Generat dict
    full_dict = rs_ea.granules_to_dict(all_results)

    # JSON
    jdict = ocpy_io.jsonify(full_dict)
    ocpy_io.savejson(outfile, jdict)
    print(f'Wrote {len(full_dict)} granules to {outfile}')

def load_from_json(json_file:str):
    # Load
    granules = ocpy_io.loadjson(json_file)

    # Build the table
    df = rs_ea.build_granule_table(granules, 
                                   fix_antimeridian=True)

    # Return
    return granules, df
        