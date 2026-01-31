
import pandas

def table_for_robert(match_file:str, output_file:str='matched_L1B_for_robert.csv'):
    # Load up Argo profiles, already matched to PACE
    matched = pandas.read_csv(match_file)

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

    df['lat'] = matched.lat
    df['lon'] = matched.lon
    df['time'] = matched.time

    # Write
    df.to_csv(output_file, index=False)
    print(f"Wrote: {output_file}")

# Command line
if __name__ == '__main__':
    matched_file = 'matched_argo_bgc_profiles_bbp.csv'

    # Robert
    table_for_robert(matched_file)