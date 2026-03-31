# Fine closest PACE granules

# Overview

This function finds the closest PACE granule for each Argo profile in the given match file.
It then appends the results to the CSV file.

The function assumes that the match file has already been created by the match_argo_to_pace function.

The function uses the distance_from_latlon function to find the closest PACE granule.


# Code

- Use Python
- Provide inline comments in the code 

# Modifications

## Update from file

To speed up re-runs, update the function to read in a CSV file that contains the closest PACE granules for one or more of the Argo profiles in the match file.
The CSV file should have the following columns:
- cruise
- profile
- closest_id
- closest_file
- closest_dist_km
- closest_time

Use these values matched to cruise and profile to skip the matching process for those Argo profiles.

# Prompts

1. Read this doc and perform the Update from file Modification.