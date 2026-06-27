#!/bin/bash
# at -f at_fitting TIME
# Main
#cd /home/xavier/Oceanography/python/bing/papers/biomass/Analysis
#python py/end_to_end_workflow.py 6 > fitting.log 2>&1

# L23 Inelastic
cd /home/xavier/Oceanography/python/bing/papers/biomass/Analysis
python py/bbp_fit_l23.py 6 > L23_Inelastic.log 2>&1
