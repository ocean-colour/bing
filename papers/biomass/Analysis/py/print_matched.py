import pandas

# Locals
import biomass_io

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Fit Rrs')
    parser.add_argument("cruise", type=int, help="Table of input values.  Required: [wave,Rrs] Optional: [sigRrs,anw,bbnw] (.csv)")
    parser.add_argument("profile", type=int, help="Comma separate list of the a, bb models.  e.g. Exp,Cst")
    parser.add_argument("--match_file", type=str, help="Save outputs to this root")
    #parser.add_argument("--satellite", type=str, help="Simulate as if observed by the chosen satellite [Aqua, PACE]")
    #parser.add_argument("--fit_method", type=str, default='mcmc', help="Method for fitting [mcmc, chisq]")
    #parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")


    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs

def main(pargs):

    if pargs.match_file is None:
        match_file = 'matched_argo_bgc_profiles_bbp_v3.csv'

    # Grab it
    matched = pandas.read_csv(match_file)
    imatched = biomass_io.get_matched_profile(matched, pargs.cruise, pargs.profile)

    # Print
    print(imatched)


if __name__ == '__main__':
    args = parser()
    main(args)