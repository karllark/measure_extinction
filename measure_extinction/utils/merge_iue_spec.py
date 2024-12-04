import argparse

# from astropy.table import Table

# from measure_extinction.merge_obsspec import merge_iue_obsspec


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("iuedir", help="directory with IUE data for one star")
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default="./",
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default="./",
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # incomplete - code to read in the files needed
    # use_small capability needed
    # recalibration by Massa & Fitzpatrick IDL code needs to be recoded in python

    # iue_mergespec = merge_iue_obsspec(stable)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    iue_file = f"{outname}_iue.fits"
    # iue_mergespec.write(f"{args.outpath}/{iue_file}", overwrite=True)
