#!/usr/bin/env python

import glob
import argparse
import numpy as np
import pkg_resources

from astropy.table import Table

from measure_extinction.merge_obsspec import merge_stis_obsspec


def read_stis_archive_format(filename):
    """
    Read the STIS archive format and make it a *normal* table
    """
    t1 = Table.read(filename)
    ot = Table()
    for ckey in t1.colnames:
        if len(t1[ckey].shape) == 2:
            ot[ckey] = t1[ckey].data[0, :]

    return ot


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--waveregion",
        choices=["UV", "Opt"],
        default="Opt",
        help="wavelength region to merge",
    )
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Orig"),
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Out"),
    )
    parser.add_argument(
        "--ralph", action="store_true", help="Ralph Bohlin reduced data"
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.ralph:
        sfilename = "%s/%s/%s.mrg" % (args.inpath, args.waveregion, args.starname)

        # determine the line for the 1st data (can vary between files)
        f = open(sfilename, "r")
        k = 0
        for x in f:
            if "###" in x:
                dstart = k + 1
            else:
                k += 1
        f.close()

        stable = Table.read(
            sfilename,
            format="ascii",
            data_start=dstart,
            names=[
                "WAVELENGTH",
                "COUNT-RATE",
                "FLUX",
                "STAT-ERROR",
                "SYS-ERROR",
                "NPTS",
                "TIME",
                "QUAL",
            ],
        )
        stable = [stable]
    else:
        sfilename = f"{args.inpath}{args.starname}*_x1d.fits"
        print(sfilename)
        sfiles = glob.glob(sfilename)
        print(sfiles)
        stable = []
        for cfile in sfiles:
            print(cfile)
            t1 = read_stis_archive_format(cfile)
            t1.rename_column("ERROR", "STAT-ERROR")
            t1["NPTS"] = np.full((len(t1["FLUX"])), 1.0)
            t1["NPTS"][t1["FLUX"] == 0.0] = 0.0
            stable.append(t1)

    rb_stis_opt = merge_stis_obsspec(stable, waveregion=args.waveregion)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    stis_opt_file = "%s_stis_%s.fits" % (outname, args.waveregion)
    rb_stis_opt.write("%s/%s" % (args.outpath, stis_opt_file), overwrite=True)
