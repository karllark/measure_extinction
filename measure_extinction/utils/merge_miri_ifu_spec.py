#!/usr/bin/env python

import glob
import argparse
import numpy as np
import pkg_resources
import matplotlib.pyplot as plt

from astropy.table import QTable
import astropy.units as u

from measure_extinction.merge_obsspec import merge_miri_ifu_obsspec


fluxunit = u.erg / (u.s * u.cm * u.cm * u.angstrom)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
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
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    sfilename = f"{args.inpath}{args.starname}*order*.fits"
    print(sfilename)
    sfiles = glob.glob(sfilename)
    print(sfiles)
    stable = []
    for cfile in sfiles:
        print(cfile)
        cdata = QTable.read(cfile)
        print(cdata.colnames)
        cdata.rename_column("wavelength", "WAVELENGTH")
        cdata.rename_column("flux", "FLUX")
        cdata.rename_column("unc", "ERROR")
        cdata["WAVELENGTH"] *= u.micron
        cdata["NPTS"] = np.full((len(cdata["FLUX"])), 1.0)
        cdata["NPTS"][cdata["FLUX"] == 0.0] = 0.0
        stable.append(cdata)

    rb_mrs = merge_miri_ifu_obsspec(stable)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    mrs_file = f"{outname}_miri_ifu.fits"
    rb_mrs.write(f"{args.outpath}/{mrs_file}", overwrite=True)


    # plot the original and merged Spectra
    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.5))

    for ctable in stable:
        gvals = ctable["NPTS"] > 0
        ax.plot(
            ctable["WAVELENGTH"][gvals],
            ctable["FLUX"][gvals],
            "k-",
            alpha=0.5,
            label="orig",
        )
    gvals = rb_mrs["NPTS"] > 0
    cfluxes = rb_mrs["FLUX"][gvals].data * fluxunit
    # cfluxes = (rb_mrs["FLUX"][gvals].data * fluxunit).to(u.Jy, equivalencies=u.spectral_density(rb_mrs["WAVELENGTH"].data)).value
    ax.plot(
        rb_mrs["WAVELENGTH"][gvals].to(u.micron),
        cfluxes,
        "b-",
        alpha=0.5,
        label="merged",
    )

    ax.set_xlabel(r"$\lambda$ [$\AA$]")
    ax.set_ylabel(r"F($\lambda$)")

    ax.legend()
    fig.tight_layout()

    fname = mrs_file.replace(".fits", "")
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()