import argparse
import glob

import numpy as np
import astropy.units as u
from astropy.table import QTable
import matplotlib.pyplot as plt

from measure_extinction.merge_obsspec import merge_iue_obsspec


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
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # incomplete - code to read in the files needed
    # use_small capability needed
    # recalibration by Massa & Fitzpatrick IDL code needs to be recoded in python

    stable = []
    sfilename = f"{args.iuedir}/*.mxlo"
    sfiles = glob.glob(sfilename)
    for cfile in sfiles:
        print(cfile)
        tt1 = QTable.read(cfile, format="fits")
        t1 = QTable()

        minwave = tt1["WAVELENGTH"][0].value
        dwave = tt1["DELTAW"][0].value
        nwave = tt1["NPOINTS"][0]
        maxwave = minwave + nwave * dwave
        t1["WAVELENGTH"] = np.arange(minwave, maxwave, dwave) * u.angstrom
        t1["FLUX"] = tt1["FLUX"][0].value * u.erg / (u.cm * u.cm * u.s * u.angstrom)
        t1["ERROR"] = tt1["SIGMA"][0].value * u.erg / (u.cm * u.cm * u.s * u.angstrom)
        t1["NPTS"] = np.full((len(t1["FLUX"])), 1.0)
        nvals = t1["FLUX"] == 0.0
        if np.sum(nvals):
            t1["NPTS"][nvals] = 0.0
        stable.append(t1)

    rb_iue = merge_iue_obsspec(stable)
    if args.outname:
        outname = args.outname
    else:
        outname = args.starname.lower()
    iue_file = f"{outname}_iue.fits"
    rb_iue.write(f"{args.outpath}/{iue_file}", overwrite=True)

    # plot the original and merged Spectra
    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.5))

    gvals = stable[0]["NPTS"] > 0
    for ctable in stable:
        ax.plot(
            ctable["WAVELENGTH"][gvals],
            ctable["FLUX"][gvals],
            "k-",
            alpha=0.5,
            label="orig",
        )
    gvals = rb_iue["NPTS"] > 0
    ax.plot(
        rb_iue["WAVELENGTH"][gvals],
        rb_iue["FLUX"][gvals],
        "b-",
        alpha=0.5,
        label="merged",
    )

    # set min/max ignoring Ly-alpha as it often has a strong core emission
    gvals = (rb_iue["WAVELENGTH"] > 1300.0) & (rb_iue["NPTS"] > 0)
    miny = np.nanmin(rb_iue["FLUX"][gvals])
    maxy = np.nanmax(rb_iue["FLUX"][gvals])
    delt = maxy - miny
    ax.set_ylim(miny - 0.2 * delt, maxy + 0.2 * delt)

    ax.set_xlabel(r"$\lambda$ [$\AA$]")
    ax.set_ylabel(r"F($\lambda$)")

    ax.legend()
    fig.tight_layout()

    fname = iue_file.replace(".fits", "")
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()