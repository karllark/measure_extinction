import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import astropy.units as u

from measure_extinction.stardata import StarData
from measure_extinction.utils.helpers import get_full_starfile


def plot_mspec_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="file with list of stars to plot")
    parser.add_argument("--path", help="path to star files", default="./")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    return parser


if __name__ == "__main__":

    # commandline parser
    parser = plot_mspec_parser()
    args = parser.parse_args()

    # get the names of the stars
    f = open(args.filelist, "r")
    file_lines = list(f)
    starnames = []
    for line in file_lines:
        if (line.find("#") != 0) & (len(line) > 0):
            name = line.rstrip()
            starnames.append(name)

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))

    # setup continuous colors
    color_indices = np.array(range(len(starnames))) / len(starnames)

    cmap = mpl.cm.plasma
    color = iter(cmap(np.linspace(0.1, 0.9, len(starnames))))

    # plot the bands and all spectra for this star
    # plot all the spectra on the same plot
    for k, cstarname in enumerate(starnames):
        c = next(color)
        fstarname, file_path = get_full_starfile(cstarname)
        starobs = StarData(fstarname, path=file_path)
        starobs.plot(ax, norm_wave_range=[0.3, 0.8] * u.micron, yoffset=2**k, pcolor=c,
                     legend_key="BAND", legend_label=cstarname)

    # finish configuring the plot
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel(r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    ax.legend(ncol=2)

    # use the whitespace better
    fig.tight_layout()

    # plot or save to a file
    save_str = "_spec"
    if args.png:
        fig.savefig(args.starname.replace(".dat", save_str + ".png"))
    elif args.pdf:
        fig.savefig(args.starname.replace(".dat", save_str + ".pdf"))
    else:
        plt.show()
