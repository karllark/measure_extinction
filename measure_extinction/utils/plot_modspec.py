import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from measure_extinction.stardata import StarData
from measure_extinction.utils.helpers import get_full_starfile
from measure_extinction.modeldata import ModelData


def plot_spec_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
    parser.add_argument("--path", help="path to star files", default="./")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    return parser


if __name__ == "__main__":

    # commandline parser
    parser = plot_spec_parser()
    args = parser.parse_args()

    # read in the observed data on the star
    fstarname, file_path = get_full_starfile("m33_j013334.26+303327")
    starobs = StarData(fstarname, path=file_path)
    band_names = starobs.data["BAND"].get_band_names()

    # get the model filenames
    print("reading in the model spectra")
    file_path = "/home/kgordon/Python_git/extstar_data/"
    tlusty_models_fullpath = glob.glob("{}/Models/tlusty_*v10.dat".format(file_path))
    tlusty_models_fullpath = tlusty_models_fullpath[0:10]
    tlusty_models = [
        tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
    ]

    spectra_names = ["BAND", "STIS"]
    # band_names = ['ACS_F814W', 'V', 'WFC3_F336W', 'WFC3_F160W',
    #               'ACS_F475W', 'WFC3_F110W', 'WFC3_F275W']
    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(
        tlusty_models,
        path="{}/Models/".format(file_path),
        band_names=band_names,
        spectra_names=spectra_names,
    )

    print(modinfo.waves["BAND"])

    # parmaeters (in fit order)
    fit_params = [4.41065817, 4.58342463, 0.20767662]

    # intrinsic sed
    modsed = modinfo.stellar_sed(fit_params[0:3], velocity=-100.0)

    # dust_extinguished sed
    # ext_modsed = modinfo.dust_extinguished_sed(fit_params[3:10], modsed)

    # hi_abs sed
    # hi_ext_modsed = modinfo.hi_abs_sed(fit_params[10:12], [args.velocity, 0.0],
    #                                    ext_modsed)

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

    # plot the bands and all spectra for this star
    for cspec in modinfo.fluxes.keys():
        if cspec == "BAND":
            ptype = "o"
        else:
            ptype = "-"

        print(modinfo.waves[cspec])

        ax.plot(modinfo.waves[cspec], modsed[cspec], "b" + ptype, label=cspec)
        # ax.plot(modinfo.waves[cspec], ext_modsed[cspec],
        #         'r'+ptype, label=cspec)
        # ax.plot(modinfo.waves[cspec], hi_ext_modsed[cspec],
        #         'g'+ptype, label=cspec)

    # finish configuring the plot
    # ax.set_ylim(8e4/norm_model, 2e9/norm_model)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel(r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # use the whitespace better
    fig.tight_layout()

    if args.png:
        fig.savefig(args.starname + "_mod_spec.png")
    else:
        plt.show()
