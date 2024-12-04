import argparse
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pandas as pd

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.stardata import StarData


def plot_HI(path, ax):
    """
    Indicate the HI-lines on the plot (between 912A and 10 micron)

    Parameters
    ----------
    ax : AxesSubplot
        Axes of plot for which to add the HI-lines

    Returns
    -------
    Indicates HI-lines on the plot
    """
    # read in HI-lines
    table = pd.read_table(f"{path}/HI_lines.list", sep=r"\s+", comment="#")
    # group lines by series
    series_groups = table.groupby("n'")
    colors = plt.get_cmap("tab10")
    series_names = {
        1: "Ly",
        2: "Ba",
        3: "Pa",
        4: "Br",
        5: "Pf",
        6: "Hu",
        7: "7",
        8: "8",
        9: "9",
        10: "10",
    }
    for name, series in series_groups:
        # plot the lines
        for wave in series.wavelength:
            ax.axvline(wave, color=colors(name - 1), lw=0.05, alpha=0.4)
        # add the name of the series
        ax.text(
            series.wavelength.mean(),
            0.04,
            series_names[name],
            transform=ax.get_xaxis_transform(),
            color=colors(name - 1),
        ).set_clip_on(True)


def zoom(ax, range):
    """
    Zoom in on the requested wavelength range by setting the axes limits to this range

    Parameters
    ----------
    ax : AxesSubplot
        Axes of plot for which new limits need to be set

    range : list of 2 floats
        Wavelength range to be plotted [min, max]
        in micron when plotting wavelengths or in 1/micron when plotting wavenumber

    Returns
    -------
    Sets the axes limits to the requested range
    """
    # set the x axis limits
    ax.set_xlim(range)

    # calculate the appropriate y axis limits
    ymin, ymax = np.inf, -np.inf
    for line in ax.get_lines():
        x_data = line.get_xdata()
        # skip the plotted HI-lines to calculate the y axis limits (because those have y_data in axes coordinates)
        if x_data[0] == x_data[1]:
            continue
        y_data = line.get_ydata()[
            np.logical_and(x_data >= range[0], x_data <= range[1])
        ]
        if y_data.size != 0 and np.nanmin(y_data) < ymin:
            ymin = np.nanmin(y_data)
        if y_data.size != 0 and np.nanmax(y_data) > ymax:
            ymax = np.nanmax(y_data)
    ax.set_ylim(ymin * 0.95, ymax * 1.05)


def plot_multi_spectra(
    starlist,
    path,
    mlam4=False,
    HI_lines=False,
    range=None,
    norm_range=None,
    spread=False,
    exclude=[],
    log=False,
    class_offset=True,
    text_offsets=[],
    text_angles=[],
    multicolor=False,
    wavenum=False,
    deredden=False,
    pdf=False,
    rebin_res=None,
    outname="all_spec.pdf",
):
    """
    Plot the observed band and spectral data of multiple stars in the same plot

    Parameters
    ----------
    starlist : list of strings
        List of stars for which to plot the spectrum

    path : string
        Path to the data files

    mlam4 : boolean [default=False]
        Whether or not to multiply the flux F(lambda) by lambda^4 to remove the Rayleigh-Jeans slope

    HI_lines : boolean [default=False]
        Whether or not to indicate the HI-lines in the plot

    range : list of 2 floats [default=None]
        Wavelength range to be plotted [min, max]
        in micron when plotting wavelengths or in 1/micron when plotting wavenumber

    norm_range : list of 2 floats [default=None]
        Wavelength range to use to normalize the data (in micron)- [min,max]

    spread : boolean [default=False]
        Whether or not to spread the spectra out by adding a vertical offset to each spectrum

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., IRS)

    log : boolean [default=False]
        Whether or not to plot the wavelengths on a log-scale

    class_offset : boolean [default=True]
        Whether or not to add an extra offset between main sequence and giant stars (only relevant when spread=True; this only works when the stars are sorted by spectral class, i.e. first the main sequence and then the giant stars)

    text_offsets : list of floats [default=[]]
        List of the same length as starlist with offsets for the annotated text

    text_angles : list of integers [default=[]]
        List of the same length as starlist with rotation angles for the annotated text

    multicolor : boolean [default=False]
        Whether or not to give all spectra a different color

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    deredden : boolean [default=False]
       Deredden the data based on dereddening parameters given in the DAT file.
       Generally used to deredden standards.

    rebin_res : float [default=None]
        Spectral resolution for rebinning spectra

    outname : string [default="all_spec.pdf"]
        Name for the output pdf file

    Returns
    -------
    Figure with band data points and spectra of multiple stars
    """
    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # create the plot
    fig, ax = plt.subplots(figsize=(15, len(starlist) * 1.25))
    colors = plt.get_cmap("tab10")

    if norm_range is not None:
        norm_range = norm_range * u.micron

    # set default text offsets and angles
    if text_offsets == []:
        text_offsets = np.full(len(starlist), 0.2)
    if text_angles == []:
        text_angles = np.full(len(starlist), 10)

    for i, star in enumerate(starlist):
        # read in all bands and spectra for this star
        starobs = StarData(
            "%s.dat" % star, path=path, use_corfac=True, deredden=deredden
        )

        # rebin spectra if desired
        if rebin_res is not None:
            gkeys = list(starobs.data.keys())
            gkeys.remove("BAND")
            for src in gkeys:
                starobs.data[src].rebin_constres(
                    starobs.data[src].wave_range, rebin_res
                )

        # spread out the spectra if requested
        # add extra whitespace when the luminosity class changes from main sequence to giant
        if spread:
            extra_off = 0
            if "V" not in starobs.sptype and class_offset:
                extra_off = 1
            yoffset = extra_off + 0.5 * i
        else:
            yoffset = 0

        # determine where to add the name of the star and its spectral type
        # find the shortest plotted wavelength, and give preference to spectral data when available
        exclude2 = []
        if "BAND" in starobs.data.keys() and len(starobs.data.keys()) > 1:
            exclude2 = ["BAND"]
        (waves, fluxes, flux_uncs) = starobs.get_flat_data_arrays(
            starobs.data.keys() - (exclude + exclude2)
        )
        if wavenum:
            x = 1 / waves
        else:
            x = waves
        if range is not None:
            x = x[x >= range[0]]
        min_x = x[0]
        # find out which data type corresponds with this wavelength
        for data_type in starobs.data.keys():
            if data_type in exclude:
                continue
            if wavenum:
                cx = 1.0 / starobs.data[data_type].waves
            else:
                cx = starobs.data[data_type].waves
            used_x = cx[starobs.data[data_type].npts > 0]
            if min_x in used_x.value:
                ann_key = data_type
        ann_range = [min_x, min_x] * u.micron

        # plot the spectrum
        if multicolor:
            pcolor = colors(i % 10)
        else:
            pcolor = "k"
        starobs.plot(
            ax,
            pcolor=pcolor,
            norm_wave_range=norm_range,
            mlam4=mlam4,
            wavenum=wavenum,
            exclude=exclude,
            yoffset=yoffset,
            yoffset_type="add",
            annotate_key=ann_key,
            annotate_wave_range=ann_range,
            annotate_text=star.upper() + "  " + starobs.sptype,
            annotate_yoffset=text_offsets[i],
            annotate_rotation=text_angles[i],
            annotate_color=pcolor,
        )

    # plot HI-lines if requested
    if HI_lines:
        plot_HI(path, ax)

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    if not spread:
        ax.set_yscale("log")
    if log:
        ax.set_xscale("log")
    if wavenum:
        xlab = r"$1/\lambda$ [$\mu m^{-1}$]"
    else:
        xlab = r"$\lambda$ [$\mu m$]"
    ax.set_xlabel(xlab, fontsize=1.5 * fontsize)
    ylabel = r"$F(\lambda)$"

    if norm_range is not None:
        if norm_range[0].unit == "micron":
            units = r"$\mu m$"
        else:
            units = norm_range[0].unit
        ylabel += "/$F$(" + str(int(np.mean(norm_range).value)) + units + ")"
    else:
        ylabel += r" [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}$]"
    if mlam4:
        ylabel = r"$\lambda^4$" + ylabel.replace("]", r" $\mu m^4$]")
        outname = outname.replace("spec", "spec_mlam4")
    if spread:
        ylabel += " + offset"
    ax.set_ylabel(ylabel, fontsize=1.5 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname, bbox_inches="tight")
    else:
        plt.show()

    # return the figure and axes for additional manipulations
    return fig, ax


def plot_spectrum(
    star,
    path,
    mlam4=False,
    HI_lines=False,
    range=None,
    norm_range=None,
    exclude=[],
    log=False,
    wavenum=False,
    deredden=False,
    rebin_res=None,
    pdf=False,
):
    """
    Plot the observed band and spectral data of a star

    Parameters
    ----------
    star : string
        Name of the star for which to plot the spectrum

    path : string
        Path to the data files

    mlam4 : boolean [default=False]
        Whether or not to multiply the flux F(lambda) by lambda^4 to remove the Rayleigh-Jeans slope

    HI_lines : boolean [default=False]
        Whether or not to indicate the HI-lines in the plot

    range : list of 2 floats [default=None]
        Wavelength range to be plotted - [min,max]

    norm_range : list of 2 floats [default=None]
        Wavelength range to use to normalize the data (in micron)- [min,max]

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., IRS)

    log : boolean [default=False]
        Whether or not to plot the wavelengths on a log-scale

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    deredden : boolean [default=False]
       Deredden the data based on dereddening parameters given in the DAT file.
       Generally used to deredden standards.

    rebin_res : float [default=None]
        Spectral resolution for rebinning spectra

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with band data points and spectrum
    """
    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # read in and plot all bands and spectra for this star
    if ".dat" not in star:
        fname = f"{star}.dat"
    else:
        fname = star
    starobs = StarData(fname, path=path, use_corfac=True, deredden=deredden)
    if norm_range is not None:
        norm_range = norm_range * u.micron
    # rebin spectra if desired
    if rebin_res is not None:
        gkeys = list(starobs.data.keys())
        gkeys.remove("BAND")
        for src in gkeys:
            starobs.data[src].rebin_constres(starobs.data[src].wave_range, rebin_res)

    starobs.plot(
        ax,
        norm_wave_range=norm_range,
        mlam4=mlam4,
        exclude=exclude,
        wavenum=wavenum,
    )
    # plot HI-lines if requested
    if HI_lines:
        plot_HI(path, ax)

    # define the output name
    outname = star + "_spec.pdf"

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    if log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(star.upper(), fontsize=50)
    if wavenum:
        xlab = r"$1/\lambda$ [$\mu m^{-1}$]"
    else:
        xlab = r"$\lambda$ [$\mu m$]"
    ax.set_xlabel(xlab, fontsize=1.5 * fontsize)
    if mlam4:
        ax.set_ylabel(
            r"$F(\lambda)\ \lambda^4$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}\ \mu m^4$]",
            fontsize=1.5 * fontsize,
        )
        outname = outname.replace("spec", "spec_mlam4")
    else:
        ax.set_ylabel(
            r"$F(\lambda)$ [$ergs\ cm^{-2}\ s^{-1}\ \AA^{-1}$]",
            fontsize=1.5 * fontsize,
        )
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # return the figure and axes for additional manipulations
    return fig, ax


def main():
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "starlist",
        nargs="+",
        help="star name or list of star names for which to plot the spectrum",
    )
    parser.add_argument(
        "--path",
        help="path to the data files",
        default=get_datapath(),
    )
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument("--HI_lines", help="indicate the HI-lines", action="store_true")
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all spectra in the same figure",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--norm_range",
        nargs="+",
        help="wavelength range to use to normalize the spectrum (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread",
        help="spread the spectra out over the figure; can only be used in combination with --onefig",
        action="store_true",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="data type(s) to be excluded from the plot",
        type=str,
        default=[],
    )
    parser.add_argument(
        "--log", help="plot wavelengths on a log-scale", action="store_true"
    )
    parser.add_argument(
        "--wavenum", help="plot wavenumbers = 1/wavelengths", action="store_true"
    )
    parser.add_argument(
        "--deredden",
        help="deredden the data based on saved parameters",
        action="store_true",
    )
    parser.add_argument(
        "--rebin_res",
        help="resolution in wavelength for rebinning spectra",
        type=float,
        default=None,
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.onefig:  # plot all spectra in the same figure
        plot_multi_spectra(
            args.starlist,
            args.path,
            mlam4=args.mlam4,
            HI_lines=args.HI_lines,
            range=args.range,
            norm_range=args.norm_range,
            spread=args.spread,
            exclude=args.exclude,
            log=args.log,
            wavenum=args.wavenum,
            deredden=args.deredden,
            rebin_res=args.rebin_res,
            pdf=args.pdf,
        )
    else:  # plot all spectra separately
        if args.spread:
            parser.error(
                "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the spectra if there is more than one spectrum in the same plot."
            )
        for star in args.starlist:
            plot_spectrum(
                star,
                args.path,
                mlam4=args.mlam4,
                HI_lines=args.HI_lines,
                range=args.range,
                norm_range=args.norm_range,
                exclude=args.exclude,
                log=args.log,
                wavenum=args.wavenum,
                deredden=args.deredden,
                rebin_res=args.rebin_res,
                pdf=args.pdf,
            )


if __name__ == "__main__":
    main()
