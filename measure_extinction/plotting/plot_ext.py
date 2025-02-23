import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pandas as pd
import os

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.extdata import ExtData
from dust_extinction.parameter_averages import CCM89


def plot_average(
    path,
    filename="average_ext.fits",
    ax=None,
    rebin_fac=None,
    extmodels=False,
    fitmodel=False,
    res=False,
    HI_lines=False,
    range=None,
    exclude=[],
    log=False,
    spread=False,
    annotate_key=None,
    annotate_wave_range=None,
    wavenum=False,
    pdf=False,
):
    """
    Plot the average extinction curve

    Parameters
    ----------
    path : string
        Path to the average extinction curve fits file

    filename : string [default="average_ext.fits"]
        Name of the average extinction curve fits file

    ax : AxesSubplot [default=None]
        Axes of plot on which to add the average extinction curve if pdf=False

    rebin_fac: int [default=None]
        factor by which to rebin extinction curve

    extmodels: boolean [default=False]
        Whether or not to overplot Milky Way extinction curve models

    fitmodel: boolean [default=False]
        Whether or not to overplot a fitted model

    res : boolean [default=False]
        Whether or not to plot the residuals of the fitting (only useful when fitmodel=True)

    HI_lines : boolean [default=False]
        Whether or not to indicate the HI-lines in the plot

    range : list of 2 floats [default=None]
        Wavelength range to be plotted (in micron) - [min,max]

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., "IRS", "IRAC1")

    log : boolean [default=False]
        Whether or not to plot the wavelengths on a log-scale

    spread : boolean [default=False]
        Whether or not to offset the average extinction curve from the other curves (only relevant when pdf=False and ax=None)

    annotate_key : string [default=None]
        type of data for which to annotate text (e.g., SpeX_LXD) (only relevant when pdf=False and ax=None)

    annotate_wave_range : list of 2 floats [default=None]
        min/max wavelength range for the annotation of the text (only relevant when pdf=False and ax=None)

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    pdf : boolean [default=False]
        - If False, the average extinction curve will be overplotted on the current plot (defined by ax)
        - If True, the average extinction curve will be plotted in a separate plot and saved as a pdf

    Returns
    -------
    Plots the average extinction curve
    """
    # read in the average extinction curve (if it exists)
    fname = f"{path}/{filename}"
    if os.path.isfile(fname):
        average = ExtData(fname)
    else:
        warnings.warn(
            "An average extinction curve with the name "
            + filename
            + " could not be found in "
            + path
            + ". Please calculate the average extinction curve first with the calc_ave_ext function in measure_extinction/utils/calc_ext.py.",
            UserWarning,
        )

    # make a new plot if requested
    if pdf:
        # plotting setup for easier to read plots
        fs = 20
        font = {"size": fs}
        plt.rc("font", **font)
        plt.rc("lines", linewidth=1)
        plt.rc("axes", linewidth=2)
        plt.rc("xtick.major", width=2, size=10)
        plt.rc("xtick.minor", width=1, size=5)
        plt.rc("ytick.major", width=2, size=10)
        plt.rc("ytick.minor", width=1, size=5)
        plt.rc("axes.formatter", min_exponent=2)
        plt.rc("xtick", direction="in", labelsize=fs * 0.8)
        plt.rc("ytick", direction="in", labelsize=fs * 0.8)

        # create the plot
        fig, ax = plt.subplots(figsize=(10, 7))
        average.plot(
            ax, exclude=exclude, rebin_fac=rebin_fac, color="k", wavenum=wavenum
        )

        # plot Milky Way extinction models if requested
        if extmodels:
            plot_extmodels(average, alax=True, wavenum=wavenum)

        # overplot a fitted model if requested
        if fitmodel:
            plot_fitmodel(average, res=res, wavenum=wavenum)

        # plot HI-lines if requested
        if HI_lines:
            plot_HI(ax, wavenum=wavenum)

        # zoom in on a specific region if requested
        if range is not None:
            zoom(ax, range)

        # finish configuring the plot
        if log:
            ax.set_xscale("log")
        ax.xaxis.label.set_size(fs)
        ax.set_ylabel(average._get_ext_ytitle(ytype=average.type), fontsize=fs)
        fig.savefig(path + "average_ext.pdf", bbox_inches="tight")

        # return the figure and axes for additional manipulations
        return fig, ax

    else:
        if spread:
            yoffset = -0.3
        else:
            yoffset = 0
        average.plot(
            ax,
            exclude=exclude,
            rebin_fac=rebin_fac,
            color="k",
            alpha=0.6,
            yoffset=yoffset,
            annotate_key=annotate_key,
            annotate_wave_range=annotate_wave_range,
            annotate_text="average",
            annotate_yoffset=0.05,
        )

        # overplot a fitted model if requested
        if fitmodel:
            plot_fitmodel(average, yoffset=yoffset, wavenum=wavenum)


def plot_extmodels(extdata, alax=False, wavenum=False):
    """
    Plot Milky Way extinction curve models of Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245), only possible for wavelengths between 0.1 and 3.33 micron

    Parameters
    ----------
    extdata : ExtData
        Extinction data under consideration

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    Returns
    -------
    Overplots extinction curve models
    """
    x = np.arange(0.1, 3.33, 0.01) * u.micron
    Rvs = [2.0, 3.1, 4.0, 5.0]
    style = ["--", "-", ":", "-."]
    for i, cRv in enumerate(Rvs):
        curve = CCM89(Rv=cRv)
        if alax:
            if extdata.type_rel_band != "V":
                emod = CCM89(cRv)
                (indx,) = np.where(extdata.type_rel_band == extdata.names["BAND"])
                axav = emod(extdata.waves["BAND"][indx[0]])
            else:
                axav = 1.0
            y = curve(x) / axav
        else:
            # compute A(V)
            if "AV" not in extdata.columns.keys():
                extdata.calc_AV()
            # convert the model curve from A(lambda)/A(V) to E(lambda-V), using the computed A(V) of the data.
            y = (curve(x) - 1) * extdata.columns["AV"][0]

        if wavenum:
            px = 1 / x
        else:
            px = x
        plt.plot(
            px.value,
            y,
            style[i],
            color="k",
            alpha=0.7,
            linewidth=1,
            label="R(V) = {:4.1f}".format(cRv),
        )
        # allow to find the best position (supports regular and wavenum)
        plt.legend()


def plot_fitmodel(extdata, alax=False, yoffset=0, res=False, wavenum=False):
    """
    Overplot a fitted model if available

    Parameters
    ----------
    extdata : ExtData
        Extinction data under consideration

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    yoffset : float [default=0]
        Offset of the corresponding extinction curve (in order to match the model to the curve)

    res : boolean [default=False]
        Whether or not to plot the residuals of the fitting (only useful when plotting a single extinction curve)

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    Returns
    -------
    Overplots a fitted model
    """
    # plot a fitted model if available
    if extdata.model:
        if extdata.model["type"] == "pow_elx":
            # in this case, fitted amplitude must be multiplied by A(V) to get the "combined" model amplitude
            labeltxt = r"$%5.2f \lambda ^{-%5.2f} - %5.2f$" % (
                extdata.model["params"][0].value * extdata.model["params"][3].value,
                extdata.model["params"][2].value,
                extdata.model["params"][3].value,
            )
        elif extdata.model["type"] == "pow_alax":
            labeltxt = r"$%5.3f \,\lambda^{-%5.2f}$" % (
                extdata.model["params"][0].value,
                extdata.model["params"][2].value,
            )
        else:
            labeltxt = "fitted model"

        # obtain the model extinctions
        mod_ext = extdata.model["exts"]

        # if the plot needs to be in A(lambda)/A(V), the model extinctions need to be converted to match the data
        if alax:
            mod_ext = (mod_ext / extdata.columns["AV"][0]) + 1

        if wavenum:
            x = 1 / extdata.model["waves"]
        else:
            x = extdata.model["waves"]

        plt.plot(
            x,
            mod_ext + yoffset,
            "-",
            lw=3,
            color="crimson",
            alpha=0.8,
            label=labeltxt,
            zorder=5,
        )
        plt.legend(loc="lower left")

        # plot the residuals if requested
        if res:
            plt.setp(plt.gca().get_xticklabels(), visible=False)
            plt.axes([0.125, 0, 0.775, 0.11], sharex=plt.gca())
            plt.scatter(x, extdata.model["residuals"], s=0.5, color="k")
            plt.axhline(ls="--", c="k", alpha=0.5)
            plt.axhline(y=0.05, ls=":", c="k", alpha=0.5)
            plt.axhline(y=-0.05, ls=":", c="k", alpha=0.5)
            plt.ylim(-0.1, 0.1)
            plt.ylabel("residual")

    else:
        warnings.warn(
            "There is no fitted model available to plot.",
            stacklevel=2,
        )


def plot_HI(ax, wavenum=False):
    """
    Indicate the HI-lines on the plot (between 912A and 10 micron)

    Parameters
    ----------
    ax : AxesSubplot
        Axes of plot for which to add the HI-lines
    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    Returns
    -------
    Indicates HI-lines on the plot
    """
    # read in HI-lines
    data_path = get_datapath()
    table = pd.read_table(f"{data_path}/HI_lines.list", sep=r"\s+", comment="#")
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
            if wavenum:
                x = 1.0 / wave
            else:
                x = wave
            ax.axvline(x, color=colors(name - 1), lw=0.05, alpha=0.4)
        # add the name of the series
        mwave = series.wavelength.mean()
        if wavenum:
            xm = 1.0 / mwave
        else:
            xm = mwave
        ax.text(
            xm,
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
        Wavelength range to be plotted (in micron) - [min,max]

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
    h = ymax - ymin
    ax.set_ylim(ymin - 0.05 * h, ymax + 0.05 * h)


def plot_multi_extinction(
    starpair_list,
    path,
    alax=False,
    average=False,
    extmodels=False,
    fitmodel=False,
    HI_lines=False,
    range=None,
    spread=0.0,
    exclude=[],
    log=False,
    text_offsets=[],
    text_angles=[],
    multicolor=False,
    wavenum=False,
    figsize=None,
    rebin_res=None,
    pdf=False,
):
    """
    Plot the extinction curves of multiple stars in the same plot

    Parameters
    ----------
    starpair_list : list of strings
        List of star pairs for which to plot the extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    average : boolean [default=False]
        Whether or not to plot the average extinction curve

    extmodels: boolean [default=False]
        Whether or not to overplot Milky Way extinction curve models

    fitmodel: boolean [default=False]
        Whether or not to overplot a fitted model

    HI_lines : boolean [default=False]
        Whether or not to indicate the HI-lines in the plot

    range : list of 2 floats [default=None]
        Wavelength range to be plotted (in micron) - [min,max]

    spread : float [default=0]
        Amount to addiatively spread the curves 
 
    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., "IRS", "IRAC1")

    log : boolean [default=False]
        Whether or not to plot the wavelengths on a log-scale

    text_offsets : list of floats [default=[]]
        List of the same length as starpair_list with offsets for the annotated text

    text_angles : list of integers [default=[]]
        List of the same length as starpair_list with rotation angles for the annotated text

    multicolor : boolean [default=False]
        Whether or not to give all curves a different color

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    figsize : tuple [default=None]
        Tuple with figure size (e.g. (8,15))

    rebin_res : float
        Spectral resolution to rebin extinction curves

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with extinction curves of multiple stars
    """
    # plotting setup for easier to read plots
    fs = 18
    font = {"size": fs}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2, size=10)
    plt.rc("xtick.minor", width=1, size=5)
    plt.rc("ytick.major", width=2, size=10)
    plt.rc("ytick.minor", width=1, size=5)
    plt.rc("axes.formatter", min_exponent=2)
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 1.1)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 1.1)

    # create the plot
    if figsize is None:
        figsize = (8, len(starpair_list))
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")

    # set default text offsets and angles
    if text_offsets == []:
        text_offsets = np.full(len(starpair_list), 0.2)
    if text_angles == []:
        text_angles = np.full(len(starpair_list), 10)

    for i, starpair in enumerate(starpair_list):
        if ".fits" not in starpair:
            fname = "%s%s_ext.fits" % (path, starpair.lower())
        else:
            fname = starpair 

        # read in the extinction curve data
        extdata = ExtData(fname)

        # spread out the curves if requested
        if spread != 0.0:
            yoffset = spread * i
        else:
            yoffset = 0.0

        # determine where to add the name of the star
        # find the shortest plotted wavelength, and give preference to spectral data when available
        exclude2 = []
        if "BAND" in extdata.waves.keys() and len(extdata.waves.keys()) > 1:
            exclude2 = ["BAND"]
        (waves, exts, ext_uncs) = extdata.get_fitdata(
            extdata.waves.keys() - (exclude + exclude2)
        )
        if range is not None:
            waves = waves[waves.value >= range[0]]
        min_wave = waves[-1]
        # find out which data type corresponds with this wavelength
        for data_type in extdata.waves.keys():
            if data_type in exclude:
                continue
            used_waves = extdata.waves[data_type][extdata.npts[data_type] > 0]
            if min_wave in used_waves:
                ann_key = data_type
        ann_range = [min_wave, min_wave] * u.micron

        # plot the extinction curve
        if multicolor:
            pcolor = colors(i % 10)
        else:
            pcolor = "k"
        extdata.plot(
            ax,
            color=pcolor,
            alpha=0.7,
            alax=alax,
            exclude=exclude,
            yoffset=yoffset,
            annotate_key=ann_key,
            annotate_wave_range=ann_range,
            annotate_text=extdata.red_file.split(".")[0].upper(),
            annotate_yoffset=text_offsets[i],
            annotate_rotation=text_angles[i],
            annotate_color=pcolor,
            wavenum=wavenum,
        )

        # overplot a fitted model if requested
        if fitmodel:
            plot_fitmodel(extdata, alax=alax, yoffset=yoffset, wavenum=wavenum)

    # overplot Milky Way extinction curve models if requested
    if extmodels:
        if alax:
            plot_extmodels(extdata, alax, wavenum=wavenum)
        else:
            warnings.warn(
                "Overplotting Milky Way extinction curve models on a figure with multiple observed extinction curves in E(lambda-V) units is disabled, because the model curves in these units are different for every star, and would overload the plot. Please, do one of the following if you want to overplot Milky Way extinction curve models: 1) Use the flag --alax to plot ALL curves in A(lambda)/A(V) units, OR 2) Plot all curves separately by removing the flag --onefig.",
                stacklevel=2,
            )

    # plot the average extinction curve if requested
    if average:
        plot_average(
            path,
            ax=ax,
            extmodels=extmodels,
            fitmodel=fitmodel,
            exclude=exclude,
            spread=spread,
            annotate_key=ann_key,
            annotate_wave_range=ann_range,
        )

    # define the output name
    outname = "all_ext_%s.pdf" % (extdata.type)

    # plot HI-lines if requested
    if HI_lines:
        plot_HI(ax, wavenum=wavenum)

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    if log:
        ax.set_xscale("log")
    if wavenum:
        xlab = r"$1/\lambda$ [$\mu m^{-1}$]"
    else:
        xlab = r"$\lambda$ [$\mu m$]"
    plt.xlabel(xlab, fontsize=1.5 * fs)
    ylabel = extdata._get_ext_ytitle(ytype=extdata.type)
    if spread:
        ylabel += " + offset"
    ax.set_ylabel(ylabel, fontsize=1.5 * fs)

    # show the figure or save it to a pdf file
    if pdf:
        fig.savefig(path + outname, bbox_inches="tight")
    else:
        plt.show()

    # return the figure and axes for additional manipulations
    return fig, ax


def plot_extinction(
    starpair,
    path,
    alax=False,
    extmodels=False,
    fitmodel=False,
    HI_lines=False,
    range=None,
    exclude=[],
    log=False,
    wavenum=False,
    rebin_res=None,
    pdf=False,
):
    """
    Plot the extinction curve of a star

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to plot the extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    alax : boolean [default=False]
        Whether or not to plot A(lambda)/A(X) instead of E(lambda-X)

    extmodels: boolean [default=False]
        Whether or not to overplot Milky Way extinction curve models

    fitmodel: boolean [default=False]
        Whether or not to overplot a fitted model

    HI_lines : boolean [default=False]
        Whether or not to indicate the HI-lines in the plot

    range : list of 2 floats [default=None]
        Wavelength range to be plotted (in micron) - [min,max]

    exclude : list of strings [default=[]]
        List of data type(s) to exclude from the plot (e.g., "IRS", "IRAC1")

    log : boolean [default=False]
        Whether or not to plot the wavelengths on a log scale

    wavenum : boolean [default=False]
        Whether or not to plot the wavelengths as wavenumbers = 1/wavelength

    rebin_res : float [default=None]
        Spectral resolution to rebin extinction curve

    pdf : boolean [default=False]
        Whether or not to save the figure as a pdf file

    Returns
    -------
    Figure with extinction curve
    """
    # plotting setup for easier to read plots
    fs = 18
    font = {"size": fs}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2, size=10)
    plt.rc("xtick.minor", width=1, size=5)
    plt.rc("ytick.major", width=2, size=10)
    plt.rc("ytick.minor", width=1, size=5)
    plt.rc("axes.formatter", min_exponent=2)
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 1.1)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 1.1)

    # create the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # read in extinction curve data for this star
    if ".fits" not in starpair:
        fname = "%s%s_ext.fits" % (path, starpair.lower())
    else:
        fname = starpair
    extdata = ExtData(fname)

    # rebin if desired
    if rebin_res is not None:
        for src in extdata.exts.keys():
            if src != "BAND":
                rebin_waverange = (
                    np.array(
                        [
                            min(extdata.waves[src]).to(u.micron).value,
                            max(extdata.waves[src]).to(u.micron).value,
                        ]
                    )
                    * u.micron
                )
                extdata.rebin_constres(src, rebin_waverange, rebin_res)

    # plot extinction curve data
    extdata.plot(
        ax,
        alax=alax,
        exclude=exclude,
        color="k",
        wavenum=wavenum,
    )

    # define the output name
    outname = "%s_ext_%s.pdf" % (starpair.lower(), extdata.type)

    # plot Milky Way extinction models if requested
    if extmodels:
        plot_extmodels(extdata, alax, wavenum=wavenum)

    # overplot a fitted model if requested
    if fitmodel:
        plot_fitmodel(extdata, alax=alax, res=True, wavenum=wavenum)

    # plot HI-lines if requested
    if HI_lines:
        plot_HI(ax, wavenum=wavenum)

    # zoom in on a specific region if requested
    if range is not None:
        zoom(ax, range)
        outname = outname.replace(".pdf", "_zoom.pdf")

    # finish configuring the plot
    if extdata.red_file == "":
        ax.set_title(starpair, fontsize=50)
    else:
        ax.set_title(extdata.red_file.replace(".dat", ""), fontsize=50)
    if (extdata.comp_file is not None) and (extdata.comp_file != ""):
        ax.text(
            0.99,
            0.95,
            "comparison: " + extdata.comp_file.replace(".dat", ""),
            fontsize=25,
            horizontalalignment="right",
            transform=ax.transAxes,
        )
    if log:
        ax.set_xscale("log")
    if wavenum:
        xlab = r"$1/\lambda$ [$\mu m^{-1}$]"
    else:
        xlab = r"$\lambda$ [$\mu m$]"
    plt.xlabel(xlab, fontsize=1.5 * fs)
    ax.set_ylabel(
        extdata._get_ext_ytitle(ytype=extdata.type),
        fontsize=1.5 * fs,
    )

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
        "starpair_list", nargs="+", help="filenames of extinction curves"
    )
    parser.add_argument(
        "--path",
        help="path to the data files",
        default="./",
    )
    parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
    parser.add_argument(
        "--average", help="plot the average extinction curve", action="store_true"
    )
    parser.add_argument(
        "--extmodels", help="plot extinction curve models", action="store_true"
    )
    parser.add_argument("--fitmodel", help="plot a fitted model", action="store_true")
    parser.add_argument("--HI_lines", help="indicate the HI-lines", action="store_true")
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all curves in the same figure",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread",
        help="spread the curves out over the figure by this amount; can only be used in combination with --onefig",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="data type(s) to be excluded from the plotting",
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
        "--rebin_res",
        help="resolution in wavelength for rebinning spectral extinction",
        type=float,
        default=None,
    )
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    if args.onefig:  # plot all curves in the same figure
        plot_multi_extinction(
            args.starpair_list,
            args.path,
            alax=args.alax,
            average=args.average,
            extmodels=args.extmodels,
            fitmodel=args.fitmodel,
            HI_lines=args.HI_lines,
            range=args.range,
            spread=args.spread,
            exclude=args.exclude,
            wavenum=args.wavenum,
            log=args.log,
            rebin_res=args.rebin_res,
            pdf=args.pdf,
        )
    else:  # plot all curves separately
        if args.spread:
            parser.error(
                "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the curves if there is more than one curve in the same plot."
            )
        if args.average:
            parser.error(
                "The flag --average can only be used in combination with the flag --onefig. It only makes sense to add the average extinction curve to a plot with multiple curves."
            )
        for starpair in args.starpair_list:
            plot_extinction(
                starpair,
                args.path,
                alax=args.alax,
                extmodels=args.extmodels,
                fitmodel=args.fitmodel,
                HI_lines=args.HI_lines,
                range=args.range,
                exclude=args.exclude,
                wavenum=args.wavenum,
                log=args.log,
                rebin_res=args.rebin_res,
                pdf=args.pdf,
            )


if __name__ == "__main__":
    main()
