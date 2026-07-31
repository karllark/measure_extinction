import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u

from measure_extinction.utils.helpers import get_datapath

__all__ = ["mock_stis_data"]


def convolve_irregular_spectrum(newwavelength, wavelength, flux, fwhm):
    """
    Convolves an irregularly-spaced spectrum with a wavelength-dependent FWHM.
    From Google/Gemini, heavily modified afterwards.

    Parameters:
    newwavelength (ndarray): 1D array for mocked spectra

    wavelength (ndarray): 1D array of accompanying flux values

    flux (ndarray): 1D array of flux values

    fwhm (ndarray): 1D array with FWHM as a function of newwavelength

    Returns:
    ndarray: Convolved flux array.
    """
    n = len(newwavelength)
    convolved_flux = np.full((n), np.nan) * flux.unit

    # Constant conversion factor from FWHM to Gaussian sigma
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))

    for i in range(n):
        # 1. Get current wavelength, local FWHM, and local sigma
        w_center = newwavelength[i]
        sigma = fwhm[i] / fwhm_to_sigma

        # 2. Dynamic window: limit computation to pixels within 4 sigma
        # This keeps the loop efficient instead of computing all NxN distances
        idx_in_window = np.where(np.abs(wavelength - w_center) <= 4.0 * sigma)[0]

        if len(idx_in_window) <= 1:
            convolved_flux[i] = np.nan
        else:
            # 3. Calculate true physical distances from the center pixel
            dw = wavelength[idx_in_window] - w_center

            # 4. Evaluate Gaussian weights using the local sigma
            weights = np.exp(-0.5 * (dw / sigma) ** 2)

            # 5. Normalize weights and perform the dot product
            weights /= np.sum(weights)
            convolved_flux[i] = np.dot(flux[idx_in_window], weights)

    return convolved_flux


def mock_stis_single_grating(moddata, gname="G140L", applylsfs=True):
    """
    Mock up a single grating STIS low resolution observation using tabulated
    line spread functions (LSFs)

    Parameters
    ----------
    moddata : astropy.table
        Model spectrum at high enough resolution to support "convolution" with
        the LSFs

    gname : str
        name of the grating to mocked

    applylsfs : boolean
        allows for mocking with and without the LSFs

    Returns
    -------
    cmoddata : astropy.table
        Convolved and cropped model spectrum for the grating requested
    """
    if gname == "G140L":
        gtags = ["G140L_1200", "G140L_1500"]
        gwaves = [1200.0, 1500.0] * u.Angstrom
        grange = [1118.7028617667227, 1715.2138336094122] * u.Angstrom
        gdelta = 0.5831004076162168 * u.Angstrom  # Angstrom/pixel
    elif gname == "G230L":
        gtags = ["G230L_1700", "G230L_2400"]
        gwaves = [1700.0, 2400.0] * u.Angstrom
        grange = [1572.0793168982548, 3155.9334544319254] * u.Angstrom
        gdelta = 1.548239955089159 * u.Angstrom  # Angstrom/pixel
    elif gname == "G430L":
        gtags = ["G430L_3200", "G430L_5500"]
        gwaves = [3200.0, 5500.0] * u.Angstrom
        grange = [2894.535384018087, 5704.064392633997] * u.Angstrom
        gdelta = 2.7463714795193273 * u.Angstrom  # Angstrom/pixel
    elif gname == "G750L":
        gtags = ["G750L_7000"]
        gwaves = [7000.0] * u.Angstrom
        grange = [5257.602037433256, 10249.424213346618] * u.Angstrom
        gdelta = 4.879600079462369 * u.Angstrom  # Angstrom/pixel
    else:
        raise ValueError(f"Grating {gname} not supported")

    nlsfs = len(gtags)

    data_path = f"{get_datapath()}/../utils/STIS_LSF/"
    lsfs = []
    for i, ctag in enumerate(gtags):
        a = QTable.read(
            f"{data_path}/data/LSF_{ctag}.txt",
            format="ascii.commented_header",
            header_start=-1,
        )
        a["DELTWAVE"] = a["Rel_pixel"] * gdelta

        if i > 0:
            if len(lsfs[0]["DELTWAVE"]) != len(a["DELTWAVE"]):
                b = QTable()
                b["DELTWAVE"] = lsfs[0]["DELTWAVE"]
                b["52x2.0"] = np.interp(
                    b["DELTWAVE"], a["DELTWAVE"], a["52x2.0"], left=0.0, right=0.0
                )
                a = b

        lsfs.append(a)

    minlsfdwave = min(lsfs[0]["DELTWAVE"])
    maxlsfdwave = min(lsfs[0]["DELTWAVE"])

    # crop wide to include full possible lsf range
    gvals = (moddata["WAVELENGTH"] >= (grange[0] - minlsfdwave)) & (
        moddata["WAVELENGTH"] <= (grange[1] + maxlsfdwave)
    )
    incwmoddata = moddata[:][gvals]

    # convolve
    outcwmoddata = moddata[:][gvals]
    if applylsfs:

        # for each wavelength, use average weighting with the appropriate LSF
        clsfwave = lsfs[0]["DELTWAVE"]
        for i, cwave in enumerate(outcwmoddata["WAVELENGTH"]):
            # generate LSFs at each wavelength by interpolating/extrapolating
            # from the 2 provided LSFs or just replicating a single LSFs
            if nlsfs == 1:
                clsf = lsfs[0]["52x2.0"]
            elif nlsfs == 2:
                clsf = lsfs[1]["52x2.0"] + (
                    (gwaves[1] - cwave) / (gwaves[1] - gwaves[0])
                ) * (lsfs[1]["52x2.0"] - lsfs[0]["52x2.0"])

            clsfwave = lsfs[0]["DELTWAVE"] + cwave
            # interpolate onto model wavelength grid
            clsf_int = np.interp(
                outcwmoddata["WAVELENGTH"], clsfwave, clsf, right=0.0, left=0.0
            )
            outcwmoddata["FLUX"][i] = np.average(incwmoddata["FLUX"], weights=clsf_int)

    # crop tight to only include the expected wavelengths
    gvals = (outcwmoddata["WAVELENGTH"] >= grange[0]) & (
        outcwmoddata["WAVELENGTH"] <= grange[1]
    )
    cmoddata = QTable()
    cmoddata["WAVELENGTH"] = outcwmoddata["WAVELENGTH"][gvals]
    cmoddata["FLUX"] = outcwmoddata["FLUX"][gvals]
    cmoddata["STAT-ERROR"] = outcwmoddata["SIGMA"][gvals]
    cmoddata["SYS-ERROR"] = outcwmoddata["SIGMA"][gvals]
    cmoddata["NPTS"] = outcwmoddata["NPTS"][gvals]

    return cmoddata


def mock_stis_data(moddata, applylsfs=True):
    """
    Mock STIS low-resolution grating observations given a model spectrum

    Parameters
    ----------
    moddata : astropy.table
        Model spectrum at high enough resolution to support "convolution" with
        the LSFs

    applylsfs : boolean
        allows for mocking with and without the LSFs

    Returns
    -------
    tablist : list of astropy.tables
        Each entry appropriate for one of the four low resolution gratings
    """
    allspec = []

    allspec.append(
        mock_stis_single_grating(moddata, gname="G140L", applylsfs=applylsfs)
    )
    allspec.append(
        mock_stis_single_grating(moddata, gname="G230L", applylsfs=applylsfs)
    )
    allspec.append(
        mock_stis_single_grating(moddata, gname="G430L", applylsfs=applylsfs)
    )
    allspec.append(
        mock_stis_single_grating(moddata, gname="G750L", applylsfs=applylsfs)
    )

    return allspec


def mock_nirspec_single_grating(moddata, gname="G140M/F100LP", applylsfs=True):
    """
    Mock up a single grating nirspec observation using wavelength dependent
    Gaussian LSF.

    Parameters
    ----------
    moddata : astropy.table
        Model spectrum at high enough resolution to support "convolution" with
        the LSFs

    gname : str
        name of the grating/filter to mocked

    applylsfs : boolean
        allows for mocking with and without the LSFs

    Returns
    -------
    cmoddata : astropy.table
        Convolved and cropped model spectrum for the grating requested
    """
    data_path = f"{get_datapath()}/../utils/NIRSpec_examples/"

    # get the wavelength grid and the resolving power versus wavelength
    if gname == "G140M/F070LP":
        exfile = "g191b2b_m_G140M_F070LP_NRS1_pfpc_x1d.fits"
        resfile = "jwst_nirspec_g140m_disp.fits"
    elif gname == "G140M/F100LP":
        exfile = "g191b2b_m_G140M_F100LP_NRS1_pfpc_x1d.fits"
        resfile = "jwst_nirspec_g140m_disp.fits"
    elif gname == "G235M/F170LP":
        exfile = "g191b2b_m_G235M_F170LP_NRS1_pfpc_x1d.fits"
        resfile = "jwst_nirspec_g235m_disp.fits"
    elif gname == "G395M/F290LP":
        exfile = "g191b2b_m_G395M_F290LP_NRS1_pfpc_x1d.fits"
        resfile = "jwst_nirspec_g395m_disp.fits"
    else:
        raise ValueError(f"Grating {gname} not supported")

    itab = QTable.read(f"{data_path}/{exfile}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=u.UnitsWarning)
        rtab = QTable.read(f"{data_path}/{resfile}")

    waves = (itab["WAVELENGTH"]).to(u.Angstrom)
    fwhm = waves / np.interp(waves, (rtab["WAVELENGTH"].value) * u.micron, rtab["R"].value)

    if applylsfs:
        new_flux = convolve_irregular_spectrum(
            waves, moddata["WAVELENGTH"], moddata["FLUX"], fwhm
        )
    else:
        new_flux = np.interp(waves, moddata["WAVELENGTH"], moddata["FLUX"])

    cmoddata = QTable()
    cmoddata["WAVELENGTH"] = waves.to(u.micron)
    cmoddata["FLUX"] = new_flux
    # cmoddata["STAT-ERROR"] = outcwmoddata["SIGMA"][gvals]
    # cmoddata["SYS-ERROR"] = outcwmoddata["SIGMA"][gvals]
    # cmoddata["NPTS"] = outcwmoddata["NPTS"][gvals]

    return cmoddata


def mock_nirspec_data(moddata, applylsfs=True):
    """
    Mock NIRSPEC low-resolution grating observations given a model spectrum

    Parameters
    ----------
    moddata : astropy.table
        Model spectrum at high enough resolution to support "convolution" with
        the LSFs

    applylsfs : boolean
        allows for mocking with and without the LSFs

    Returns
    -------
    tablist : list of astropy.tables
        Each entry appropriate for one of the four low resolution gratings
    """
    allspec = []

    allspec.append(
        mock_nirspec_single_grating(moddata, gname="G140M/F070LP", applylsfs=applylsfs)
    )

    allspec.append(
        mock_nirspec_single_grating(moddata, gname="G140M/F100LP", applylsfs=applylsfs)
    )

    allspec.append(
        mock_nirspec_single_grating(moddata, gname="G235M/F170LP", applylsfs=applylsfs)
    )

    allspec.append(
        mock_nirspec_single_grating(moddata, gname="G395M/F290LP", applylsfs=applylsfs)
    )

    return allspec


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nirspec_m", help="mock NIRSpec medium gratings instead of STIS", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    moddata = QTable.read(
        "/home/kgordon/Python/extstar_data/Models/tlusty_z001t15000g175v10_full.fits"
    )

    if args.nirspec_m:
        outname = "nirspec_m"
        nspec = 4
        mockobs_wolsfs = mock_nirspec_data(moddata, applylsfs=False)
        mockobs = mock_nirspec_data(moddata)
    else:
        outname = "stis"
        nspec = 4
        mockobs_wolsfs = mock_stis_data(moddata, applylsfs=False)
        mockobs = mock_stis_data(moddata)

    fig, ax = plt.subplots(nrows=nspec, figsize=(18, 10))

    # setup the plots
    fontsize = 12
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    for i, cmockobs in enumerate(mockobs):
        ax[i].plot(mockobs_wolsfs[i]["WAVELENGTH"], mockobs_wolsfs[i]["FLUX"], "k-")
        ax[i].plot(cmockobs["WAVELENGTH"], cmockobs["FLUX"], "b-")
        ax[i].set_ylabel("Flux")

    waveunit = cmockobs["WAVELENGTH"].unit
    ax[nspec - 1].set_xlabel(rf"$\lambda$ [{waveunit}]")

    fig.tight_layout()

    if args.png:
        fig.savefig(f"mock_{outname}_obs.png")
    elif args.pdf:
        fig.savefig(f"mock_{outname}_obs.pdf")
    else:
        plt.show()
