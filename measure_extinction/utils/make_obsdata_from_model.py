import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from astropy.io import ascii
from astropy.table import QTable, Column
import astropy.units as u
from astropy.convolution import Gaussian1DKernel, convolve
from synphot import SpectralElement
import stsynphot as STS

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.stardata import BandData
from measure_extinction.merge_obsspec import (
    obsspecinfo,
    merge_gen_obsspec,
    merge_iue_obsspec,
    merge_stis_obsspec,
)
from measure_extinction.utils.mock_spectra_data import mock_stis_data

__all__ = ["make_obsdata_from_model"]


def rebin_spectrum(wave, flux, resolution, wave_range):
    """
    Rebin spectrum to input resolution over input wavelength range
    High to lower resolution only

    Parameters
    ----------
    wave : vector
        wavelengths of spectrum

    flux : vector
        spectrum in flux units

    resolution : float
        resolution of output spectrum

    wave_range : [float, float]
        wavelength range of output spectrum

    Outputs
    ------
    wave, flux, npts : tuple of vectors
        the model wavelength, flux, and npts at the requested wavelength
    """
    npts = int(
        np.log10(wave_range[1] / wave_range[0])
        / np.log10((1.0 + 2.0 * resolution) / (2.0 * resolution - 1.0))
    )

    twave = np.logspace(
        np.log10(wave_range[0]), np.log10(wave_range[1]), num=npts + 1, endpoint=True
    )
    full_wave_min = twave[0:-1]
    full_wave_max = twave[1:]
    full_wave = 0.5 * (full_wave_min + full_wave_max)

    full_flux = np.zeros((npts))
    full_npts = np.zeros((npts), dtype=int)

    for k in range(npts):
        (indxs,) = np.where((wave >= full_wave_min[k]) & (wave < full_wave_max[k]))
        n_indxs = len(indxs)
        if n_indxs > 0:
            full_flux[k] = np.sum(flux[indxs])
            full_npts[k] = n_indxs

    # divide by the # of points to create the final rebinned spectrum
    (indxs,) = np.where(full_npts > 0)
    if len(indxs):
        full_flux[indxs] = full_flux[indxs] / full_npts[indxs]

    # interpolate to fill in missing points in the rebinned spectrum
    #  e.g., the model spectrum is not computed at a high enough resolution
    #        at all the needed wavelengths
    (zindxs,) = np.where(full_npts <= 0)
    if len(zindxs):
        ifunc = interp1d(
            full_wave[indxs], full_flux[indxs], kind="linear", bounds_error=False
        )
        full_flux = ifunc(full_wave)
        full_npts[zindxs] = 1
        (nanindxs,) = np.where(~np.isfinite(full_flux))
        if len(nanindxs):
            full_flux[nanindxs] = 0.0
            full_npts[nanindxs] = 0

    return (full_wave, full_flux, full_npts)


def get_phot(mwave, mflux, band_names, band_resp_filenames):
    """
    Compute the magnitudes in the requested bands.

    Parameters
    ----------
    mwave : vector
        wavelengths of model flux

    mflux : vector
        model fluxes

    band_names: list of strings
        names of bands requested

    band_resp_filename : list of strings
        filenames of band response functions for the requested bands

    Outputs
    -------
    bandinfo : BandData object
    """
    # get a BandData object
    bdata = BandData("BAND")

    # path for non-HST band response curves
    data_path = f"{get_datapath()}/Band_RespCurves/"

    # compute the fluxes in each band
    for k, cband in enumerate(band_names):
        if "HST" in cband:
            bp_info = cband.split("_")
            bp = STS.band("%s,%s,%s" % (bp_info[1], bp_info[2], bp_info[3]))
            ncband = "%s_%s" % (bp_info[1], bp_info[3])
        else:
            bp = SpectralElement.from_file("%s%s" % (data_path, band_resp_filenames[k]))
            ncband = cband

        # check if the wavelength units are in microns instead of Angstroms
        #   may not work
        if max(bp.waveset) < 500 * u.Angstrom:
            # print(f"filter {cband} wavelengths not in angstroms, assuming in microns")
            mfac = 1e-4
        else:
            mfac = 1.0
        iresp = bp(mwave * mfac)
        inttop = np.trapezoid(mwave * iresp * mflux, mwave)
        intbot = np.trapezoid(mwave * iresp, mwave)
        bflux = inttop / intbot
        bflux_unc = 0.0
        bdata.band_fluxes[ncband] = (bflux, bflux_unc)

    # calculate the band magnitudes from the fluxes
    bdata.get_band_mags_from_fluxes()

    # get the band fluxes from the magnitudes
    #   partially redundant, but populates variables useful later
    bdata.get_band_fluxes()

    return bdata


def write_dat_file(
    filename,
    bandinfo,
    specinfo,
    header_info=["# obsdata created from model spectrum"],
    modelparams=None,
):
    """
    Write out a DAT file containing the photometry and pointers to
    spectroscopy files.

    Parameters
    ----------
    filename: string
        full file name of output DAT file

    bandinfo : BandData object
        contains the photometry data

    specinfo : dict of {type: filename}

    header_info: string array
        comments to add to the header

    modelparams: dict of {type: value}
        model parameters
        e.g., {'Teff': 10000.0, 'logg': 4.0, 'Z': 1, 'vturb': 2.0}
    """
    dfile = open(filename, "w")

    for hline in header_info:
        dfile.write("%s\n" % (hline))

    for cband in bandinfo.bands.keys():
        dfile.write(
            "%s = %f +/- %f\n"
            % (cband, bandinfo.bands[cband][0], bandinfo.bands[cband][1])
        )

    if specinfo is not None:
        for ckey in specinfo.keys():
            dfile.write("%s = %s\n" % (ckey, specinfo[ckey]))

    if modelparams is not None:
        for ckey in modelparams.keys():
            if isinstance(modelparams[ckey], str):
                dfile.write("%s = %s\n" % (ckey, modelparams[ckey]))
            else:
                dfile.write("%s = %f\n" % (ckey, modelparams[ckey]))

    dfile.close()


def make_obsdata_from_model(
    model_filename,
    model_type="tlusty",
    model_params=None,
    output_filebase=None,
    output_path=None,
    show_plot=False,
    specs="all",
):
    """
    Create the necessary data files (.dat and spectra) from a
    stellar model atmosphere model to use as the unreddened
    comparsion star in the measure_extinction package

    Parameters
    ----------
    model_filename: string
        name of the file with the stellar atmosphere model spectrum

    model_type: string [default = 'tlusty']
        model type

    model_params: dict of {type: value}
        model parameters
        e.g., {'Teff': 10000.0, 'logg': 4.0, 'Z': 1, 'vturb': 2.0}

    output_filebase: string
        base for the output files
        E.g., output_filebase.dat and output_filebase_stis.fits

    output_path: string
        path to use for output files

    show_plot: boolean
        show a plot of the original and rebinned spectra/photometry

    specs: list
        specify the specific spectra to mock, [default="all"]
        set to None to turn off outputing any spectra (i.e., only output DAT file)
    """

    if output_filebase is None:
        output_filebase = "%s_standard" % (model_filename)

    if output_path is None:
        output_path = "/home/kgordon/Python_git/extstar_data/"

    allowed_model_types = ["tlusty"]
    if model_type not in allowed_model_types:
        raise ValueError("%s not an allowed model type" % (model_type))

    # read in the model spectrum
    mspec = ascii.read(
        model_filename,
        format="no_header",
        fast_reader={"exponent_style": "D"},
        names=["Wave", "SFlux"],
    )

    # convert the type to float
    mspec["SFlux"] = mspec["SFlux"].astype(float)

    # error in file where the exponent 'D' is missing
    #   means that SFlux is read in as a string
    # solution is to remove the rows with the problem and replace
    #   the fortran 'D' with an 'E' and then convert to floats
    # mspec["SFlux"] = mspec["SFlux"].astype(float)
    # if mspec["SFlux"].dtype != float:
    #     indxs = [k for k in range(len(mspec)) if "D" not in mspec["SFlux"][k]]
    #     if len(indxs) > 0:
    #         indxs = [k for k in range(len(mspec)) if "D" in mspec["SFlux"][k]]
    #         mspec = mspec[indxs]
    #         new_strs = [cval.replace("D", "E") for cval in mspec["SFlux"].data]
    #         mspec["SFlux"] = new_strs
    #         mspec["SFlux"] = mspec["SFlux"].astype(float)

    # set the units
    fluxunit = u.erg / (u.s * u.cm * u.cm * u.angstrom)
    mspec["Wave"].unit = u.angstrom
    mspec["SFlux"].unit = fluxunit

    # now extract the wave and flux colums
    mwave = mspec["Wave"]
    mflux = mspec["SFlux"]

    # determine which spectra to mock and output
    if specs == "all":
        # fmt: off
        specs = ["MODEL_FULL_LOWRES", "MODEL_FULL", "IUE", "STIS", "IRS",
                 "WFC3_G102", "WFC3_G141",
                 "NIRISS_SOSS", "NIRCAM_SS", "MIRI_LRS", "MIRI_IFU"]
        # fmt: on

    # rebin to R=10000 for speed
    #   use a wavelength range that spans FUSE to Spitzer IRS
    rbres = 10000.0
    wave_rebin, flux_rebin, npts_rebin = rebin_spectrum(
        mwave.value, mflux.value, rbres, [912.0, 310000.0]
    )

    # compute photometry
    john_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
    john_fnames = [f"John{cband}.dat" for cband in john_bands]
    hst_bands = [
        "HST_WFC3_UVIS1_F275W",
        "HST_WFC3_UVIS1_F336W",
        "HST_WFC3_UVIS1_F475W",
        "HST_WFC3_UVIS1_F625W",
        "HST_WFC3_UVIS1_F775W",
        "HST_WFC3_UVIS1_F814W",
        "HST_WFC3_IR_F110W",
        "HST_WFC3_IR_F160W",
        "HST_ACS_WFC1_F475W",
        "HST_ACS_WFC1_F814W",
        "HST_WFPC2_4_F170W",
        "HST_WFPC2_4_F255W",
        "HST_WFPC2_4_F336W",
        "HST_WFPC2_4_F439W",
        "HST_WFPC2_4_F555W",
        "HST_WFPC2_4_F814W",
    ]
    hst_fnames = [""]
    # fmt: off
    ir_bands = ['IRAC1', 'IRAC2', 'IRAC3', 'IRAC4', 'IRS15', 'MIPS24',
                'WISE1', 'WISE2', 'WISE3', 'WISE4']
    # fmt: on
    ir_fnames = [f"{cband}.dat" for cband in ir_bands]
    bands = john_bands + ir_bands + hst_bands
    band_fnames = john_fnames + ir_fnames + hst_fnames

    bandinfo = get_phot(wave_rebin, flux_rebin, bands, band_fnames)

    # dictionary to saye names of spectroscopic filenames
    specinfo = {}

    # rebin to R=100 for speed of reddened photometry calculation
    #   use a wavelength range that spans FUSE to Spitzer IRS
    rbres_lowres = 100.0
    wave_rebin_lowres, flux_rebin_lowres, npts_rebin_lowres = rebin_spectrum(
        mwave.value, mflux.value, rbres_lowres, [912.0, 310000.0]
    )
    full_file_lowres = "%s_full_lowres.fits" % (output_filebase)
    if (specs is not None) and ("MODEL_FULL_LOWRES" in specs):
        # save the full spectrum to a binary FITS table
        otable_lowres = QTable()
        otable_lowres["WAVELENGTH"] = Column(wave_rebin_lowres, unit=u.angstrom)
        otable_lowres["FLUX"] = Column(flux_rebin_lowres, unit=fluxunit)
        otable_lowres["SIGMA"] = Column(flux_rebin_lowres * 0.01, unit=fluxunit)
        otable_lowres["NPTS"] = Column(npts_rebin_lowres)
        otable_lowres.write(
            "%s/Models/%s" % (output_path, full_file_lowres), overwrite=True
        )
    specinfo["MODEL_FULL_LOWRES"] = full_file_lowres

    full_file = "%s_full.fits" % (output_filebase)
    # save the full spectrum to a binary FITS table
    if specs is not None:
        otable = QTable()
        otable["WAVELENGTH"] = Column(wave_rebin, unit=u.angstrom)
        otable["FLUX"] = Column(flux_rebin, unit=fluxunit)
        otable["SIGMA"] = Column(flux_rebin * 0.01, unit=fluxunit)
        otable["NPTS"] = Column(npts_rebin)
        if "MODEL_FULL" in specs:
            otable.write("%s/Models/%s" % (output_path, full_file), overwrite=True)
    specinfo["MODEL_FULL"] = full_file

    iue_file = "%s_iue.fits" % (output_filebase)
    if (specs is not None) and ("IUE" in specs):
        # IUE mock observation
        # Resolution approximately 400-600
        iue_fwhm_pix = rbres / 600.0
        g = Gaussian1DKernel(stddev=iue_fwhm_pix / 2.355)
        # Convolve data to give the expected observed spectral resolution
        nflux = convolve(otable["FLUX"].data, g)

        iue_table = QTable()
        iue_table["WAVELENGTH"] = otable["WAVELENGTH"]
        iue_table["FLUX"] = nflux * fluxunit
        # number of models points in the rebinned spectrum
        iue_table["NPTS"] = otable["NPTS"]
        # no error in the models, hence required for the merge function
        iue_table["ERROR"] = Column(np.full((len(iue_table)), 1.0)) * fluxunit

        rb_iue = merge_iue_obsspec([iue_table])
        # set the uncertainties to zero as this is a model
        rb_iue["SIGMA"] = rb_iue["FLUX"] * 0.0
        rb_iue.write("%s/Models/%s" % (output_path, iue_file), overwrite=True)

    specinfo["IUE"] = iue_file

    stis_uv_file = "%s_stis_uv.fits" % (output_filebase)
    stis_opt_file = "%s_stis_opt.fits" % (output_filebase)
    stis_uv_g140l_file = stis_uv_file.replace("uv", "g140l")
    stis_uv_g230l_file = stis_uv_file.replace("uv", "g230l")
    stis_uv_g430l_file = stis_uv_file.replace("uv", "g430l")
    stis_uv_g750l_file = stis_uv_file.replace("uv", "g750l")
    if (specs is not None) and ("STIS" in specs):
        # create the ultraviolet HST/STIS mock observation
        stis_table = mock_stis_data(otable)

        # merged versions
        # UV STIS obs
        rb_stis_uv = merge_stis_obsspec(stis_table[0:2], waveregion="UV")
        rb_stis_uv["SIGMA"] = rb_stis_uv["FLUX"] * 0.0
        rb_stis_uv.write("%s/Models/%s" % (output_path, stis_uv_file), overwrite=True)
        # Optical STIS obs
        rb_stis_opt = merge_stis_obsspec(stis_table[2:4], waveregion="Opt")
        rb_stis_opt["SIGMA"] = rb_stis_opt["FLUX"] * 0.0
        rb_stis_opt.write("%s/Models/%s" % (output_path, stis_opt_file), overwrite=True)

        # individual grating versions
        # UV G140L
        rb_stis_uv_g140l = merge_stis_obsspec([stis_table[0]], waveregion="UV")
        rb_stis_uv_g140l["SIGMA"] = rb_stis_uv_g140l["FLUX"] * 0.0
        rb_stis_uv_g140l.write("%s/Models/%s" % (output_path, stis_uv_g140l_file), overwrite=True)
        # UV G230L
        rb_stis_uv_g230l = merge_stis_obsspec([stis_table[1]], waveregion="UV")
        rb_stis_uv_g230l["SIGMA"] = rb_stis_uv_g230l["FLUX"] * 0.0
        rb_stis_uv_g230l.write("%s/Models/%s" % (output_path, stis_uv_g230l_file), overwrite=True)
        # Opt G430L
        rb_stis_uv_g430l = merge_stis_obsspec([stis_table[2]], waveregion="Opt")
        rb_stis_uv_g430l["SIGMA"] = rb_stis_uv_g430l["FLUX"] * 0.0
        rb_stis_uv_g430l.write("%s/Models/%s" % (output_path, stis_uv_g430l_file), overwrite=True)
        # Opt G750L
        rb_stis_uv_g750l = merge_stis_obsspec([stis_table[3]], waveregion="Opt")
        rb_stis_uv_g750l["SIGMA"] = rb_stis_uv_g750l["FLUX"] * 0.0
        rb_stis_uv_g750l.write("%s/Models/%s" % (output_path, stis_uv_g750l_file), overwrite=True)

    specinfo["STIS"] = stis_uv_file
    specinfo["STIS_Opt"] = stis_opt_file
    specinfo["STIS_G140L"] = stis_uv_g140l_file
    specinfo["STIS_G230L"] = stis_uv_g230l_file
    specinfo["STIS_G430L"] = stis_uv_g430l_file
    specinfo["STIS_G750L"] = stis_uv_g750l_file

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(wave_rebin * 1e-4, flux_rebin * 2.0, "b-")
        ax.plot(bandinfo.waves, bandinfo.fluxes, "ro")

        if (specs is not None) and ("STIS" in specs):
            (indxs,) = np.where(rb_stis_uv["NPTS"] > 0)
            ax.plot(
                rb_stis_uv["WAVELENGTH"][indxs].to(u.micron),
                rb_stis_uv["FLUX"][indxs],
                "m-",
            )
            (indxs,) = np.where(rb_stis_opt["NPTS"] > 0)
            ax.plot(
                rb_stis_opt["WAVELENGTH"][indxs].to(u.micron),
                rb_stis_opt["FLUX"][indxs],
                "g-",
            )

        if (specs is not None) and ("IUE" in specs):
            (indxs,) = np.where(rb_iue["NPTS"] > 0)
            ax.plot(
                rb_iue["WAVELENGTH"][indxs].to(u.micron), rb_iue["FLUX"][indxs], "r-"
            )

    for cspec in obsspecinfo.keys():
        cres = obsspecinfo[cspec][0]
        ofile = f"{output_filebase}_{cspec}.fits"
        if (specs is not None) and (cspec.upper() in specs):
            fwhm_pix = rbres / cres
            g = Gaussian1DKernel(stddev=fwhm_pix / 2.355)
            nflux = convolve(otable["FLUX"].data, g)

            outtable = QTable()
            outtable["WAVELENGTH"] = otable["WAVELENGTH"]
            outtable["FLUX"] = nflux * fluxunit
            outtable["NPTS"] = otable["NPTS"]
            outtable["ERROR"] = Column(np.full((len(otable)), 1.0)) * fluxunit

            rb_info = merge_gen_obsspec(
                [outtable], obsspecinfo[cspec][1], output_resolution=cres
            )

            rb_info["SIGMA"] = rb_info["FLUX"] * 0.0
            rb_info.write(f"{output_path}/Models/{ofile}", overwrite=True)

            if show_plot:
                gvals = rb_info["NPTS"] > 0
                ax.plot(
                    rb_info["WAVELENGTH"][gvals].to(u.micron),
                    rb_info["FLUX"][gvals],
                    "c-",
                )

        specinfo[cspec.upper()] = ofile

    # create the DAT file
    dat_filename = "%s/Models/%s.dat" % (output_path, output_filebase)
    header_info = [
        "# obsdata created from %s model atmosphere" % model_type,
        "# %s" % (output_filebase),
        "# file created by make_obsdata_from_model.py",
        "model_type = %s" % model_type,
    ]
    write_dat_file(
        dat_filename,
        bandinfo,
        specinfo,
        modelparams=model_params,
        header_info=header_info,
    )

    if show_plot:
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cont", help="use the continuum only", action="store_true")
    args = parser.parse_args()

    if args.cont:
        ftype = "cont"
        extstr = "_cont"
    else:
        ftype = "spec"
        extstr = ""

    mname = (
        f"/home/kgordon/Python/extstar_data/Models/Tlusty_2025/z100t30000g400v2.{ftype}.gz"
    )

    model_params = {}
    model_params["origin"] = "tlusty"
    model_params["Teff"] = 30000.0
    model_params["logg"] = 4.0
    model_params["Z"] = 1.00
    model_params["vturb"] = 2.0
    make_obsdata_from_model(
        mname,
        model_type="tlusty",
        output_filebase=f"z100t30000g400v2{extstr}",
        output_path="/home/kgordon/Python/extstar_data",
        model_params=model_params,
        show_plot=True,
        # specs="all",
        # specs=["WFC3_G102", "WFC3_G141"],
    )
