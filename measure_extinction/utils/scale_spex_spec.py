#!/usr/bin/env python

from synphot import SpectralElement, SourceSpectrum, Observation
from measure_extinction.stardata import StarData
from synphot.models import Empirical1D

import argparse
import numpy as np
import astropy.units as u

from measure_extinction.utils.helpers import get_datapath


# function to get photometry from a spectrum
def get_phot(spec, bands):
    """
    Compute the fluxes in the requested bands.

    Parameters
    ----------
    spec : SpecData object
        the spectrum

    bands: list of strings
        bands requested

    Outputs
    -------
    band fluxes : numpy array
        calculated band fluxes
    """

    # create a SourceSpectrum object from the spectrum, excluding the bad regions
    spectrum = SourceSpectrum(
        Empirical1D,
        points=spec.waves.to(u.Angstrom)[spec.npts != 0],
        lookup_table=spec.fluxes[spec.npts != 0],
    )

    # path for band response curves
    band_path = f"{get_datapath()}/Band_RespCurves/"

    # dictionary linking the bands to their response curves
    bandnames = {
        "J": "2MASSJ",
        "H": "2MASSH",
        "K": "2MASSKs",
        "IRAC1": "IRAC1",
        "IRAC2": "IRAC2",
        "WISE1": "WISE1",
        "WISE2": "WISE2",
        "L": "AAOL",
        "M": "AAOM",
    }

    # define the units of the output fluxes
    funit = u.erg / (u.s * u.cm * u.cm * u.Angstrom)

    # compute the flux in each band
    fluxes = np.zeros(len(bands))
    for k, band in enumerate(bands):
        # create the bandpass (as a SpectralElement object)
        bp = SpectralElement.from_file(
            "%s%s.dat" % (band_path, bandnames[band])
        )  # assumes wavelengths are in Angstrom!!
        # integrate the spectrum over the bandpass, only if the bandpass fully overlaps with the spectrum (this actually excludes WISE2)
        if bp.check_overlap(spectrum) == "full":
            obs = Observation(spectrum, bp)
            fluxes[k] = obs.effstim(funit).value
        else:
            fluxes[k] = np.nan
    return fluxes


# function to compare the photometry obtained from the spectrum with the band photometry
# and to calculate an average correction factor for the spectrum
def calc_corfac(star_phot, star_spec, bands):
    """
    Compute the correction factor for the spectrum.

    Parameters
    ----------
    star_phot : BandData object
        band data of the star

    star_spec : SpecData object
        spectral data of the star

    bands: list of strings
        bands to use in the calculation of the correction factor

    Outputs
    -------
    corfac : float
        mean correction factor for the spectrum
    """

    # check which bands have photometry
    bands = [band for band in bands if band in star_phot.band_fluxes.keys()]
    # for LXD spectra: pick the best bands for the scaling
    use_bands = []
    if "IRAC1" in bands:
        use_bands.append("IRAC1")
    elif "WISE1" in bands:
        use_bands.append("WISE1")
    elif "L" in bands:
        use_bands.append("L")
    if "IRAC2" in bands:
        use_bands.append("IRAC2")
    elif "WISE2" in bands:
        use_bands.append("WISE2")
    elif "M" in bands:
        use_bands.append("M")
    if use_bands:
        bands = use_bands
    if not bands:
        print(
            "No photometric data available to scale " + star_spec.type + " spectrum!!"
        )
        return None
    fluxes_bands = np.array([star_phot.band_fluxes[band][0] for band in bands])
    fluxes_spectra = get_phot(star_spec, bands)
    corfacs = fluxes_bands / fluxes_spectra
    return "%.3f" % np.nanmean(corfacs)


# function to read in the available data, calculate the correction factors and save the factors in the data file
def calc_save_corfac_spex(starname, path):
    # read in the data file (do not use the correction factors at this point)
    star_data = StarData("%s.dat" % starname.lower(), path=path, use_corfac=False)
    star_phot = star_data.data["BAND"]

    # if the LXD scaling factor has been set manually, warn the user and do not recalculate the scaling factors.
    if star_data.LXD_man:
        print(
            "The LXD scaling factor has been set manually and the scaling factors (SXD and LXD) will not be recalculated!"
        )
        return

    # check which spectra are available,
    # and calculate the correction factors for the spectra (if they have not been set manually)
    if "SpeX_SXD" in star_data.data.keys():
        star_spec_SXD = star_data.data["SpeX_SXD"]
        corfac_SXD = calc_corfac(star_phot, star_spec_SXD, ["J", "H", "K"])
    else:
        corfac_SXD = None
    if "SpeX_LXD" in star_data.data.keys():
        star_spec_LXD = star_data.data["SpeX_LXD"]
        corfac_LXD = calc_corfac(
            star_phot, star_spec_LXD, ["IRAC1", "IRAC2", "WISE1", "WISE2", "L", "M"]
        )
    else:
        corfac_LXD = None

    # add the correction factors to the data file if not already in there,
    # otherwise overwrite the existing correction factors
    if "SpeX_SXD" in star_data.corfac.keys():
        datafile = open(path + "%s.dat" % starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_SXD" in line:
                star_data.datfile_lines[ln] = (
                    "corfac_spex_SXD = " + str(corfac_SXD) + "\n"
                )
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
        with open(path + "%s.dat" % starname.lower(), "a") as datafile:
            datafile.write("corfac_spex_SXD = " + str(corfac_SXD) + "\n")
    if "SpeX_LXD" in star_data.corfac.keys():
        datafile = open(path + "%s.dat" % starname.lower(), "w")
        for ln, line in enumerate(star_data.datfile_lines):
            if "corfac_spex_LXD" in line:
                star_data.datfile_lines[ln] = (
                    "corfac_spex_LXD = " + str(corfac_LXD) + "\n"
                )
            datafile.write(star_data.datfile_lines[ln])
        datafile.close()
    else:
        with open(path + "%s.dat" % starname.lower(), "a") as datafile:
            datafile.write("corfac_spex_LXD = " + str(corfac_LXD) + "\n")


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()

    parser.add_argument("starname", help="name of the star (filebase)")
    parser.add_argument(
        "--path",
        help="path where data files are stored",
        default=get_datapath(),
    )
    args = parser.parse_args()

    # calculate and save the SpeX correction factors
    calc_save_corfac_spex(args.starname, args.path)
