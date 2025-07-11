import math
import warnings
import os.path

from collections import OrderedDict

import numpy as np

from astropy import constants as const
import astropy.units as u

from dust_extinction.parameter_averages import CCM89
from dust_extinction.shapes import _curve_F99_method

from measure_extinction.merge_obsspec import _wavegrid
from measure_extinction.helpers import (
    fluxunit,
    read_gen_spectra,
    read_spex,
    read_irs,
)

__all__ = ["StarData", "BandData", "SpecData"]

# Jy to ergs/(cm^2 s A)
#   1 Jy = 1e-26 W/(m^2 Hz)
#   1 W = 1e7 ergs
#   1 m^2 = 1e4 cm^2
#   1 um = 1e4 A
#   Hz/um = c[micron/s]/(lambda[micron])^2
# const = 1e-26*1e7*1e-4*1e-4 = 1e-27
Jy_to_cgs_const = 1e-27 * const.c.to("micron/s").value

spec_rinfo = {
    "FUSE": read_gen_spectra,
    "IUE": read_gen_spectra,
    "STIS_Opt": read_gen_spectra,
    "STIS": read_gen_spectra,
    "WFC3_G102": read_gen_spectra,
    "WFC3_G141": read_gen_spectra,
    "SpeX_SXD": read_spex,
    "SpeX_LXD": read_spex,
    "NIRCAM_SS": read_gen_spectra,
    "NIRISS_SOSS": read_gen_spectra,
    "IRS": read_irs,
    "MIRI_LRS": read_gen_spectra,
    "MIRI_IFU": read_gen_spectra,
    "MODEL_FULL_LOWRES": read_gen_spectra,
    "MODEL_FULL": read_gen_spectra,
}


def _getspecfilename(line, path):
    """
    Get the full filename including path from the line in the dat file

    Parameters
    ----------
    line : string
        formated line from DAT file
        example: 'IUE = hd029647_iue.fits'

    path : string
        path of the FITS file

    Returns
    -------
    full_filename : str
        full name of file including path
    """
    eqpos = line.find("=")
    tfile = line[eqpos + 2 :].rstrip()

    return f"{path}/{tfile}"


class BandData:
    """
    Photometric band data (used by StarData)

    Attributes:

    type : string
        descriptive string of type of data (currently always BAND)

    waves : array of floats
        wavelengths

    fluxes : array of floats
        fluxes

    uncs : array of floats
        uncertainties on the fluxes

    npts : array of floats
        number of measurements contributing to the flux points

    n_bands : int
        number of bands

    bands : dict of band_name:measurement
        measurement is a tuple of (value, uncertainty)

    band_units : dict of band_name:strings
        band units ('mag' or 'mJy')

    band_waves : dict of band_name:floats
        band wavelengths in micron

    band_fluxes : dict of band_name:floats
        band fluxes in ergs/(cm^2 s A)
    """

    def __init__(self, type):
        """
        Parameters
        ----------
        type: string
            descriptive string of type of data (currently always BAND)
        """
        self.type = type
        self.n_bands = 0
        self.bands = OrderedDict()
        self.band_units = OrderedDict()
        self.band_waves = OrderedDict()
        self.band_fluxes = OrderedDict()

    def read_bands(self, lines, only_bands=None):
        """
        Read the photometric band data from a DAT file
        and update class variables.
        Bands are filled in wavelength order to make life
        easier in subsequent calculations (interpolations!)

        Parameters
        ----------
        lines : list of string
            lines from a DAT formatted file
        only_bands : list
            Only read in the bands given

        Returns
        -------
        Updates self.bands, self.band_units, and self.n_bands
        """
        for line in lines:
            eqpos = line.find("=")
            pmpos = line.find("+/-")

            if (eqpos >= 0) & (pmpos >= 0) & (line[0] != "#"):
                # check for reference or unit
                colpos = max(
                    (
                        line.find(";"),
                        line.find("#"),
                        line.find("mJy"),
                        line.find("ABmag"),
                    )
                )
                if colpos == -1:
                    colpos = len(line)
                # if there is both a reference and a unit
                elif (line.find(";") != -1) and (line.find("mJy") != -1):
                    colpos = min(line.find(";"), line.find("mJy"))
                elif (line.find(";") != -1) and (line.find("ABmag") != -1):
                    colpos = min(line.find(";"), line.find("ABmag"))
                band_name = line[0:eqpos].strip()

                save_band = False
                if only_bands is None:
                    save_band = True
                else:
                    if band_name in only_bands:
                        save_band = True
                if save_band:
                    self.bands[band_name] = (
                        float(line[eqpos + 1 : pmpos]),
                        float(line[pmpos + 3 : colpos]),
                    )
                    # units
                    if line.find("mJy") >= 0:
                        self.band_units[band_name] = "mJy"
                    elif line.find("ABmag") >= 0:
                        self.band_units[band_name] = "ABmag"
                    else:
                        self.band_units[band_name] = "mag"

        self.n_bands = len(self.bands)

    @staticmethod
    def get_poss_bands():
        """
        Provides the possible bands and bands wavelengths and zeromag fluxes

        Returns
        -------
        band_info : dict of band_name: value
            value is tuple of ( zeromag_flux, wavelength [micron] )
            The zeromag_flux is the flux in erg/cm2/s/A for a star of (Vega) mag=0.
            It gives the conversion factor from Vega magnitudes to erg/cm2/s/A.
        """
        _johnson_band_names = ["U", "B", "V", "R", "I", "J", "H", "K", "L", "M"]
        _johnson_band_waves = np.array(
            [0.366, 0.438, 0.545, 0.641, 0.798, 1.22, 1.63, 2.19, 3.45, 4.75]
        )
        _johnson_band_zeromag_fluxes = (
            np.array(
                [417.5, 632.0, 363.1, 217.7, 112.6, 31.47, 11.38, 3.961, 0.699, 0.204]
            )
            * 1e-11
        )

        _spitzer_band_names = ["IRAC1", "IRAC2", "IRAC3", "IRAC4", "IRS15", "MIPS24"]
        _spitzer_band_waves = np.array([3.52, 4.45, 5.66, 7.67, 15.4, 23.36])
        _spitzer_band_zeromag_fluxes = (
            np.array([0.6605, 0.2668, 0.1069, 0.03055, 1.941e-3, 3.831e-4]) * 1e-11
        )

        # WISE bands. Wavelenghts and zeropoints are taken from http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=WISE.
        _wise_band_names = ["WISE1", "WISE2", "WISE3", "WISE4"]
        _wise_band_waves = np.array([3.3526, 4.6028, 11.5608, 22.0883])
        _wise_band_zeromag_fluxes = np.array(
            [8.1787e-12, 2.415e-12, 6.5151e-14, 5.0901e-15]
        )

        # JWST/MIRI bands
        _miri_band_names = ["MIRI_FND"]
        _miri_band_waves = np.array([12.9])
        _miri_band_zeromag_fluxes = np.array([5.175e-14])

        # WFPC2 bands
        _wfpc2_band_names = [
            "WFPC2_F170W",
            "WFPC2_F255W",
            "WFPC2_F336W",
            "WFPC2_F439W",
            "WFPC2_F555W",
            "WFPC2_F814W",
        ]
        _wfpc2_band_waves = np.array([0.170, 0.255, 0.336, 0.439, 0.555, 0.814])
        _wfpc2_photflam = np.array(
            [1.551e-15, 5.736e-16, 5.613e-17, 2.945e-17, 3.483e-18, 2.508e-18]
        )
        _wfpc2_vegamag = np.array([16.335, 17.019, 19.429, 20.884, 22.545, 21.639])
        _n_wfpc2_bands = len(_wfpc2_vegamag)
        _wfpc2_band_zeromag_fluxes = np.zeros(_n_wfpc2_bands)

        # zeromag Vega flux not given in standard WFPC2 documentation
        # instead the flux and Vega magnitudes are given for 1 DN/sec
        # the following code coverts these numbers to zeromag Vega fluxes
        for i in range(_n_wfpc2_bands):
            _wfpc2_band_zeromag_fluxes[i] = _wfpc2_photflam[i] * (
                10 ** (0.4 * _wfpc2_vegamag[i])
            )

        # WFC3 bands - updated for 2020 solutions  Assuming UVIS1
        _wfc3_band_names = [
            "WFC3_F225W",
            "WFC3_F275W",
            "WFC3_F336W",
            "WFC3_F475W",
            "WFC3_F625W",
            "WFC3_F775W",
            "WFC3_F814W",
            "WFC3_F110W",
            "WFC3_F160W",
        ]
        # from Calamida et al. 2022 (Amp C)
        _wfc3_band_waves = np.array(
            [
                0.235839,
                0.270330,
                0.335465,
                0.477217,
                0.624196,
                0.764830,
                0.802932,
                1.153446,
                1.536918,
            ]
        )

        _wfc3_photflam = np.array(
            [
                4.585e-18,
                3.223e-18,
                1.286e-18,
                2.498e-19,
                1.723e-19,
                2.093e-18,
                1.477e-19,
                1.53e-20,
                1.93e-20,
            ]
        )
        _wfc3_vegamag = np.array(
            [22.426, 22.676, 23.526, 25.809, 25.374, 24.84, 24.712, 26.063, 24.695]
        )
        _n_wfc3_bands = len(_wfc3_vegamag)
        _wfc3_band_zeromag_fluxes = np.zeros(_n_wfc3_bands)

        # zeromag Vega flux not given in standard WFC3 documentation
        # instead the flux and Vega magnitudes are given for 1 DN/sec
        # the following code coverts these numbers to zeromag Vega fluxes
        for i in range(_n_wfc3_bands):
            _wfc3_band_zeromag_fluxes[i] = _wfc3_photflam[i] * (
                10 ** (0.4 * _wfc3_vegamag[i])
            )

        # ACS bands
        _acs_band_names = ["ACS_F475W", "ACS_F814W"]
        _acs_band_waves = np.array([0.4746, 0.8045])
        _acs_photflam = np.array([1.827e-19, 7.045e-20])
        _acs_vegamag = np.array([26.149, 25.517])
        _n_acs_bands = len(_acs_vegamag)
        _acs_band_zeromag_fluxes = np.zeros(_n_acs_bands)

        # zeromag Vega flux not given in standard WFPC2 documenation
        # instead the flux and Vega magnitudes are given for 1 DN/sec
        # the following code coverts these numbers to zeromag Vega fluxes
        for i in range(_n_acs_bands):
            _acs_band_zeromag_fluxes[i] = _acs_photflam[i] * (
                10 ** (0.4 * _acs_vegamag[i])
            )

        # combine all the possible
        _poss_band_names = np.concatenate(
            [
                _johnson_band_names,
                _spitzer_band_names,
                _wise_band_names,
                _miri_band_names,
                _wfpc2_band_names,
                _wfc3_band_names,
                _acs_band_names,
            ]
        )
        _poss_band_waves = np.concatenate(
            [
                _johnson_band_waves,
                _spitzer_band_waves,
                _wise_band_waves,
                _miri_band_waves,
                _wfpc2_band_waves,
                _wfc3_band_waves,
                _acs_band_waves,
            ]
        )
        _poss_band_zeromag_fluxes = np.concatenate(
            [
                _johnson_band_zeromag_fluxes,
                _spitzer_band_zeromag_fluxes,
                _wise_band_zeromag_fluxes,
                _miri_band_zeromag_fluxes,
                _wfpc2_band_zeromag_fluxes,
                _wfc3_band_zeromag_fluxes,
                _acs_band_zeromag_fluxes,
            ]
        )

        # zip everything together into a dictonary to pass back
        #   and make sure it is in wavelength order
        windxs = np.argsort(_poss_band_waves)
        return OrderedDict(
            zip(
                _poss_band_names[windxs],
                zip(_poss_band_zeromag_fluxes[windxs], _poss_band_waves[windxs]),
            )
        )

    def get_band_names(self):
        """
        Get the names of the bands in the data

        Returns
        -------
        names : string array
            names of the bands in the data
        """
        pbands = self.get_poss_bands()
        gbands = []
        for cband in pbands.keys():
            mag = self.get_band_mag(cband)
            if mag is not None:
                gbands.append(cband)

        return gbands

    def get_band_mag(self, band_name):
        """
        Get the magnitude and uncertainties for a band based
        on measurements colors and V band or direct measurements in the band

        Parameters
        ----------
        band_name : str
            name of the band

        Returns
        -------
        info: tuple
           (mag, unc, unit)
        """
        band_data = self.bands.get(band_name)
        if band_data is not None:
            if self.band_units[band_name] == "mJy":
                # case of the IRAC/MIPS photometry
                # need to convert to Vega magnitudes
                pbands = self.get_poss_bands()
                band_zflux_wave = pbands[band_name]

                flux_p_unc = self.bands[band_name]
                flux = (1e-3 * Jy_to_cgs_const / band_zflux_wave[1] ** 2) * flux_p_unc[
                    0
                ]
                mag_unc = -2.5 * np.log10(
                    (flux_p_unc[0] - flux_p_unc[1]) / (flux_p_unc[0] + flux_p_unc[1])
                )
                mag = -2.5 * np.log10(flux / band_zflux_wave[0])

                return (mag, mag_unc, "mag")
            else:
                return self.bands[band_name] + (self.band_units[band_name],)
        else:
            _mag = 0.0
            _mag_unc = 0.0
            if band_name == "U":
                if (
                    (self.bands.get("V") is not None)
                    & (self.bands.get("(B-V)") is not None)
                    & (self.bands.get("(U-B)") is not None)
                ):
                    _mag = (
                        self.bands["V"][0]
                        + self.bands["(B-V)"][0]
                        + self.bands["(U-B)"][0]
                    )
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2
                        + self.bands["(B-V)"][1] ** 2
                        + self.bands["(U-B)"][1] ** 2
                    )
                elif (self.bands.get("V") is not None) & (
                    self.bands.get("(U-V)") is not None
                ):
                    _mag = self.bands["V"][0] + self.bands["(U-V)"][0]
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2 + self.bands["(U-V)"][1] ** 2
                    )
            elif band_name == "B":
                if (self.bands.get("V") is not None) & (
                    self.bands.get("(B-V)") is not None
                ):
                    _mag = self.bands["V"][0] + self.bands["(B-V)"][0]
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2 + self.bands["(B-V)"][1] ** 2
                    )
            elif band_name == "R":
                if (self.bands.get("V") is not None) & (
                    self.bands.get("(V-R)") is not None
                ):
                    _mag = self.bands["V"][0] - self.bands["(V-R)"][0]
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2 + self.bands["(V-R)"][1] ** 2
                    )
            elif band_name == "I":
                if (self.bands.get("V") is not None) & (
                    self.bands.get("(V-I)") is not None
                ):
                    _mag = self.bands["V"][0] - self.bands["(V-I)"][0]
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2 + self.bands["(V-I)"][1] ** 2
                    )
                elif (
                    (self.bands.get("V") is not None)
                    & (self.bands.get("(V-R)") is not None)
                    & (self.bands.get("(R-I)") is not None)
                ):
                    _mag = (
                        self.bands["V"][0]
                        - self.bands["(V-R)"][0]
                        - self.bands["(R-I)"][0]
                    )
                    _mag_unc = math.sqrt(
                        self.bands["V"][1] ** 2
                        + self.bands["(V-R)"][1] ** 2
                        + self.bands["(R-I)"][1] ** 2
                    )
            if (_mag != 0.0) & (_mag_unc != 0.0):
                return (_mag, _mag_unc, "mag")

    def get_band_flux(self, band_name):
        """
        Compute the flux for the input band

        Parameters
        ----------
        band_name : str
            name of the band

        Returns
        -------
        info: tuple
           (flux, unc)
        """
        mag_vals = self.get_band_mag(band_name)
        if mag_vals is not None:
            # get the zero mag fluxes
            poss_bands = self.get_poss_bands()
            if mag_vals[2] == "mag":
                # flux +/- unc
                flux1 = poss_bands[band_name][0] * (
                    10 ** (-0.4 * (mag_vals[0] + mag_vals[1]))
                )
                flux2 = poss_bands[band_name][0] * (
                    10 ** (-0.4 * (mag_vals[0] - mag_vals[1]))
                )
                return (
                    0.5 * (flux1 + flux2),
                    0.5 * (flux2 - flux1),
                    poss_bands[band_name][1],
                )
            elif mag_vals[2] == "mJy":
                mfac = 1e-3 * Jy_to_cgs_const / np.square(poss_bands[band_name][0])
                return (
                    mag_vals[0] * mfac,
                    mag_vals[1] * mfac,
                    poss_bands[band_name][1],
                )
            else:
                warnings.warn("cannot get flux for %s" % band_name, UserWarning)
        else:
            warnings.warn("cannot get flux for %s" % band_name, UserWarning)

    def get_band_fluxes(self):
        """
        Compute the fluxes and uncertainties in each band

        Returns
        -------
        Updates self.band_fluxes and self.band_waves
        Also sets self.(n_waves, waves, fluxes, npts)
        """
        poss_bands = self.get_poss_bands()

        for pband_name in poss_bands.keys():
            _mag_vals = self.get_band_mag(pband_name)
            if _mag_vals is not None:
                if _mag_vals[2] == "mag":
                    _flux1 = poss_bands[pband_name][0] * (
                        10 ** (-0.4 * (_mag_vals[0] + _mag_vals[1]))
                    )
                    _flux2 = poss_bands[pband_name][0] * (
                        10 ** (-0.4 * (_mag_vals[0] - _mag_vals[1]))
                    )
                    self.band_fluxes[pband_name] = (
                        0.5 * (_flux1 + _flux2),
                        0.5 * (_flux2 - _flux1),
                    )
                    self.band_waves[pband_name] = poss_bands[pband_name][1]
                elif _mag_vals[2] == "ABmag":
                    _flux1_nu = np.power(
                        10.0, (-0.4 * (_mag_vals[0] + _mag_vals[1] + 48.60))
                    )
                    _flux1_nu *= u.erg / (u.cm * u.cm * u.s * u.Hz)
                    _flux1 = _flux1_nu.to(
                        fluxunit,
                        equivalencies=u.spectral_density(
                            poss_bands[pband_name][1] * u.micron
                        ),
                    )
                    _flux2_nu = np.power(
                        10.0, (-0.4 * (_mag_vals[0] - _mag_vals[1] + 48.60))
                    )
                    _flux2_nu *= u.erg / (u.cm * u.cm * u.s * u.Hz)
                    _flux2 = _flux2_nu.to(
                        fluxunit,
                        equivalencies=u.spectral_density(
                            poss_bands[pband_name][1] * u.micron
                        ),
                    )
                    self.band_fluxes[pband_name] = (
                        0.5 * (_flux1.value + _flux2.value),
                        0.5 * (_flux2.value - _flux1.value),
                    )
                    self.band_waves[pband_name] = poss_bands[pband_name][1]
                elif _mag_vals[2] == "mJy":
                    self.band_waves[pband_name] = poss_bands[pband_name][1]
                    mfac = (
                        1e-3 * Jy_to_cgs_const / np.square(self.band_waves[pband_name])
                    )
                    self.band_fluxes[pband_name] = (
                        _mag_vals[0] * mfac,
                        _mag_vals[1] * mfac,
                    )
                else:
                    warnings.warn("cannot get flux for %s" % pband_name, UserWarning)
        # also store the band data in flat numpy vectors for
        #   computational speed in fitting routines
        # mimics the format of the spectral data
        self.n_waves = len(self.band_waves)
        self.waves = np.zeros(self.n_waves)
        self.fluxes = np.zeros(self.n_waves)
        self.uncs = np.zeros(self.n_waves)
        self.npts = np.full(self.n_waves, 1.0)

        for k, pband_name in enumerate(self.band_waves.keys()):
            self.waves[k] = self.band_waves[pband_name]
            self.fluxes[k] = self.band_fluxes[pband_name][0]
            self.uncs[k] = self.band_fluxes[pband_name][1]

        self.wave_range = [min(self.waves), max(self.waves)]

        # add the units
        self.waves = self.waves * u.micron
        self.wave_range = self.wave_range * u.micron
        self.fluxes = self.fluxes * (u.erg / ((u.cm**2) * u.s * u.angstrom))
        self.uncs = self.uncs * (u.erg / ((u.cm**2) * u.s * u.angstrom))

    def get_band_mags_from_fluxes(self):
        """
        Compute the magnitudes from fluxes in each band
        Useful for generating "observed data" from models

        Returns
        -------
        Updates self.bands and self.band_units
        """
        poss_bands = self.get_poss_bands()

        for cband in self.band_fluxes.keys():
            if cband in poss_bands.keys():
                self.bands[cband] = (
                    -2.5 * np.log10(self.band_fluxes[cband][0] / poss_bands[cband][0]),
                    0.0,
                )
                self.band_waves[cband] = poss_bands[cband][1]
                self.band_units[cband] = "mag"
            else:
                warnings.warn("cannot get mag for %s" % cband, UserWarning)

        self.n_bands = len(self.bands)


class SpecData:
    """
    Spectroscopic data (used by StarData)

    Attributes:

    waves : array of floats
        wavelengths

    fluxes : array of floats
        fluxes

    uncs : array of floats
        uncertainties on the fluxes

    npts : array of floats
        number of measurements contributing to the flux points

    n_waves : int
        number of wavelengths

    wmin, wmax : floats
        wavelength min and max
    """

    # -----------------------------------------------------------
    def __init__(self, type):
        """
        Parameters
        ----------
        type: string
            descriptive string of type of data (e.g., IUE, FUSE, IRS)
        """
        self.type = type
        self.n_waves = 0

    def rebin_constres(self, waverange, resolution):
        """
        Rebin the spectrum to a fixed spectral resolution
        and min/max wavelength range.

        Parameters
        ----------
        waverange : 2 element array of astropy Quantities
            Min/max of wavelength range with units
        resolution : float
            Spectral resolution of rebinned spectrum

        Returns
        -------
        measure_extinction SpecData
            Object with rebinned spectrum

        """
        # setup new wavelength grid
        full_wave, full_wave_min, full_wave_max = _wavegrid(
            resolution, waverange.to(u.micron).value
        )
        n_waves = len(full_wave)

        # setup the new rebinned vectors
        new_waves = full_wave * u.micron
        new_fluxes = np.zeros((n_waves), dtype=float)
        new_uncs = np.zeros((n_waves), dtype=float)
        new_npts = np.zeros((n_waves), dtype=int)

        # rebin using a weighted average
        owaves = self.waves.to(u.micron).value
        for k in range(n_waves):
            # check for zero uncs to avoid divide by zero
            # errors when the flux uncertainty of a real measurement
            # is zero for any reason
            (indxs,) = np.where(
                (owaves >= full_wave_min[k])
                & (owaves < full_wave_max[k])
                & (self.npts > 0.0)
                & (self.uncs > 0.0)
            )
            if len(indxs) > 0:
                weights = 1.0 / np.square(self.uncs[indxs].value)
                sweights = np.sum(weights)
                new_fluxes[k] = np.sum(weights * self.fluxes[indxs].value) / sweights
                new_uncs[k] = 1.0 / np.sqrt(sweights)
                new_npts[k] = np.sum(self.npts[indxs])

        # update values
        self.waves = new_waves
        self.fluxes = new_fluxes * self.fluxes.unit
        self.uncs = new_uncs * self.uncs.unit
        self.npts = new_npts


class StarData:
    """
    Photometric and spectroscopic data for a star

    Attributes
    ----------
    file : string
        DAT filename

    path : string
        DAT filename path

    sptype : string
        spectral type of star

    model_params : dict
        has the stellar atmosphere model parameters
        empty dict if observed data

    data : dict of key:BandData or SpecData
        key gives the type of data (e.g., BAND, IUE, IRS)

    photonly: boolean
        Only read in the photometry (no spectroscopy)

    corfac : dict of key:correction factors
        key gives the type (e.g., IRS, IRS_slope)

    use_corfac : boolean
        whether or not to use the correction factors, default = True

    LXD_man : boolean
        whether or not the LXD scaling factor has been set manually, default = False
    """

    def __init__(
        self,
        datfile,
        path="",
        photonly=False,
        use_corfac=True,
        deredden=False,
        only_bands=None,
        only_data="ALL",
    ):
        """
        Parameters
        ----------
        datfile: string
            filename of the DAT file

        path: string, optional
            DAT file path

        photonly: boolean
            Only read in the photometry (no spectroscopy)

        use_corfac: boolean
            Modify the spectra based on precomputed correction factors
            Currently only affects Spitzer/IRS data and SpeX data

        deredden : boolean [default=False]
           Deredden the data based on dereddening parameters given in the DAT file.
           Generally used to deredden standards.

        only_bands : list
            Only read in the bands given

         only_data : list
            Only read in the data given
        """
        self.file = datfile
        self.path = path
        self.sptype = ""
        self.model_params = {}
        self.data = {}
        self.corfac = {}
        self.deredden_params = {}
        self.photonly = photonly
        self.use_corfac = use_corfac
        self.LXD_man = False
        self.dereddened = deredden

        if self.file is not None:
            self.read(deredden=deredden, only_bands=only_bands, only_data=only_data)

    def read(self, deredden=False, only_bands=None, only_data="ALL"):
        """
        Populate the object from a DAT file + spectral files

        Parameters
        ----------
        deredden : boolean [default=False]
           Deredden the data based on dereddening parameters given in the DAT file.
           Generally used to deredden standards.
        only_bands : list
            Only read in the bands given
        only_data : list
            Only read in the data given
        """

        # open and read all the lines in the file
        f = open(f"{self.path}/{self.file}", "r")
        self.datfile_lines = list(f)
        f.close()

        # get the photometric band data
        self.data["BAND"] = BandData("BAND")
        self.data["BAND"].read_bands(self.datfile_lines, only_bands=only_bands)

        if self.data["BAND"].n_bands == 0:
            # no bands, delete BAND entry
            del self.data["BAND"]
        else:
            # covert the photoemtric band data to fluxes in all possible bands
            self.data["BAND"].get_band_fluxes()

        # go through and get info before reading the spectra
        poss_mod_params = [
            "model_type",
            "Z",
            "vturb",
            "logg",
            "logg_unc",
            "Teff",
            "Teff_unc",
            "velocity",
            "origin",
        ]
        for line in self.datfile_lines:
            cpair = self._parse_dfile_line(line)
            if cpair is not None:
                if cpair[0] == "sptype":
                    self.sptype = cpair[1]
                elif cpair[0] in poss_mod_params:
                    self.model_params[cpair[0]] = cpair[1]
                elif cpair[0] == "corfac_spex_SXD":
                    self.corfac["SpeX_SXD"] = eval(cpair[1])
                elif cpair[0] == "corfac_spex_LXD":
                    self.corfac["SpeX_LXD"] = eval(cpair[1])
                elif cpair[0] == "LXD_man":
                    self.LXD_man = eval(cpair[1])
                elif cpair[0] == "corfac_irs_zerowave":
                    self.corfac["IRS_zerowave"] = float(cpair[1])
                elif cpair[0] == "corfac_irs_slope":
                    self.corfac["IRS_slope"] = float(cpair[1])
                elif cpair[0] == "corfac_irs_maxwave":
                    self.corfac["IRS_maxwave"] = float(cpair[1])
                elif cpair[0] == "corfac_irs":
                    self.corfac["IRS"] = float(cpair[1])
                elif cpair[0].find("dered") != -1:
                    if cpair[0].find("RV") != -1:
                        self.deredden_params["RV"] = float(cpair[1])
                    elif cpair[0].find("AV") != -1:
                        self.deredden_params["AV"] = float(cpair[1])
                    elif cpair[0].find("FM") != -1:
                        self.deredden_params["FM90"] = [
                            float(cfm) for cfm in cpair[1].split("  ")
                        ]

        # read the spectra
        if not self.photonly:
            for line in self.datfile_lines:
                if (line[0] == "#") or ("IRS15" in line):
                    pass
                else:
                    for cspec in spec_rinfo.keys():
                        if line.upper().startswith(cspec.upper()) and (
                            cspec in only_data or only_data == "ALL"
                        ):
                            print(f"found {cspec}")
                            fname = _getspecfilename(line, self.path)
                            if os.path.isfile(fname):
                                self.data[cspec] = SpecData(cspec)
                                spec_rinfo[cspec](
                                    self.data[cspec],
                                    fname,
                                    use_corfac=self.use_corfac,
                                    corfac=self.corfac,
                                )
                            else:
                                warnings.warn(f"{fname} does not exist", UserWarning)
                                exit()
                            break

        # if desired and the necessary dereddening parameters are present
        if deredden:
            self.deredden()

    @staticmethod
    def _parse_dfile_line(line):
        """
        Parses a string and return key, value pair.
        Pair separated by '=' and ends with ';'.

        Parameters
        ----------
        line : string
            DAT file formatted string

        Returns
        -------
        substring : string
            The value substring in a DAT file formatted string
        """
        if line[0] != "#":
            eqpos = line.find("=")
            if eqpos >= 0:
                colpos = line.find(";")
                if colpos == 0:
                    colpos = len(line)
                return (line[0 : eqpos - 1].strip(), line[eqpos + 1 : colpos].strip())

    def deredden(self):
        """
        Remove the effects of extinction from all the data.
        Prime use to deredden a standard star for a small amount of extinction.
        Information on the extinction curve to use is given in the DAT_file.
        Uses FM90 parameters for the UV portion of the extinction curve
        and CCM89 extinction values for the optical/NIR based on R(V).
        A(V) values used for dust column.
        """
        if self.deredden_params:
            # deredden the BANDs and IUE/STIS/FUSE data
            #  will need to add other options if/when dereddening expanded

            # info for dereddening model (F99 method)
            optnirext = CCM89()
            gvals = self.data["BAND"].waves < 3.0 * u.micron
            optnir_axav_x = 1.0 / self.data["BAND"].waves[gvals]
            optnir_axav_y = optnirext(optnir_axav_x)
            fm = self.deredden_params["FM90"]
            optnir_sindxs = np.argsort(optnir_axav_x)

            xrange = np.flip(1.0 / np.array(optnirext.x_range)) * u.micron
            for curtype in self.data.keys():
                cwaves = self.data[curtype].waves
                gvals = (cwaves > xrange[0]) & (cwaves < xrange[1])
                if np.any(gvals):
                    alav = _curve_F99_method(
                        self.data[curtype].waves[gvals],
                        self.deredden_params["RV"],
                        fm[0],
                        fm[1],
                        fm[2],
                        fm[3],
                        fm[4],
                        fm[5],
                        optnir_axav_x.value[optnir_sindxs],
                        optnir_axav_y[optnir_sindxs],
                        optnirext.x_range,
                        "F99_method",
                    )
                    self.data[curtype].fluxes[gvals] /= 10 ** (
                        -0.4 * alav * self.deredden_params["AV"]
                    )
                else:
                    warnings.warn(f"{curtype} cannot be dereddened", UserWarning)
        else:
            warnings.warn(
                "cannot deredden as no dereddening parameters set", UserWarning
            )

    def get_flat_data_arrays(self, req_datasources):
        """
        Get the data in a simple format

        Parameters
        ----------
        req_datasources : list of str
            list of data sources (e.g., ['IUE', 'BAND'])

        Returns
        -------
        (waves, fluxes, uncs) : tuple of numpy.ndarrays
            arrays are sorted from short to long wavelengths
            waves is wavelengths in microns
            fluxes is fluxes in erg/cm2/s/A
            uncs is uncertainties on flux in erg/cm2/s/A
        """
        wavedata = []
        fluxdata = []
        uncdata = []
        nptsdata = []
        for data_source in req_datasources:
            if data_source in self.data.keys():
                wavedata.append(self.data[data_source].waves.to(u.micron).value)
                fluxdata.append(
                    self.data[data_source]
                    .fluxes.to(
                        fluxunit,
                        equivalencies=u.spectral_density(self.data[data_source].waves),
                    )
                    .value
                )
                uncdata.append(
                    self.data[data_source]
                    .uncs.to(
                        fluxunit,
                        equivalencies=u.spectral_density(self.data[data_source].waves),
                    )
                    .value
                )
                nptsdata.append(self.data[data_source].npts)
        waves = np.concatenate(wavedata)
        fluxes = np.concatenate(fluxdata)
        uncs = np.concatenate(uncdata)
        npts = np.concatenate(nptsdata)

        # sort the arrays from short to long wavelengths
        # at the same time, remove points with no data
        (gindxs,) = np.where(npts > 0)
        sindxs = np.argsort(waves[gindxs])
        gindxs = gindxs[sindxs]
        waves = waves[gindxs]
        fluxes = fluxes[gindxs]
        uncs = uncs[gindxs]

        return (waves, fluxes, uncs)

    def plot(
        self,
        ax,
        pcolor=None,
        norm_wave_range=None,
        mlam4=False,
        wavenum=False,
        fluxunit=fluxunit,
        exclude=[],
        yoffset=None,
        yoffset_type="multiply",
        annotate_key=None,
        annotate_wave_range=None,
        annotate_alignment="left",
        annotate_text=None,
        annotate_rotation=0.0,
        annotate_yoffset=0.0,
        annotate_color="k",
        legend_key=None,
        legend_label=None,
        fontsize=None,
    ):
        """
        Plot all the data for a star (bands and spectra)

        Parameters
        ----------
        ax : matplotlib plot object

        pcolor : matplotlib color [default=None]
            color to use for all the data

        norm_wave_range : list of 2 floats [default=None]
            min/max wavelength range to use to normalize data

        mlam4 : boolean [default=False]
            plot the data multiplied by lambda^4 to remove the Rayleigh-Jeans slope

        wavenum : boolean [default=False]
            plot x axis as 1/wavelength as is standard for UV extinction curves

        fluxunit : astropy unit
            flux units for plot, default is ergs/(cm^2 s A)

        exclude : list of strings [default=[]]
            List of data type(s) to exclude from the plot (e.g., "IRS", "IRAC1",...)

        yoffset : float [default=None]
            multiplicative or additive offset for the data

        yoffset_type : str [default="multiply"]
            yoffset type: "multiply" or "add"

        annotate_key : string [default=None]
            type of data for which to annotate text (e.g., SpeX_LXD)

        annotate_wave_range : list of 2 floats [default=None]
            min/max wavelength range for the annotation of the text

        annotate_alignment : string [default="left"]
            horizontal alignment of the annotated text ("left", "center", "right")

        annotate_text : string [default=None]
            text to annotate

        annotate_rotation : float [default=0.0]
            rotation angle of the annotated text

        annotate_yoffset : float [default=0.0]
            y-offset for the annotated text

        annotate_color : string [default="k"]
            color of the annotated text

        legend_key : string [default=None]
            legend the spectrum using the given data key

        legend_label : string [default=None]
            label to use for legend

        fontsize : int [default=None]
            fontsize for plot
        """

        if yoffset is None:
            if yoffset_type == "multiply":
                yoffset = 1.0
            else:
                yoffset = 0.0

        # find the data to use for the normalization if requested
        if norm_wave_range is not None:
            normtype = None
            for curtype in self.data.keys():
                if (norm_wave_range[0] >= self.data[curtype].wave_range[0]) & (
                    (norm_wave_range[1] <= self.data[curtype].wave_range[1])
                ):
                    # prioritize spectra over photometric bands
                    if normtype is not None:
                        if normtype == "BAND":
                            normtype = curtype
                    else:
                        normtype = curtype
            if normtype is None:
                return
                # raise ValueError("requested normalization range not valid")

            (gindxs,) = np.where(
                (self.data[normtype].npts > 0)
                & (
                    (self.data[normtype].waves >= norm_wave_range[0])
                    & (self.data[normtype].waves <= norm_wave_range[1])
                )
            )

            if len(gindxs) > 0:
                waves = self.data[normtype].waves[gindxs]
                fluxes = (
                    self.data[normtype]
                    .fluxes[gindxs]
                    .to(fluxunit, equivalencies=u.spectral_density(waves))
                    .value
                )

                if mlam4:
                    ymult = np.power(waves.value, 4.0)
                else:
                    ymult = np.full((len(waves.value)), 1.0)
                normval = np.average(fluxes * ymult)
            else:
                raise ValueError("no good data in requested norm range")
        else:
            normval = 1.0

        # plot all band and spectral data for this star
        for curtype in self.data.keys():
            # do not plot the excluded data type(s)
            if curtype in exclude:
                continue

            x = self.data[curtype].waves.value
            if wavenum:
                x = 1.0 / x

            # replace fluxes by NaNs for wavelength regions that need to be excluded from the plot, to avoid separate regions being connected artificially
            self.data[curtype].fluxes[self.data[curtype].npts == 0] = np.nan
            if mlam4:
                ymult = np.power(self.data[curtype].waves.value, 4.0)
            else:
                ymult = np.full((len(self.data[curtype].waves.value)), 1.0)
            # multiply by the overall normalization
            ymult /= normval

            if legend_key == curtype:
                if legend_label is None:
                    red_name = self.file.replace(".dat", "")
                    red_name = red_name.replace("DAT_files/", "")
                    clabel = "%s / %s" % (red_name, self.sptype)
                else:
                    clabel = legend_label
            else:
                clabel = None

            yvals = (
                self.data[curtype]
                .fluxes.to(
                    fluxunit,
                    equivalencies=u.spectral_density(self.data[curtype].waves),
                )
                .value
            )
            yuncs = (
                self.data[curtype]
                .uncs.to(
                    fluxunit,
                    equivalencies=u.spectral_density(self.data[curtype].waves),
                )
                .value
            )
            yplotvals = ymult * yvals
            if yoffset_type == "multiply":
                yplotvals *= 10**yoffset
            else:
                yplotvals += yoffset
            if curtype == "BAND":
                # do not plot the excluded band(s)
                for i, bandname in enumerate(self.data[curtype].get_band_names()):
                    if bandname in exclude:
                        yplotvals[i] = np.nan
                # plot band data as points with errorbars
                ax.errorbar(
                    x,
                    yplotvals,
                    yerr=ymult * yuncs,
                    fmt="o",
                    color=pcolor,
                    label=clabel,
                    mfc="white",
                )
            else:
                ax.plot(
                    x,
                    yplotvals,
                    "-",
                    color=pcolor,
                    label=clabel,
                )

            if curtype == annotate_key:
                # annotate the spectra
                waves = self.data[curtype].waves
                ann_indxs = np.where(
                    (waves >= annotate_wave_range[0])
                    & (waves <= annotate_wave_range[1])
                )
                ann_val = np.nanmedian(yplotvals[ann_indxs])
                if yoffset_type == "multiply":
                    ann_val *= 10**annotate_yoffset
                else:
                    ann_val += yoffset
                ann_val += (annotate_yoffset,)
                ann_xval = 0.5 * np.sum(annotate_wave_range.value)
                if wavenum:
                    ann_xval = 1 / ann_xval
                ax.text(
                    ann_xval,
                    ann_val,
                    annotate_text,
                    color=annotate_color,
                    horizontalalignment=annotate_alignment,
                    rotation=annotate_rotation,
                    fontsize=fontsize,
                )
