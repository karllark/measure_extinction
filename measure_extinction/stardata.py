import math
import warnings
import os.path

from collections import OrderedDict

import numpy as np

# from astropy.io import fits
from astropy.table import Table
from astropy import constants as const
import astropy.units as u

__all__ = ["StarData", "BandData", "SpecData"]

# Jy to ergs/(cm^2 s A)
#   1 Jy = 1e-26 W/(m^2 Hz)
#   1 W = 1e7 ergs
#   1 m^2 = 1e4 cm^2
#   1 um = 1e4 A
#   Hz/um = c[micron/s]/(lambda[micron])^2
# const = 1e-26*1e7*1e-4*1e-4 = 1e-27
Jy_to_cgs_const = 1e-27 * const.c.to("micron/s").value


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

    def read_bands(self, lines):
        """
        Read the photometric band data from a DAT file
        and upate class variables.
        Bands are filled in wavelength order to make life
        easier in subsequent calcuations (interpolations!)

        Parameters
        ----------
        lines : list of string
            lines from a DAT formatted file

        Returns
        -------
        Updates self.bands, self.band_units, and self.n_bands
        """
        for line in lines:
            eqpos = line.find("=")
            pmpos = line.find("+/-")

            if (eqpos >= 0) & (pmpos >= 0) & (line[0] != "#"):
                # check for reference or unit
                colpos = max((line.find(";"), line.find("#"), line.find("mJy")))
                if colpos == -1:
                    colpos = len(line)
                # if there is both a reference and a unit
                elif line.find(";") != -1 and line.find("mJy") != -1:
                    colpos = min(line.find(";"), line.find("mJy"))
                band_name = line[0:eqpos].strip()
                self.bands[band_name] = (
                    float(line[eqpos + 1 : pmpos]),
                    float(line[pmpos + 3 : colpos]),
                )
                # units
                if line.find("mJy") >= 0:
                    self.band_units[band_name] = "mJy"
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
            The zeromag_flux is the flux in erg/cm2/s/A for a star of (Vega) mag=0. It gives the conversion factor from Vega magnitudes to erg/cm2/s/A.
        """
        _johnson_band_names = ["U", "B", "V", "R", "I", "J", "H", "K", "L", "M"]
        # JHK for 2MASS (Cohen et al. 2003 as given in the 2MASS explanatory suppliment)
        # note that K=Ks here (potential confusion, not major, but maybe should be dealt
        #    with at some point)
        _johnson_band_waves = np.array(
            [0.366, 0.438, 0.545, 0.641, 0.798, 1.235, 1.662, 2.159, 3.45, 4.75]
        )
        _johnson_band_zeromag_fluxes = (
            np.array(
                [417.5, 632.0, 363.1, 217.7, 112.6, 31.29, 11.33, 4.283, 0.699, 0.204]
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

        # zeromag Vega flux not given in standard WFPC2 documenation
        # instead the flux and Vega magnitudes are given for 1 DN/sec
        # the following code coverts these numbers to zeromag Vega fluxes
        for i in range(_n_wfpc2_bands):
            _wfpc2_band_zeromag_fluxes[i] = _wfpc2_photflam[i] * (
                10 ** (0.4 * _wfpc2_vegamag[i])
            )

        # WFC3 bands
        _wfc3_band_names = [
            "WFC3_F275W",
            "WFC3_F336W",
            "WFC3_F475W",
            "WFC3_F814W",
            "WFC3_F110W",
            "WFC3_F160W",
        ]
        _wfc3_band_waves = np.array([0.2373, 0.3355, 0.4772, 0.8053, 1.1534, 1.5369])
        _wfc3_photflam = np.array(
            [3.186e-18, 1.267e-18, 2.458e-19, 1.477e-19, 1.53e-20, 1.93e-20]
        )
        _wfc3_vegamag = np.array([22.331, 23.513, 25.809, 24.712, 26.063, 24.695])
        _n_wfc3_bands = len(_wfc3_vegamag)
        _wfc3_band_zeromag_fluxes = np.zeros(_n_wfc3_bands)

        # zeromag Vega flux not given in standard WFPC2 documenation
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
        self.fluxes = self.fluxes * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))
        self.uncs = self.uncs * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))

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

    return path + tfile


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

    def read_spectra(self, line, path="./"):
        """
        Read spectra from a FITS file

        FITS file has a binary table in the 1st extension
        Header needs to have:

        - wmin, wmax : min/max of wavelengths in file

        Expected columns are:

        - wave
        - flux
        - sigma [uncertainty in flux units]
        - npts [number of observations include at this wavelength]

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'IUE = hd029647_iue.fits'

        path : string, optional
            location of the FITS files path

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        full_filename = _getspecfilename(line, path)

        # open and read the spectrum
        tdata = Table.read(full_filename)

        self.waves = tdata["WAVELENGTH"].quantity
        self.fluxes = tdata["FLUX"].quantity
        self.uncs = tdata["SIGMA"].quantity
        self.npts = tdata["NPTS"].quantity
        self.n_waves = len(self.waves)

        # include the model if it exists
        #   currently only used for FUSE H2 model
        if "MODEL" in tdata.colnames:
            self.model = tdata["MODEL"].quantity

        # fix odd unit designations
        if self.waves.unit == "ANGSTROM":
            self.waves = self.waves.value * u.angstrom
        if self.waves.unit == "MICRON":
            self.waves = self.waves.value * u.micron
        if self.fluxes.unit == "ERG/CM2/S/A":
            self.fluxes = self.fluxes.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))
            self.uncs = self.uncs.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))

        # compute the min/max wavelengths
        self.wave_range = (
            np.array([min(self.waves.value), max(self.waves.value)]) * self.waves.unit
        )

        # trim any data that is not finite
        (indxs,) = np.where(~np.isfinite(self.fluxes))
        if len(indxs) > 0:
            self.fluxes[indxs] = 0.0
            self.npts[indxs] = 0

        # convert wavelengths to microns (standardization)
        self.waves = self.waves.to(u.micron)
        self.wave_range = self.wave_range.to(u.micron)

    def read_fuse(self, line, path="./"):
        """
        Read in FUSE spectra

        Converts the wavelengths from Anstroms to microns

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'STIS = hd029647_fuse.fits'

        path : string, optional
            location of the FITS files path

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        self.read_spectra(line, path)

    def read_iue(self, line, path="./"):
        """
        Read in IUE spectra

        Removes data with wavelengths > 3200 A
        Converts the wavelengths from Anstroms to microns

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'IUE = hd029647_iue.fits'

        path : string, optional
            location of the FITS files path

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        self.read_spectra(line, path)

        # trim the long wavelength data by setting the npts to zero
        (indxs,) = np.where(self.waves > 3200.0 * u.angstrom)
        if len(indxs) > 0:
            self.npts[indxs] = 0

    def read_stis(self, line, path="./"):
        """
        Read in STIS spectra

        Converts the wavelengths from Anstroms to microns

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'STIS = hd029647_stis.fits'

        path : string, optional
            location of the FITS files path

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        self.read_spectra(line, path)

        # add units
        self.fluxes = self.fluxes.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))
        self.uncs = self.uncs.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))

    def read_spex(self, line, path="./", use_corfac=True, corfac=None):
        """
        Read in SpeX spectra

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'SpeX = hd029647_spex.fits'

        path : string, optional
            location of the FITS files path

        corfac : dict of key: coefficients
            keys identify the spectrum to be corrected and how

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        self.read_spectra(line, path)

        # determine which correction factor to use
        if self.type == "SpeX_SXD":
            if "SpeX_SXD" in corfac.keys():
                corfac = corfac["SpeX_SXD"]
            else:
                corfac = None
        else:
            if "SpeX_LXD" in corfac.keys():
                corfac = corfac["SpeX_LXD"]
            else:
                corfac = None

        # correct the SpeX spectra if desired and if the correction factor is defined
        if use_corfac and corfac is not None:
            self.fluxes *= corfac
            self.uncs *= corfac

        # add units
        self.fluxes = self.fluxes.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))
        self.uncs = self.uncs.value * (u.erg / ((u.cm ** 2) * u.s * u.angstrom))

    def read_irs(self, line, path="./", use_corfac=True, corfac=None):
        """
        Read in Spitzer/IRS spectra

        Correct the IRS spectra if the appropriate corfacs are present
        in the DAT file.
        Does a multiplicative correction that can include a linear
        term if corfac_irs_zerowave and corfac_irs_slope factors are present.
        Otherwise, just apply a multiplicative factor based on corfac_irs.

        Parameters
        ----------
        line : string
            formatted line from DAT file
            example: 'IRS = hd029647_irs.fits'

        path : string, optional
            location of the FITS files path

        corfac : dict of key: coefficients
            keys identify the spectrum to be corrected and how

        Returns
        -------
        Updates self.(file, wave_range, waves, flux, uncs, npts, n_waves)
        """
        self.read_spectra(line, path)

        # standardization
        # mfac = Jy_to_cgs_const/np.square(self.waves)
        # self.fluxes *= mfac
        # self.uncs *= mfac

        # correct the IRS spectra if desired and if corfacs are defined
        if use_corfac and "IRS" in corfac.keys():
            if ("IRS_zerowave" in corfac.keys()) and ("IRS_slope" in corfac.keys()):
                mod_line = corfac["IRS"] + (
                    corfac["IRS_slope"] * (self.waves.value - corfac["IRS_zerowave"])
                )
                self.fluxes *= mod_line
                self.uncs *= mod_line
            else:
                self.fluxes *= corfac["IRS"]
                self.uncs *= corfac["IRS"]

        # remove bad long wavelength IRS data if keyword set
        if "IRS_maxwave" in corfac.keys():
            (indxs,) = np.where(self.waves.value > corfac["IRS_maxwave"])
            if len(indxs) > 0:
                self.npts[indxs] = 0

        if "IRS_maxsnr" in corfac.keys():
            # probably a more clever way to do this, but this at least works
            gindxs, = np.where(self.npts > 0)
            above_maxsnr = (self.fluxes[gindxs] / self.uncs[gindxs]) > corfac["IRS_maxsnr"]
            new_uncs = self.uncs[gindxs]
            new_uncs[above_maxsnr] = self.fluxes[gindxs][above_maxsnr] / corfac["IRS_maxsnr"]
            self.uncs[gindxs] = new_uncs

        # add units
        self.fluxes = self.fluxes.value * (u.Jy)
        self.uncs = self.uncs.value * (u.Jy)


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

    corfac : dict of key:correction factors
        key gives the type (e.g., IRS, IRS_slope)

    use_corfac : boolean
        whether or not to use the correction factors, default = True

    LXD_man : boolean
        whether or not the LXD scaling factor has been set manually, default = False
    """

    def __init__(self, datfile, path="", photonly=False, use_corfac=True):
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
        """
        self.file = datfile
        self.path = path
        self.sptype = ""
        self.model_params = {}
        self.data = {}
        self.corfac = {}
        self.photonly = photonly
        self.use_corfac = use_corfac
        self.LXD_man = False

        if self.file is not None:
            self.read()

    def read(self):
        """
        Populate the object from a DAT file + spectral files
        """

        # open and read all the lines in the file
        f = open(self.path + self.file, "r")
        self.datfile_lines = list(f)
        f.close()
        # get the photometric band data
        self.data["BAND"] = BandData("BAND")
        self.data["BAND"].read_bands(self.datfile_lines)

        # covert the photoemtric band data to fluxes in all possible bands
        self.data["BAND"].get_band_fluxes()

        # go through and get info before reading the spectra
        poss_mod_params = ["model_type", "Z", "vturb", "logg", "Teff", "origin"]
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
                elif cpair[0] == "corfac_irs_maxsnr":
                    self.corfac["IRS_maxsnr"] = float(cpair[1])
                elif cpair[0] == "corfac_irs":
                    self.corfac["IRS"] = float(cpair[1])

        # read the spectra
        if not self.photonly:
            for line in self.datfile_lines:
                if line.find("IUE") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["IUE"] = SpecData("IUE")
                        self.data["IUE"].read_iue(line, path=self.path)
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("FUSE") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["FUSE"] = SpecData("FUSE")
                        self.data["FUSE"].read_fuse(line, path=self.path)
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("STIS_Opt") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["STIS_Opt"] = SpecData("STIS_Opt")
                        self.data["STIS_Opt"].read_stis(line, path=self.path)
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("STIS") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["STIS"] = SpecData("STIS")
                        self.data["STIS"].read_stis(line, path=self.path)
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("SpeX_SXD") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["SpeX_SXD"] = SpecData("SpeX_SXD")
                        self.data["SpeX_SXD"].read_spex(
                            line,
                            path=self.path,
                            use_corfac=self.use_corfac,
                            corfac=self.corfac,
                        )
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("SpeX_LXD") == 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["SpeX_LXD"] = SpecData("SpeX_LXD")
                        self.data["SpeX_LXD"].read_spex(
                            line,
                            path=self.path,
                            use_corfac=self.use_corfac,
                            corfac=self.corfac,
                        )
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)
                elif line.find("IRS") == 0 and line.find("IRS15") < 0:
                    fname = _getspecfilename(line, self.path)
                    if os.path.isfile(fname):
                        self.data["IRS"] = SpecData("IRS")
                        self.data["IRS"].read_irs(
                            line,
                            path=self.path,
                            use_corfac=self.use_corfac,
                            corfac=self.corfac,
                        )
                    else:
                        warnings.warn(f"{fname} does not exist", UserWarning)

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
        fluxunit = u.erg / ((u.cm ** 2) * u.s * u.angstrom)
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
        exclude=[],
        yoffset=None,
        yoffset_type="multiply",
        annotate_key=None,
        annotate_wave_range=None,
        annotate_text=None,
        annotate_rotation=0.0,
        annotate_yoffset=0.0,
        annotate_color="k",
        legend_key=None,
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

        exclude : list of strings [default=[]]
            Which data type(s) to exclude from the plot (e.g., IRS)

        yoffset : float [default=None]
            multiplicative or additive offset for the data

        yoffset_type : str [default="multiply"]
            yoffset type: "multiply" or "add"

        annotate_key : string [default=None]
            type of data for which to annotate text (e.g., SpeX_LXD)

        annotate_wave_range : list of 2 floats [default=None]
            min/max wavelength range for the annotation of the text

        annotate_text : string [default=None]
            text to annotate

        annotate_rotation : float [default=0.0]
            annotation angle

        annotate_yoffset : float [default=0.0]
            y-offset for the annotated text

        annotate_color : string [default="k"]
            color of the annotated text

        legend_key : string [default=None]
            legend the spectrum using the given data key

        fontsize : int [default=None]
            fontsize for plot
        """
        fluxunit = u.erg / ((u.cm ** 2) * u.s * u.angstrom)

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
            # replace fluxes by NaNs for wavelength regions that need to be excluded from the plot, to avoid separate regions being connected artificially
            self.data[curtype].fluxes[self.data[curtype].npts == 0] = np.nan
            if mlam4:
                ymult = np.power(self.data[curtype].waves.value, 4.0)
            else:
                ymult = np.full((len(self.data[curtype].waves.value)), 1.0)
            # multiply by the overall normalization
            ymult /= normval

            if curtype == legend_key:
                red_name = self.file.replace(".dat", "")
                red_name = red_name.replace("DAT_files/", "")
                legval = "%s / %s" % (red_name, self.sptype)
            else:
                legval = None
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
                yplotvals *= yoffset
            else:
                yplotvals += yoffset
            if curtype == "BAND":
                # plot band data as points with errorbars
                ax.errorbar(
                    self.data[curtype].waves.value,
                    yplotvals,
                    yerr=ymult * yuncs,
                    fmt="o",
                    color=pcolor,
                    label=legval,
                    mfc="white",
                )
            else:
                ax.plot(
                    self.data[curtype].waves.value,
                    yplotvals,
                    "-",
                    color=pcolor,
                    label=legval,
                )

            if curtype == annotate_key:
                # annotate the spectra
                waves = self.data[curtype].waves
                ann_indxs = np.where(
                    (waves >= annotate_wave_range[0])
                    & (waves <= annotate_wave_range[1])
                )
                ann_val = np.nanmedian(yplotvals[ann_indxs])
                ann_val += (annotate_yoffset,)
                ann_xval = 0.5 * np.sum(annotate_wave_range.value)
                ax.text(
                    ann_xval,
                    ann_val,
                    annotate_text,
                    color=annotate_color,
                    horizontalalignment="left",
                    rotation=annotate_rotation,
                )
