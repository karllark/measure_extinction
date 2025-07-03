import warnings
import numpy as np
import astropy.units as u
from astropy.table import QTable

fluxunit = u.erg / ((u.cm**2) * u.s * u.angstrom)


def read_spectra(sdata, full_filename):
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
    Updates sdata.(file, wave_range, waves, flux, uncs, npts, n_waves)
    """
    # full_filename = _getspecfilename(line, path)

    # open and read the spectrum
    # ignore units warnings as non-standard units are explicitly handled a few lines later
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", u.UnitsWarning)
        tdata = QTable.read(full_filename)

    sdata.waves = tdata["WAVELENGTH"]
    sdata.fluxes = tdata["FLUX"]
    sdata.uncs = tdata["SIGMA"]
    sdata.npts = tdata["NPTS"]
    sdata.n_waves = len(sdata.waves)

    # include the model if it exists
    #   currently only used for FUSE H2 model
    if "MODEL" in tdata.colnames:
        sdata.model = tdata["MODEL"]

    # fix odd unit designations
    if sdata.waves.unit == "ANGSTROM":
        sdata.waves = sdata.waves.value * u.angstrom
    if sdata.waves.unit == "MICRON":
        sdata.waves = sdata.waves.value * u.micron
    if sdata.fluxes.unit == "ERG/CM2/S/A":
        sdata.fluxes = sdata.fluxes.value * (u.erg / ((u.cm**2) * u.s * u.angstrom))
        sdata.uncs = sdata.uncs.value * (u.erg / ((u.cm**2) * u.s * u.angstrom))

    # compute the min/max wavelengths
    sdata.wave_range = (
        np.array([min(sdata.waves.value), max(sdata.waves.value)]) * sdata.waves.unit
    )

    # trim any data that is not finite
    (indxs,) = np.where(~np.isfinite(sdata.fluxes))
    if len(indxs) > 0:
        sdata.fluxes[indxs] = 0.0
        sdata.npts[indxs] = 0

    # convert wavelengths to microns (standardization)
    sdata.waves = sdata.waves.to(u.micron)
    sdata.wave_range = sdata.wave_range.to(u.micron)


def read_gen_spectra(sdata, filename, use_corfac=True, corfac=None):
    """
    Read in spectra that need no specific special processing

    Parameters
    ----------
    sdata : SpecData
        spectral data object

    filename : string
        full filename of spectral file

    use_corfac : boolean
        turn on/off using the correction factors

    corfac : dict of key: coefficients
        keys identify the spectrum to be corrected and how

    Returns
    -------
    Updates sdata.(file, wave_range, waves, flux, uncs, npts, n_waves)
    """
    read_spectra(sdata, filename)

    sdata.fluxes = sdata.fluxes.to(
        fluxunit, equivalencies=u.spectral_density(sdata.waves)
    )
    sdata.uncs = sdata.uncs.to(fluxunit, equivalencies=u.spectral_density(sdata.waves))


def read_spex(sdata, filename, use_corfac=True, corfac=None):
    """
    Read in SpeX spectra

    Parameters
    ----------
    sdata : SpecData
        spectral data object

    filename : string
        full filename of spectral file

    use_corfac : boolean
        turn on/off using the correction factors

    corfac : dict of key: coefficients
        keys identify the spectrum to be corrected and how

    Returns
    -------
    Updates sdata.(file, wave_range, waves, flux, uncs, npts, n_waves)
    """
    read_spectra(sdata, filename)

    # determine which correction factor to use
    if sdata.type == "SpeX_SXD":
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
        sdata.fluxes *= corfac
        sdata.uncs *= corfac

    sdata.fluxes = sdata.fluxes.to(
        fluxunit, equivalencies=u.spectral_density(sdata.waves)
    )
    sdata.uncs = sdata.uncs.to(fluxunit, equivalencies=u.spectral_density(sdata.waves))


def read_irs(sdata, filename, use_corfac=True, corfac=None):
    """
    Read in Spitzer/IRS spectra

    Correct the IRS spectra if the appropriate corfacs are present
    in the DAT file.
    Does a multiplicative correction that can include a linear
    term if corfac_irs_zerowave and corfac_irs_slope factors are present.
    Otherwise, just apply a multiplicative factor based on corfac_irs.

    Parameters
    ----------
    sdata : SpecData
        spectral data object

    filename : string
        full filename of spectral file

    use_corfac : boolean
        turn on/off using the correction factors

    corfac : dict of key: coefficients
        keys identify the spectrum to be corrected and how

    Returns
    -------
    Updates sdata.(file, wave_range, waves, flux, uncs, npts, n_waves)
    """
    read_spectra(sdata, filename)

    # standardization
    # mfac = Jy_to_cgs_const/np.square(sdata.waves)
    # sdata.fluxes *= mfac
    # sdata.uncs *= mfac

    # correct the IRS spectra if desired and if corfacs are defined
    if use_corfac and "IRS" in corfac.keys():
        if ("IRS_zerowave" in corfac.keys()) and ("IRS_slope" in corfac.keys()):
            mod_line = corfac["IRS"] + (
                corfac["IRS_slope"] * (sdata.waves.value - corfac["IRS_zerowave"])
            )
            sdata.fluxes *= mod_line
            sdata.uncs *= mod_line
        else:
            sdata.fluxes *= corfac["IRS"]
            sdata.uncs *= corfac["IRS"]

    # remove bad long wavelength IRS data if keyword set
    if "IRS_maxwave" in corfac.keys():
        (indxs,) = np.where(sdata.waves.value > corfac["IRS_maxwave"])
        if len(indxs) > 0:
            sdata.npts[indxs] = 0

    sdata.fluxes = sdata.fluxes.to(
        fluxunit, equivalencies=u.spectral_density(sdata.waves)
    )
    sdata.uncs = sdata.uncs.to(fluxunit, equivalencies=u.spectral_density(sdata.waves))
