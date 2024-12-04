import numpy as np
from astropy.table import Table, Column
import astropy.units as u

__all__ = [
    "merge_iue_obsspec",
    "merge_stis_obsspec",
    "merge_spex_obsspec",
    "merge_nircam_ss_obsspec",
    "merge_irs_obsspec",
    "merge_miri_lrs_obsspec",
    "merge_miri_ifu_obsspec",
]

fluxunit = u.erg / (u.s * u.cm * u.cm * u.angstrom)


def _wavegrid(resolution, wave_range):
    """
    Define a wavelength grid at a specified resolution given
    the min/max as input

    Parameters
    ----------
    resolution : float
        resolution of grid

    wave_range : [float, float]
        min/max of grid

    Returns
    -------
    wave_info : tuple [waves, waves_bin_min, waves_bin_max]
        wavelength grid center, min, max wavelengths
    """
    npts = int(
        np.log10(wave_range[1] / wave_range[0])
        / np.log10((1.0 + 2.0 * resolution) / (2.0 * resolution - 1.0))
    )
    delta_wave_log = (np.log10(wave_range[1]) - np.log10(wave_range[0])) / npts
    wave_log10 = np.arange(
        np.log10(wave_range[0]),
        np.log10(wave_range[1]) - delta_wave_log,
        delta_wave_log,
    )
    full_wave_min = 10**wave_log10
    full_wave_max = 10 ** (wave_log10 + delta_wave_log)

    full_wave = (full_wave_min + full_wave_max) / 2.0

    return (full_wave, full_wave_min, full_wave_max)


def merge_stis_obsspec(obstables, waveregion="UV", output_resolution=1000):
    """
    Merge one or more STIS 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed STIS spectra
        usually the result of reading tables

    waveregion : string [default = 'UV']
        wavelength region of spectra
        possibilities are 'UV', 'Opt'

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    if waveregion == "UV":
        wave_range = [1000.0, 3400.0] * u.angstrom
    elif waveregion == "Opt":
        wave_range = [2850.0, 10250.0] * u.angstrom

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)
    for ctable in obstables:
        # may want to add in the SYS-ERROR, but need to be careful
        # to propagate it correctly, SYS-ERROR will not reduce with
        # multiple spectra or measurements in a wavelength bin
        cuncs = ctable["STAT-ERROR"]  # / 1e-10  # deal with overflow errors
        cwaves = ctable["WAVELENGTH"].data
        cfluxes = ctable["FLUX"]
        cnpts = ctable["NPTS"].data
        cnpts[cuncs == 0] = 0
        for k in range(n_waves):
            gvals = (
                (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
            )
            if np.sum(gvals) > 0:
                weights = np.square(1.0 / cuncs[gvals].value)
                full_flux[k] += np.sum(weights * cfluxes[gvals].value)
                full_unc[k] += np.sum(weights)
                full_npts[k] += np.sum(gvals)

    # make sure any wavelengths with no unc are not used
    full_npts[full_unc == 0] = 0

    # divide by the net weights
    (indxs,) = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(
            1.0 / full_unc[indxs]
        )  # * 1e-10  # put back factor for overflow errors

    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_spex_obsspec(obstable, mask=[], output_resolution=2000):
    """
    Merge one or more IRTF SpeX 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstable : astropy Table object
        table containing the observed SpeX spectrum
        usually the result of reading tables

    mask : list of tuples [default=[]]
        list of tuples with wavelength regions (in micron) that need to be masked, e.g. [(2.55,2.61),(3.01,3.10)]

    output_resolution : float [default=2000]
        output resolution of spectrum
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    waves = obstable["WAVELENGTH"].data * 1e4
    fluxes = obstable["FLUX"].data
    uncs = obstable["ERROR"].data
    npts = np.full((len(obstable["FLUX"])), 1.0)

    # take out data points that were "flagged" as bad by SpeXtool (i.e. FLAG is not zero)
    npts[obstable["FLAG"] != 0] = 0
    # take out data points with NaN fluxes
    npts[np.isnan(fluxes)] = 0
    # quadratically add 1 percent uncertainty to account for unknown uncertainties
    uncs = np.sqrt(uncs**2 + (0.01 * fluxes) ** 2)
    # take out data points with low SNR
    npts[np.less(fluxes / uncs, 10, where=~np.isnan(fluxes / uncs))] = 0
    # take out wavelength regions affected by the atmosphere
    npts[np.logical_and(1.347e4 < waves, waves < 1.415e4)] = 0
    npts[np.logical_and(1.798e4 < waves, waves < 1.949e4)] = 0
    npts[np.logical_and(2.514e4 < waves, waves < 2.880e4)] = 0
    npts[np.logical_and(4.000e4 < waves, waves < 4.594e4)] = 0
    # take out data points that need to be masked
    for region in mask:
        npts[(waves > region[0] * 1e4) & (waves < region[1] * 1e4)] = 0

    # determine the wavelength range and calculate the wavelength grid
    if np.max(waves) < 25000:  # SXD
        wave_range = [0.8, 2.45] * u.micron
    else:  # LXD (this includes all 3 LXD modes: 1.9, 2.1 and 2.3, to make sure all LXD spectra have the same wavelength grid, independent of the original observing mode)
        wave_range = [1.9, 5.5] * u.micron

    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    # create empty arrays
    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)

    # fill the arrays
    for k in range(n_waves):
        (indxs,) = np.where(
            (waves >= full_wave_min[k]) & (waves < full_wave_max[k]) & (npts > 0)
        )
        if len(indxs) > 0:
            weights = 1.0 / np.square(uncs[indxs])
            full_flux[k] = np.sum(weights * fluxes[indxs])
            full_unc[k] = np.sum(weights)
            full_npts[k] = len(indxs)

    # divide by the net weights
    (indxs,) = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(
            1.0 / full_unc[indxs]
        )  # this is the standard error of the weighted mean

    # create the output table
    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["SIGMA"] = Column(full_unc, unit=u.erg / (u.s * u.cm * u.cm * u.angstrom))
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_gen_obsspec(obstables, wave_range, output_resolution=100):
    """
    Merge one or more generic spectra into a single spectrum
    on a uniform wavelength scale.  Useful for spectra that
    do not require specific processing.

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed spectra
        usually the result of reading tables

    wave_range : 2 element float
        min/max wavelengths with units for output grid

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    iwave_range = wave_range.to(u.angstrom).value
    full_wave, full_wave_min, full_wave_max = _wavegrid(output_resolution, iwave_range)

    n_waves = len(full_wave)
    full_flux = np.zeros((n_waves), dtype=float)
    full_unc = np.zeros((n_waves), dtype=float)
    full_npts = np.zeros((n_waves), dtype=int)
    for ctable in obstables:
        cwaves = ctable["WAVELENGTH"].to(u.angstrom).value
        cfluxes = (
            ctable["FLUX"]
            .to(fluxunit, equivalencies=u.spectral_density(ctable["WAVELENGTH"]))
            .value
        )
        cuncs = (
            ctable["ERROR"]
            .to(fluxunit, equivalencies=u.spectral_density(ctable["WAVELENGTH"]))
            .value
        )
        cnpts = ctable["NPTS"].value
        for k in range(n_waves):
            (indxs,) = np.where(
                (cwaves >= full_wave_min[k]) & (cwaves < full_wave_max[k]) & (cnpts > 0)
            )
            if len(indxs) > 0:
                weights = 1.0 / np.square(cuncs[indxs])
                full_flux[k] += np.sum(weights * cfluxes[indxs])
                full_unc[k] += np.sum(weights)
                full_npts[k] += len(indxs)

    # divide by the net weights
    (indxs,) = np.where(full_npts > 0)
    if len(indxs) > 0:
        full_flux[indxs] /= full_unc[indxs]
        full_unc[indxs] = np.sqrt(1.0 / full_unc[indxs])

    otable = Table()
    otable["WAVELENGTH"] = Column(full_wave, unit=u.angstrom)
    otable["FLUX"] = Column(full_flux, unit=fluxunit)
    otable["SIGMA"] = Column(full_unc, unit=fluxunit)
    otable["NPTS"] = Column(full_npts)

    return otable


def merge_iue_obsspec(obstables, output_resolution=1000):
    """
    Merge one or more IUE 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IUE spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [1000.0, 3400.0] * u.angstrom

    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution,
    )
    return otable


def merge_irs_obsspec(obstables, output_resolution=150):
    """
    Merge one or more Spitzer IRS 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [5.0, 40.0] * u.micron
    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution,
    )
    return otable


def merge_niriss_soss_obsspec(obstables, output_resolution=700):
    """
    Merge one or more NIRCam slitless 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [0.85, 2.75] * u.micron
    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution
    )
    return otable


def merge_nircam_ss_obsspec(obstables, output_resolution=1600):
    """
    Merge one or more NIRCam slitless 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [2.35, 5.55] * u.micron
    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution
    )
    return otable


def merge_miri_lrs_obsspec(obstables, output_resolution=100):
    """
    Merge one or more MIRI LRS spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [5.0, 13.0] * u.micron
    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution,
    )
    return otable


def merge_miri_ifu_obsspec(obstables, output_resolution=3000):
    """
    Merge one or more MIRI IFU 1D spectra into a single spectrum
    on a uniform wavelength scale

    Parameters
    ----------
    obstables : list of astropy Table objects
        list of tables containing the observed IRS spectra
        usually the result of reading tables

    output_resolution : float
        output resolution of spectra
        input spectrum assumed to be at the appropriate resolution

    Returns
    -------
    output_table : astropy Table object
        merged spectra
    """
    wave_range = [4.8, 29.0] * u.micron
    otable = merge_gen_obsspec(
        obstables, wave_range, output_resolution=output_resolution,
    )

    return otable
