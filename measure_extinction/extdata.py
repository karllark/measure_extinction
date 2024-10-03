import warnings
import numpy as np
import astropy.units as u

from astropy.io import fits
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling import Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.optimize import curve_fit
from scipy import stats

from dust_extinction.conversions import AxAvToExv

from measure_extinction.merge_obsspec import _wavegrid

__all__ = ["ExtData", "AverageExtData"]


# globals
# possible datasets (also extension names in saved FITS file)
_poss_datasources = [
    "BAND",
    "IUE",
    "FUSE",
    "STIS",
    "STIS_Opt",
    "SpeX_SXD",
    "SpeX_LXD",
    "IRS",
    "NIRCam_SS",
    "MIRI_IFU",
]


def _rebin(waves, exts, rebin_fac):
    """
    Code to rebin a 1d extinction curve

    Parameters
    ----------
    waves : np.ndarray
        the wavelengths of the extinction curve

    exts : np.ndarray
        the extinction values of the extinction curve

    rebin_fac : int
        the factor by which to rebin the extinction curve

    Returns
    -------
    The rebinned wavelengths and extinction values
    """
    # calculate the number of bins
    nbins = int(len(waves) / rebin_fac)

    # take out nans from the extinction values and take out the corresponding wavelengths (otherwise all the wavelengths in a bin would be used to calculate the mean wavelength, while only the non-nan extinctions would be used to calculate the mean extinction)
    mask = ~np.isnan(exts)
    waves = waves[mask]
    exts = exts[mask]

    # calculate the mean wavelength and mean extinction in every bin
    # caution: the new wavelength grid is not equally spaced, since the mean wavelength in every bin is calculated
    new_waves, new_exts = stats.binned_statistic(
        waves,
        (waves, exts),
        statistic="mean",
        bins=nbins,
    )[0]

    return new_waves, new_exts


def _flux_unc_as_mags(fluxes_in, uncs_in):
    """
    Provide the flux uncertainties in magnitudes accounting for the
    case where (fluxes-uncs) is negative
    """
    fluxes = np.atleast_1d(fluxes_in)
    uncs = np.atleast_1d(uncs_in)

    uncs_mag = np.empty(len(np.atleast_1d(fluxes)))

    # fluxes-uncs case
    (indxs,) = np.where(fluxes - uncs <= 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5 * np.log10(fluxes[indxs] / (fluxes[indxs] + uncs[indxs]))

    # normal case
    (indxs,) = np.where(fluxes - uncs > 0)
    if len(indxs) > 0:
        uncs_mag[indxs] = -2.5 * np.log10(
            (fluxes[indxs] - uncs[indxs]) / (fluxes[indxs] + uncs[indxs])
        )

    return uncs_mag


def _hierarch_keywords(names):
    """
    Prepend the 'HIERARCH ' string to all keywords > 8 characters
    Avoids FITS VerifyWarning.

    Parameters
    ----------
    names : list
        keywords

    Returns
    -------
    new_names : list
        keywords with HIERARCH prepended as apprpriate
    """
    new_names = []
    for cname in names:
        if len(cname) >= 8:
            new_names.append(f"HIERARCH {cname}")
        else:
            new_names.append(cname)

    return new_names


def _get_column_val(column):
    """

    Parameters
    ----------
    column : float or tuple
        gives the column or (column, unc) or (column, punc, munc)

    Returns
    -------
    column: float
        column value
    """
    if isinstance(column, tuple):
        return float(column[0])
    else:
        return float(column)


def _get_column_plus_unc(column):
    """

    Parameters
    ----------
    column : float or tuple
        gives the column or (column, unc) or (column, punc, munc)

    Returns
    -------
    column: tuple
        (column, unc)
    """
    if isinstance(column, tuple):
        if len(column) == 3:
            return (column[0], 0.5 * (column[1] + column[2]))
        else:
            return column
    else:
        return (column, 0.0)


def _get_rel_band(red, comp, rel_band):
    """
    Get the band to reference the extinction curve.
    Supports using a photometric band or a spectroscopic flux
    """
    if isinstance(rel_band, str):    # reference photometric band
        red_rel_band = red.data["BAND"].get_band_mag(rel_band)
        comp_rel_band = comp.data["BAND"].get_band_mag(rel_band)
    else:   # reference spectroscopic wavelength
        # find the source that has the requested wavelength
        red_rel_band = 0.0
        for ckey in red.data.keys():
            if ckey != "BAND":
                if np.min(red.data[ckey].waves) <= rel_band <= np.max(red.data[ckey].waves):
                    rflux = np.interp(rel_band, red.data[ckey].waves, red.data[ckey].fluxes).value
                    runc = np.interp(rel_band, red.data[ckey].waves, red.data[ckey].uncs).value
                    red_rel_band = (-2.5 * np.log10(rflux), _flux_unc_as_mags(rflux, runc)[0])

                    cflux = np.interp(rel_band, comp.data[ckey].waves, comp.data[ckey].fluxes).value
                    cunc = np.interp(rel_band, comp.data[ckey].waves, comp.data[ckey].uncs).value
                    comp_rel_band = (-2.5 * np.log10(cflux), _flux_unc_as_mags(cflux, cunc)[0])
        if red_rel_band == 0.0:
            raise ValueError("requested spectroscopic rel_band wavelength not present in any spectra")

    return (red_rel_band, comp_rel_band)


def AverageExtData(extdatas, min_number=3, mask=[]):
    """
    Generate the average extinction curve from a list of ExtData objects

    Parameters
    ----------
    extdatas : list of ExtData objects
        list of extinction curves to average

    min_number : int [default=3]
        minimum number of extinction curves that are required to measure the average extinction; if less than min_number of curves are available at certain wavelengths, the average extinction will still be calculated, but the number of points (npts) at those wavelengths will be set to zero (e.g. used in the plotting)

    mask : list of tuples [default=[]]
        list of tuples with wavelength regions (in micron) that need to be masked, e.g. [(2.55,2.61),(3.01,3.10)]

    Returns
    -------
    aveext: ExtData object
        the average extintion curve
    """
    aveext = ExtData()
    keys = []
    names = []
    bwaves = []
    for extdata in extdatas:
        # check the data type of the extinction curve, and convert if needed
        # the average curve must be calculated from the A(lambda)/A(V) curves
        if extdata.type != "alav" and extdata.type != "alax" and extdata.type != "elvebv":
            extdata.trans_elv_alav()

        # collect the keywords of the data in the extinction curves, and collect the names of the BAND data in the extinction curves, and determine the wavelengths of the data
        for src in extdata.waves.keys():
            if src not in keys:
                keys.append(src)
                aveext.waves[src] = extdata.waves[src]
            if src == "BAND":
                for i, name in enumerate(extdata.names["BAND"]):
                    if name not in names:
                        names.append(name)
                        bwaves.append(extdata.waves["BAND"][i].value)
    aveext.names["BAND"] = names
    aveext.waves["BAND"] = bwaves * u.micron
    aveext.type = extdatas[0].type
    aveext.type_rel_band = extdatas[0].type_rel_band

    # collect all the extinction data
    bexts = {k: [] for k in aveext.names["BAND"]}
    for src in keys:
        exts = []
        for extdata in extdatas:
            if src in extdata.waves.keys():
                if src == "BAND":
                    for i, name in enumerate(extdata.names["BAND"]):
                        bexts[name].append(extdata.exts["BAND"][i])
                else:
                    extdata.exts[src][np.where(extdata.npts[src] == 0)] = np.nan
                    exts.append(extdata.exts[src])

        # calculate the average and uncertainties of the band extinction data
        if src == "BAND":
            aveext.exts["BAND"] = np.zeros(len(names))
            aveext.npts["BAND"] = np.zeros(len(names))
            aveext.stds["BAND"] = np.zeros(len(names))
            aveext.uncs["BAND"] = np.zeros(len(names))
            for i, name in enumerate(aveext.names["BAND"]):
                aveext.exts["BAND"][i] = np.nanmean(bexts[name])
                aveext.npts["BAND"][i] = len(bexts[name])

                # calculation of the standard deviation (this is the spread of the sample around the population mean)
                aveext.stds["BAND"][i] = np.nanstd(bexts[name], ddof=1)

            # calculation of the standard error of the average (the standard error of the sample mean is an estimate of how far the sample mean is likely to be from the population mean)
            aveext.uncs["BAND"] = aveext.stds["BAND"] / np.sqrt(aveext.npts["BAND"])

        # calculate the average and uncertainties of the spectral extinction data
        else:
            exts = np.stack(exts, axis=0)
            aveext.exts[src] = np.nanmean(exts, axis=0)
            aveext.npts[src] = np.sum(~np.isnan(exts), axis=0)
            aveext.stds[src] = np.nanstd(exts, axis=0, ddof=1)
            aveext.uncs[src] = aveext.stds[src] / np.sqrt(aveext.npts[src])

        # take out the data points where less than a certain number of values was averaged, and give a warning
        if min_number > 1:
            aveext.npts[src][aveext.npts[src] < min_number] = 0
            warnings.warn(
                "The minimum number of "
                + str(min_number)
                + " extinction curves was not reached for certain wavelengths, and the number of points (npts) for those wavelengths was set to 0.",
                UserWarning,
            )
        # take out data points in masked region(s)
        for region in mask:
            aveext.npts[src][
                (aveext.waves[src].value >= region[0])
                & (aveext.waves[src].value <= region[1])
            ] = 0

    return aveext


class ExtData:
    """
    Extinction for a single line-of-sight

    Attributes:

    type : string
        extinction curve type (e.g., elx or alax)

    type_rel_band : string
        band name for relative extinction measurement (x in elx)

    red_file : string
        reddened star filename

    comp_file : string
        comparison star filename

    columns : dict of tuples of column measurements
        measurements are A(V), R(V), N(HI), etc.
        tuples are measurement, uncertainty

    waves : dict of key:wavelengths
        key is BAND, IUE, IRS, etc.

    ext : dict of key:E(lambda-X) or A(lambda)/A(X) measurements

    uncs : dict of key:E(lambda-X) or A(lambda)/A(X) measurement uncertainties

    stds : dict of key:A(lambda)/A(X) standard deviations (only defined if the curve is an average of a set of curves, in which case the standard deviation is the spread of the sample around the population mean)

    npts : dict of key:number of measurements at each wavelength

    names : dict of key:names of names of each wavelength (if photometric bands)

    fm90 : list of FM90 parameters tuples
        tuples are measurement, uncertainty

    model : dict of key:value with model fitting results, including
        - waves: np.ndarray with the wavelengths used in the fitting
        - exts: np.ndarray with the fitted powerlaw model to the extinction curve
        - residuals: np.ndarray with the fractional residuals, i.e. (data-fit)/fit
        - params: tuple with the parameters (amplitude, alpha) if data in A(lambda)/A(V) or (amplitude, alpha, A(V)) if data in E(lambda-V)
    """

    def __init__(self, filename=None):
        """
        Parameters
        ----------
        filename : string, optional [default=None]
            Full filename to a saved extinction curve
        """
        self.type = ""
        self.type_rel_band = ""
        self.red_file = ""
        self.comp_file = ""
        self.columns = {}
        self.fm90 = {}
        self.waves = {}
        self.exts = {}
        self.uncs = {}
        self.stds = {}
        self.npts = {}
        self.names = {}
        self.model = {}

        if filename is not None:
            self.read(filename)

    def calc_elx_bands(self, red, comp, rel_band="V"):
        """
        Calculate the E(lambda-X) for the photometric band data

        Separate from the spectral case as the bands in common must
        be found. In addition, some of the photometric observations are
        reported as colors (e.g., B-V) with uncertainties on those colors.
        As colors are what is needed for the extinction curve, we want to
        work in those colors to preserve the inheritly lower uncertainties.

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        comp : :class:StarData
            Observed data for the comparison star

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts, names)['BAND']
        """
        red_rel_band, comp_rel_band = _get_rel_band(red, comp, rel_band)

        # possible bands for the band extinction curve
        poss_bands = red.data["BAND"].get_poss_bands()

        waves = []
        exts = []
        uncs = []
        npts = []
        names = []
        for pband_name in poss_bands.keys():
            red_mag = red.data["BAND"].get_band_mag(pband_name)
            comp_mag = comp.data["BAND"].get_band_mag(pband_name)
            if (red_mag is not None) & (comp_mag is not None):
                ext = (red_mag[0] - red_rel_band[0]) - (comp_mag[0] - comp_rel_band[0])
                unc = np.sqrt(
                    red_mag[1] ** 2
                    + red_rel_band[1] ** 2
                    + comp_mag[1] ** 2
                    + comp_rel_band[1] ** 2
                )
                waves.append(red.data["BAND"].band_waves[pband_name])
                exts.append(ext)
                uncs.append(unc)
                npts.append(1)
                names.append(pband_name)

        if len(waves) > 0:
            self.waves["BAND"] = np.array(waves) * u.micron
            self.exts["BAND"] = np.array(exts)
            self.uncs["BAND"] = np.array(uncs)
            self.npts["BAND"] = np.array(npts)
            self.names["BAND"] = np.array(names)

    def calc_elx_spectra(self, red, comp, src, rel_band="V"):
        """
        Calculate the E(lambda-X) for the spectroscopic data

        Parameters
        ----------
        red : :class:StarData
            Observed data for the reddened star

        star : :class:StarData
            Observed data for the comparison star

        src : string
            data source (see global _poss_datasources)

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts)[src]
        """
        if (src in red.data.keys()) & (src in comp.data.keys()):
            # check that the wavelength grids are identical
            delt_wave = red.data[src].waves - comp.data[src].waves
            if np.sum(np.absolute(delt_wave)) > 0.01 * u.micron:
                warnings.warn("wavelength grids not equal for %s" % src, UserWarning)
            else:
                red_rel_band, comp_rel_band = _get_rel_band(red, comp, rel_band)

                # setup the needed variables
                self.waves[src] = red.data[src].waves
                n_waves = len(self.waves[src])
                self.exts[src] = np.zeros(n_waves)
                self.uncs[src] = np.zeros(n_waves)
                self.npts[src] = np.zeros(n_waves)

                # only compute the extinction for good, positive fluxes
                (indxs,) = np.where(
                    (red.data[src].npts > 0)
                    & (comp.data[src].npts > 0)
                    & (red.data[src].fluxes.value > 0)
                    & (comp.data[src].fluxes.value > 0)
                )
                self.exts[src][indxs] = -2.5 * np.log10(
                    red.data[src].fluxes[indxs] / comp.data[src].fluxes[indxs]
                ) + (comp_rel_band[0] - red_rel_band[0])
                self.uncs[src][indxs] = np.sqrt(
                    np.square(
                        _flux_unc_as_mags(
                            red.data[src].fluxes[indxs], red.data[src].uncs[indxs]
                        )
                    )
                    + np.square(
                        _flux_unc_as_mags(
                            comp.data[src].fluxes[indxs], comp.data[src].uncs[indxs]
                        )
                    )
                    + np.square(red_rel_band[1])
                    + np.square(comp_rel_band[1])
                )
                self.npts[src][indxs] = np.full(len(indxs), 1)

    def calc_elx(self, redstar, compstar, rel_band="V"):
        """
        Calculate the E(lambda-X) basic extinction measurement

        Parameters
        ----------
        redstar : :class:StarData
            Observed data for the reddened star

        compstar : :class:StarData
            Observed data for the comparison star

        rel_band : string
            Band to use for relative extinction measurement
            default = "V"

        Returns
        -------
        updates self.(waves, exts, uncs, npts, names)
        """
        self.type = "elx"
        self.type_rel_band = rel_band
        self.red_file = redstar.file
        self.comp_file = compstar.file
        for cursrc in _poss_datasources:
            if cursrc == "BAND":
                self.calc_elx_bands(redstar, compstar, rel_band=rel_band)
            else:
                self.calc_elx_spectra(redstar, compstar, cursrc, rel_band=rel_band)

    def calc_EBV(self):
        """
        Calculate E(B-V) from the observed extinction curve

        Returns
        -------
        Updates self.columns["EBV"]
        """
        # determine the index for the B band
        dwaves = np.absolute(self.waves["BAND"] - 0.438 * u.micron)
        sindxs = np.argsort(dwaves)
        bindx = sindxs[0]
        if dwaves[bindx] > 0.02 * u.micron:
            warnings.warn("no B band measurement in E(l-V)", UserWarning)
        else:
            self.columns["EBV"] = (self.exts["BAND"][bindx], self.uncs["BAND"][bindx])

    def calc_AV(self, akav=0.105):
        """
        Calculate A(V) from the observed extinction curve:
            - fit a powerlaw to the SpeX extinction curve, if available
            - otherwise: extrapolate the K-band extinction

        Parameters
        ----------
        akav : float
           Value of A(K)/A(V)
           default is from Decleir et al. (2022)
           for Rieke & Lebofsky (1985) use akav=0.112
           van de Hulst No. 15 curve has A(K)/A(V) = 0.0885

        Returns
        -------
        Updates self.columns["AV"]
        """
        # if SpeX extinction curve is available: compute A(V) by fitting the NIR extintion curve with a powerlaw.
        if "SpeX_SXD" in self.waves.keys() or "SpeX_LXD" in self.waves.keys():
            self.fit_spex_ext()
            if not isinstance(self.columns["AV"], tuple):
                self.columns["AV"] = (self.columns["AV"], 0.0)

        # if no SpeX spectrum is available: compute A(V) from E(K-V)
        else:
            dwaves = np.absolute(self.waves["BAND"] - 2.19 * u.micron)
            kindx = dwaves.argmin()
            if dwaves[kindx] > 0.04 * u.micron:
                warnings.warn(
                    "No K band measurement available in E(lambda-V)!", stacklevel=2
                )
            else:
                ekv = self.exts["BAND"][kindx]
                av = ekv / (akav - 1)
                avunc = np.absolute(av * (self.uncs["BAND"][kindx] / ekv))
                self.columns["AV"] = (av, avunc)

    def calc_AV_JHK(
        self,
        ref_waves=np.array([1.25, 1.6, 2.2]) * u.micron,
        ref_alav=[0.269, 0.163, 0.105],
        ref_wave=None,
    ):
        """
        Calculate A(V) from the observed extinction curve:
            - extrapolate from J, H, & K photometry

        Parameters
        ----------
        ref_waves : floats
            wavelengths for reference values (default = JHK)
        ref_alav : floats
            A(lambda)/A(V) values for reference
            default is for JHK from Decleir et al. (2022)
            for Rieke, Rieke, & Paul (1989) use ref_alav=[0.2815534, 0.17475728, 0.11197411],

        Returns
        -------
        Updates self.columns["AV"]
        """
        # J, H, K

        avs = []
        avs_unc = []
        for cwave, calav in zip(ref_waves, ref_alav):
            dwaves = np.absolute(self.waves["BAND"] - cwave)
            kindx = dwaves.argmin()
            if dwaves[kindx] < 0.1 * u.micron:
                cav = self.exts["BAND"][kindx] / (calav - 1)
                cavunc = np.absolute(
                    cav * (self.uncs["BAND"][kindx] / self.exts["BAND"][kindx])
                )
                avs.append(cav)
                avs_unc.append(cavunc)

        if len(avs) > 0:
            weights = 1.0 / np.square(avs_unc)
            av = np.average(avs, weights=weights)
            avunc = np.sqrt(1.0 / np.sum(weights))
            self.columns["AV"] = (av, avunc)
        else:
            warnings.warn(
                "No JHK band measurement available in E(lambda-V) so no A(V) measurement",
                stacklevel=2,
            )

    def calc_RV(self):
        """
        Calculate R(V) from the observed extinction curve

        Returns
        -------
        Updates self.columns["RV"]
        """
        # obtain or calculate A(V)
        if "AV" not in self.columns.keys():
            self.calc_AV()
        av = _get_column_plus_unc(self.columns["AV"])

        # obtain or calculate E(B-V)
        if "EBV" not in self.columns.keys():
            self.calc_EBV()
        ebv = _get_column_plus_unc(self.columns["EBV"])

        rv = av[0] / ebv[0]
        rvunc = rv * np.sqrt((av[1] / av[0]) ** 2 + (ebv[1] / ebv[0]) ** 2)
        self.columns["RV"] = (rv, rvunc)

    def trans_elv_elvebv(self, ebv=None):
        """
        Transform E(lambda-V) to E(lambda -V)/E(B-V) by
        normalizing by E(B-V)).

        Parameters
        ----------
        ebv : float [default = None]
            value of E(B-V) to use - otherwise take it from the columns of the object
            or calculate it from the E(lambda-V) curve

        Returns
        -------
        Updates self.(exts, uncs)
        """
        if self.type_rel_band != "V":
            warnings.warn(
                "attempt to normalize a non E(lambda-V) curve with E(B-V)", UserWarning
            )
        elif self.type != "elx":
            warnings.warn(
                "attempt to normalize a non E(lambda-V) curve with E(B-V)", UserWarning
            )
        else:
            if ebv is None:
                if "EBV" not in self.columns.keys():
                    self.calc_EBV()
                fullebv = _get_column_plus_unc(self.columns["EBV"])
            else:
                fullebv = _get_column_plus_unc(ebv)

            for curname in self.exts.keys():
                # only compute where there is data and exts is not zero
                gvals = (self.exts[curname] != 0) & (self.npts[curname] > 0)
                self.uncs[curname][gvals] = (
                    np.absolute(self.exts[curname][gvals] / fullebv[0])
                ) * np.sqrt(
                    np.square(self.uncs[curname][gvals] / self.exts[curname][gvals])
                    + np.square(fullebv[1] / fullebv[0])
                )
                self.exts[curname] /= fullebv[0]

            self.type = "elvebv"

    def trans_elv_alav(self, av=None, akav=0.112):
        """
        Transform E(lambda-V) to A(lambda)/A(V) by normalizing to A(V) and adding 1. If A(V) is in the columns of the extdata object, use that value. If A(V) is passed explicitly, use that value instead. If no A(V) is available, calculate A(V) from the input elx curve.

        Parameters
        ----------
        av : float [default = None]
            value of A(V) to use - otherwise take it from the columns of the object or calculate it

        akav : float  [default = 0.112]
           Value of A(K)/A(V), only needed if A(V) has to be calculated from the K-band extinction
           default is from Rieke & Lebofsky (1985)
           van de Hulst No. 15 curve has A(K)/A(V) = 0.0885

        Returns
        -------
        Updates self.(exts, uncs)
        """
        if (self.type_rel_band != "V") and (self.type_rel_band != 0.55 * u.micron):
            warnings.warn(
                "attempt to normalize a non-E(lambda-V) curve with A(V)", UserWarning
            )
        elif self.type != "elx":
            warnings.warn(
                "attempt to normalize a non E(lambda-V) curve with A(V)", UserWarning
            )
        else:
            if av is None:
                if "AV" not in self.columns.keys():
                    self.calc_AV(akav=akav)
                fullav = _get_column_plus_unc(self.columns["AV"])
            else:
                fullav = _get_column_plus_unc(av)

            for curname in self.exts.keys():
                # special case for the E(lambda - V) = 0 see below
                zvals = (self.exts[curname] == 0) & (self.npts[curname] > 0)
                # formal error propagation where zero extinctions do not
                # require separate treatment to avoid divide by zero errors
                self.uncs[curname] = (
                    np.sqrt(
                        np.square(self.uncs[curname])
                        + np.square(self.exts[curname] * fullav[1] / fullav[0])
                    )
                    / fullav[0]
                )

                self.exts[curname] = (self.exts[curname] / fullav[0]) + 1
                # replace the V band uncertainty with the fractional A(V) uncertainty
                # as this is the only term nominally in the A(lam)/A(V) extinction
                # that is by definition 1.  Fractional as the extinction at this
                # wavelength is normalized to A(V).
                #  zvals is defined to only be True for V band
                if np.sum(zvals) > 0:
                    self.uncs[curname][zvals] = fullav[1] / fullav[0]

                # make sure measurements with npts = 0 have zero exts and uncs
                zvals = self.npts[curname] <= 0
                if np.sum(zvals) > 0:
                    self.exts[curname][zvals] = 0.0
                    self.uncs[curname][zvals] = 0.0

            self.type = "alax"

    def rebin_constres(self, source, waverange, resolution):
        """
        Rebin the source extinction curve to a fixed spectral resolution
        and min/max wavelength range.

        Parameters
        ----------
        source : str
            source of extinction (i.e. "IUE", "IRS")
        waverange : 2 element array of astropy Quantities
            Min/max of wavelength range with units
        resolution : float
            Spectral resolution of rebinned extinction curve

        Returns
        -------
        measure_extinction ExtData
            Object with source extinction curve rebinned

        """
        if source == "BAND":
            raise ValueError("BAND extinction cannot be rebinned")

        if source not in self.exts.keys():
            warnings.warn(f"{source} extinction not present")
        else:
            # setup new wavelength grid
            full_wave, full_wave_min, full_wave_max = _wavegrid(
                resolution, waverange.to(u.micron).value
            )
            n_waves = len(full_wave)

            # setup the new rebinned vectors
            new_waves = full_wave * u.micron
            new_exts = np.zeros((n_waves), dtype=float)
            new_uncs = np.zeros((n_waves), dtype=float)
            new_npts = np.zeros((n_waves), dtype=int)

            # check if uncertainties defined and set temporarily to 1 if not
            # needed to avoid infinite weights
            nouncs = False
            if np.sum(self.uncs[source] > 0.0) == 0:
                nouncs = True
                self.uncs[source] = np.full((len(self.waves[source])), 1.0)

            # rebin using a weighted average
            owaves = self.waves[source].to(u.micron).value
            for k in range(n_waves):
                (indxs,) = np.where(
                    (owaves >= full_wave_min[k])
                    & (owaves < full_wave_max[k])
                    & (self.npts[source] > 0.0)
                )
                if len(indxs) > 0:
                    weights = 1.0 / np.square(self.uncs[source][indxs])
                    sweights = np.sum(weights)
                    new_exts[k] = np.sum(weights * self.exts[source][indxs]) / sweights
                    new_uncs[k] = 1.0 / np.sqrt(sweights)
                    new_npts[k] = np.sum(self.npts[source][indxs])

            if nouncs:
                new_uncs = np.full((n_waves), 0.0)

            # update source values
            self.waves[source] = new_waves
            self.exts[source] = new_exts
            self.uncs[source] = new_uncs
            self.npts[source] = new_npts

    def get_fitdata(
        self,
        req_datasources,
        remove_uvwind_region=False,
        remove_lya_region=False,
        remove_irsblue=False,
    ):
        """
        Get the data to use in fitting the extinction curve

        Parameters
        ----------
        req_datasources : list of str
            list of data sources (e.g., ['IUE', 'BAND'])

        remove_uvwind_region : boolean, optional (default=False)
            remove the UV wind regions from the returned data

        remove_lya_region : boolean, optional (default=False)
            remove the Ly-alpha regions from the returned data

        remove_irsblue : boolean, optional (default=False)
            remove the IRS blue photometry from the returned data

        Returns
        -------
        (wave, y, y_unc) : tuple of arrays
            wave is wavelength in microns
            y is extinction (no units)
            y_unc is uncertainty on y (no units)
        """
        xdata = []
        ydata = []
        uncdata = []
        nptsdata = []
        for cursrc in req_datasources:
            if cursrc in self.waves.keys():
                if (cursrc == "BAND") & remove_irsblue:
                    ibloc = np.logical_and(
                        14.0 * u.micron <= self.waves[cursrc],
                        self.waves[cursrc] < 16.0 * u.micron,
                    )
                    self.npts[cursrc][ibloc] = 0
                xdata.append(self.waves[cursrc].to(u.micron).value)
                ydata.append(self.exts[cursrc])
                uncdata.append(self.uncs[cursrc])
                nptsdata.append(self.npts[cursrc])
        wave = np.concatenate(xdata) * u.micron
        y = np.concatenate(ydata)
        unc = np.concatenate(uncdata)
        npts = np.concatenate(nptsdata)

        # remove uv wind line regions
        x = wave.to(1.0 / u.micron, equivalencies=u.spectral())
        if remove_uvwind_region:
            npts[np.logical_and(6.4 / u.micron <= x, x < 6.6 / u.micron)] = 0
            npts[np.logical_and(7.1 / u.micron <= x, x < 7.3 / u.micron)] = 0

        # remove Lya line region
        if remove_lya_region:
            npts[np.logical_and(8.0 / u.micron <= x, x < 8.475 / u.micron)] = 0

        # sort the data
        # at the same time, remove points with no data
        (gindxs,) = np.where(npts > 0)
        sindxs = np.argsort(x[gindxs])
        gindxs = gindxs[sindxs]
        wave = wave[gindxs]
        y = y[gindxs]
        unc = unc[gindxs]
        return (wave, y, unc)

    def save(
        self,
        ext_filename,
        column_info=None,
        save_params=None,
        fm90_best_params=None,
        fm90_per_params=None,
        p92_best_params=None,
        p92_per_params=None,
    ):
        """
        Save the extinction curve to a FITS file

        Parameters
        ----------
        filename : string
            Full filename to save extinction curve

        column_info : dict
            dictionary with information about the dust column
            for example: {'ebv': 0.1, 'rv': 4.2, 'av': 0.42}

        save_params : dict
            "type" - type of parameters (e.g., FM90, P92)
            "best" - best fit parameters as tuple (names, values)
            "per" - percentile parameters as tuple (names, p50s, puncs, muncs)

        fm90_best_params : tuple of 2 float vectors
           parameter names and best fit values for the FM90 fit
           (legacy, use save_params instead)

        fm90_per_params : tuple of 2 float vectors
           parameter names and (p50, +unc, -unc) values for the FM90 fit
           (legacy, use save_params instead)

        p92_best_params : tuple of 2 float vectors
           parameter names and best fit values for the P92 fit
           (legacy, use save_params instead)

        p92_per_params : tuple of 2 float vectors
           parameter names and (p50, +unc, -unc) values for the P92 fit
           (legacy, use save_params instead)
        """
        # generate the primary header
        pheader = fits.Header()
        hname = ["EXTTYPE", "EXTBAND", "R_FILE", "C_FILE"]
        hcomment = [
            "Type of ext curve (options: elx, elxebv, alax)",
            "Band name for relative extinction measurement",
            "Reddened Star File",
            "Comparison Star File",
        ]
        if isinstance(self.type_rel_band, str):
            trelband = self.type_rel_band
        else:
            trelband = f"{self.type_rel_band}"
        hval = [self.type, trelband, self.red_file, self.comp_file]

        ext_col_info = {
            "ebv": ("EBV", "E(B-V)"),
            "ebvunc": ("EBV_UNC", "E(B-V) uncertainty"),
            "av": ("AV", "A(V)"),
            "avunc": ("AV_UNC", "A(V) uncertainty"),
            "rv": ("RV", "R(V)"),
            "rvunc": ("RV_UNC", "R(V) uncertainty"),
        }
        # give preference to the column info that is given as argument to the save function
        if column_info is not None:
            for ckey in column_info.keys():
                if ckey in ext_col_info.keys():
                    keyname, desc = ext_col_info[ckey]
                    hname.append(keyname)
                    hcomment.append(desc)
                    hval.append(column_info[ckey])
                else:
                    print(ckey + " not supported for saving extcurves")
        else:  # save the column info if available in the extdata object
            colkeys = ["AV", "RV", "IRV", "EBV", "LOGHI"]
            colinfo = [
                "V-band extinction A(V)",
                "total-to-selective extintion R(V)",
                "selective-to-total 1/R(V)",
                "color excess E(B-V)",
                "log10 of the HI column density N(HI)",
            ]
            for i, ckey in enumerate(colkeys):
                if ckey in self.columns.keys():
                    hname.append(f"{ckey}")
                    hcomment.append(f"{colinfo[i]}")
                    if isinstance(self.columns[f"{ckey}"], tuple):
                        hval.append(self.columns[f"{ckey}"][0])
                        if len(self.columns[f"{ckey}"]) == 2:
                            hname.append(f"{ckey}_UNC")
                            hcomment.append(f"{ckey} uncertainty")
                            hval.append(self.columns[f"{ckey}"][1])
                        elif len(self.columns[f"{ckey}"]) == 3:
                            hname.append(f"{ckey}_PUNC")
                            hcomment.append(f"{ckey} upper uncertainty")
                            hval.append(self.columns[f"{ckey}"][1])
                            hname.append(f"{ckey}_MUNC")
                            hcomment.append(f"{ckey} lower uncertainty")
                            hval.append(self.columns[f"{ckey}"][2])
                    else:
                        hval.append(self.columns[f"{ckey}"])

        # legacy save param keywords
        if fm90_best_params is not None:
            save_params = {"type": "FM90", "best": fm90_best_params}
            if fm90_per_params is not None:
                save_params["per"] = fm90_per_params
        if p92_best_params is not None:
            save_params = {"type": "P92", "best": p92_best_params}
            if p92_per_params is not None:
                save_params["per"] = p92_per_params

        # save parameters
        if save_params is not None:
            if "type" in save_params.keys():
                tstr = save_params["type"]
            else:
                raise ValueError("type not in save_params dict")

            if "best" in save_params.keys():
                best_params = save_params["best"]
                best_keys = _hierarch_keywords(best_params[0])
                hname = np.concatenate((hname, best_keys))
                hval = np.concatenate((hval, best_params[1]))
                tcomment = [f"{tstr} parameter" for pname in best_params[0]]
                hcomment = np.concatenate((hcomment, tcomment))

            if "per" in save_params.keys():
                params = save_params["per"]
                # p50 values
                p50_keys = _hierarch_keywords([f"{cp}_p50" for cp in params[0]])
                hname = np.concatenate((hname, p50_keys))
                hval = np.concatenate((hval, [cv[0] for cv in params[1]]))
                tcomment = [f"{tstr} p50 parameter" for pname in params[0]]
                hcomment = np.concatenate((hcomment, tcomment))

                # +unc values
                punc_keys = _hierarch_keywords([f"{cp}_punc" for cp in params[0]])
                hname = np.concatenate((hname, punc_keys))
                hval = np.concatenate((hval, [cv[1] for cv in params[1]]))
                tcomment = [f"{tstr} punc parameter" for pname in params[0]]
                hcomment = np.concatenate((hcomment, tcomment))

                # -unc values
                munc_keys = _hierarch_keywords([f"{cp}_munc" for cp in params[0]])
                hname = np.concatenate((hname, munc_keys))
                hval = np.concatenate((hval, [cv[2] for cv in params[1]]))
                tcomment = [f"{tstr} munc parameter" for pname in params[0]]
                hcomment = np.concatenate((hcomment, tcomment))

        # other possible header keywords
        #   setup to populate if info passed (TBD)
        #         'LOGT','LOGT_UNC','LOGG','LOGG_UNC','LOGZ','LOGZ_UNC',
        #         'AV','AV_unc','RV','RV_unc',
        #         'FMC2','FMC2U','FMC3','FMC3U','FMC4','FMC4U',
        #         'FMx0','FMx0U','FMgam','FMgamU',
        #         'LOGHI','LOGHI_U','LOGHIMW','LHIMW_U',
        #         'NHIAV','NHIAV_U','NHIEBV','NHIEBV_U'
        for k in range(len(hname)):
            pheader.set(hname[k], hval[k], hcomment[k])

        pheader.add_comment("Created with measure_extinction package")
        pheader.add_comment("https://github.com/karllark/measure_extinction")
        phdu = fits.PrimaryHDU(header=pheader)

        hdulist = fits.HDUList([phdu])

        # write the portions of the extinction curve from each dataset
        # individual extensions so that the full info is perserved
        for curname in self.exts.keys():
            col1 = fits.Column(
                name="WAVELENGTH", format="E", array=self.waves[curname].to(u.micron)
            )
            col2 = fits.Column(name="EXT", format="E", array=self.exts[curname])
            col3 = fits.Column(name="UNC", format="E", array=self.uncs[curname])
            col4 = fits.Column(name="NPTS", format="E", array=self.npts[curname])
            cols = [col1, col2, col3, col4]
            if curname in self.stds.keys():
                cols.append(
                    fits.Column(name="STDS", format="E", array=self.stds[curname])
                )
            if curname == "BAND":
                cols.append(
                    fits.Column(name="NAME", format="A20", array=self.names[curname])
                )
            columns = fits.ColDefs(cols)
            tbhdu = fits.BinTableHDU.from_columns(columns)
            tbhdu.header.set(
                "EXTNAME", "%sEXT" % curname, "%s based extinction" % curname
            )
            hdulist.append(tbhdu)

        # write the fitted model if available
        if self.model:
            if isinstance(self.model["waves"], u.Quantity):
                outvals = self.model["waves"].to(u.micron)
            else:
                outvals = self.model["waves"]
            col1 = fits.Column(name="MOD_WAVE", format="E", array=outvals)
            col2 = fits.Column(name="MOD_EXT", format="E", array=self.model["exts"])
            col3 = fits.Column(
                name="RESIDUAL", format="E", array=self.model["residuals"]
            )
            columns = fits.ColDefs([col1, col2, col3])
            tbhdu = fits.BinTableHDU.from_columns(columns)
            if "params" in self.model.keys():
                # add the paramaters and their uncertainties
                for i, param in enumerate(self.model["params"]):
                    # add numbers to make sure all keywords are unique
                    tbhdu.header.set(
                        param.name[:6] + str(i).zfill(2),
                        param.value,
                        param.name
                        + " | bounds="
                        + str(param.bounds)
                        + " | fixed="
                        + str(param.fixed),
                    )
                    tbhdu.header.set(
                        param.name[0] + "_MUNC" + str(i).zfill(2),
                        param.unc_minus,
                        param.name + " lower uncertainty",
                    )
                    tbhdu.header.set(
                        param.name[0] + "_PUNC" + str(i).zfill(2),
                        param.unc_plus,
                        param.name + " upper uncertainty",
                    )
            tbhdu.header.set("MOD_TYPE", self.model["type"], "Type of fitted model")
            if "chi2" in self.model.keys():
                tbhdu.header.set(
                    "chi2", self.model["chi2"], "Chi squared for the fitted model"
                )
            tbhdu.header.set("EXTNAME", "MODEXT", "Fitted model extinction")
            hdulist.append(tbhdu)

        hdulist.writeto(ext_filename, overwrite=True)

    def read(self, ext_filename):
        """
        Read in a saved extinction curve from a FITS file

        Parameters
        ----------
        filename : string
            Full filename of the saved extinction curve
        """
        # read in the FITS file
        hdulist = fits.open(ext_filename)

        # get the list of extension names
        extnames = [hdulist[i].name for i in range(len(hdulist))]

        # the extinction curve itself
        for curname in _poss_datasources:
            curext = "%sEXT" % curname
            if curext in extnames:
                self.waves[curname] = hdulist[curext].data["WAVELENGTH"] * u.micron
                self.exts[curname] = hdulist[curext].data["EXT"]
                if "UNC" in hdulist[curext].data.columns.names:
                    self.uncs[curname] = hdulist[curext].data["UNC"]
                else:
                    self.uncs[curname] = hdulist[curext].data["EXT_UNC"]
                if "STDS" in hdulist[curext].data.columns.names:
                    self.stds[curname] = hdulist[curext].data["STDS"]
                if "NPTS" in hdulist[curext].data.columns.names:
                    self.npts[curname] = hdulist[curext].data["NPTS"]
                else:
                    self.npts[curname] = np.full(len(self.waves[curname]), 1)
                if "NAME" in hdulist[curext].data.columns.names:
                    self.names[curname] = hdulist[curext].data["NAME"]

        # get the parameters of the extinction curve
        pheader = hdulist[0].header
        self.type = pheader.get("EXTTYPE")
        self.type_rel_band = pheader.get("EXTBAND")
        if self.type_rel_band is None:
            self.type_rel_band = "V"
        self.red_file = pheader.get("R_FILE")
        self.comp_file = pheader.get("C_FILE")

        column_keys = ["AV", "EBV", "RV", "IRV", "LOGHI", "LOGHIMW", "NHIAV"]
        for curkey in column_keys:
            if pheader.get(curkey):
                if pheader.get("%s_UNC" % curkey):
                    tunc = (
                        float(pheader.get(curkey)),
                        float(pheader.get("%s_UNC" % curkey)),
                    )
                elif pheader.get("%s_PUNC" % curkey):
                    tunc = (
                        float(pheader.get(curkey)),
                        float(pheader.get("%s_MUNC" % curkey)),
                        float(pheader.get("%s_PUNC" % curkey)),
                    )
                else:
                    tunc = (float(pheader.get(curkey)), 0.0)
                self.columns[curkey] = tunc

        # get the fitted model if available
        if "MODEXT" in extnames:
            data = hdulist["MODEXT"].data
            hdr = hdulist["MODEXT"].header
            self.model["waves"] = data["MOD_WAVE"]
            self.model["exts"] = data["MOD_EXT"]
            self.model["residuals"] = data["RESIDUAL"]
            self.model["params"] = []
            paramkeys = [
                "AMPLIT00",
                "X_001",
                "ALPHA02",
                "SCALE03",
                "X_O04",
                "GAMMA_05",
                "ASYM06",
                "SCALE07",
                "X_O08",
                "GAMMA_09",
                "ASYM10",
                "AV11",
                "AV03",
                "AV07",
            ]
            self.model["type"] = hdr["MOD_TYPE"]
            for paramkey in paramkeys:
                if paramkey in list(hdr.keys()):
                    comment = hdr.comments[paramkey].split(" |")
                    param = Parameter(
                        name=comment[0],
                        default=hdr[paramkey],
                        bounds=comment[1].split("=")[1],
                        fixed=comment[2].split("=")[1],
                    )
                    param.unc_minus = hdr[paramkey[0] + "_MUNC" + paramkey[-2:]]
                    param.unc_plus = hdr[paramkey[0] + "_PUNC" + paramkey[-2:]]
                    self.model["params"].append(param)

        # get the columns p50 +unc -unc fit parameters if they exist
        if pheader.get("AV_p50"):
            self.columns_p50_fit = {}
            for bkey in column_keys:
                if pheader.get(f"{bkey}_p50"):
                    val = float(pheader.get(f"{bkey}_p50"))
                    punc = float(pheader.get(f"{bkey}_punc"))
                    munc = float(pheader.get(f"{bkey}_munc"))
                    self.columns_p50_fit[bkey] = (val, punc, munc)

        # get FM90 parameters if they exist
        #   include variant with B3=C3/gamma^2 instead of C3
        FM90_keys = ["C1", "C2", "C3", "B3", "C4", "XO", "GAMMA"]
        if pheader.get("C2"):
            self.fm90_best_fit = {}
            for curkey in FM90_keys:
                if pheader.get(curkey):
                    self.fm90_best_fit[curkey] = float(pheader.get("%s" % curkey))
            # for completeness, populate C1 using from the FM07 relationship
            # if not already present
            if "C1" not in self.fm90_best_fit.keys():
                self.fm90_best_fit["C1"] = (
                    2.09 - 2.84 * self.fm90_best_fit["C2"][0],
                    2.84 * self.fm90_best_fit["C2"][1],
                )

        # get the FM90 p50 +unc -unc fit parameters if they exist
        if pheader.get("C2_p50"):
            self.fm90_p50_fit = {}
            for bkey in FM90_keys:
                if pheader.get(f"{bkey}_p50"):
                    val = float(pheader.get(f"{bkey}_p50"))
                    punc = float(pheader.get(f"{bkey}_punc"))
                    munc = float(pheader.get(f"{bkey}_munc"))
                    self.fm90_p50_fit[bkey] = (val, punc, munc)

        # get P92 best fit parameters if they exist
        P92_mkeys = ["BKG", "FUV", "NUV", "SIL1", "SIL2", "FIR"]
        P92_types = ["AMP", "LAMBDA", "WIDTH", "B", "N"]
        if pheader.get("BKG_amp"):
            self.p92_best_fit = {}
            for curmkey in P92_mkeys:
                for curtype in P92_types:
                    curkey = "%s_%s" % (curmkey, curtype)
                    if pheader.get(curkey):
                        self.p92_best_fit[curkey] = float(pheader.get("%s" % curkey))

        # get the P92 p50 +unc -unc fit parameters if they exist
        if pheader.get("BKG_amp_p50"):
            self.p92_p50_fit = {}
            for curmkey in P92_mkeys:
                for curtype in P92_types:
                    bkey = "%s_%s" % (curmkey, curtype)
                    if pheader.get(f"{bkey}_p50"):
                        val = float(pheader.get(f"{bkey}_p50"))
                        punc = float(pheader.get(f"{bkey}_punc"))
                        munc = float(pheader.get(f"{bkey}_munc"))
                        self.p92_p50_fit[bkey] = (val, punc, munc)

        # get G21 parameters if they exist
        # fmt: off
        G21_keys = ["SCALE", "ALPHA",
                    "SIL1_AMP", "SIL1_CENTER", "SIL1_FWHM", "SIL1_ASYM",
                    "SIL2_AMP", "SIL2_CENTER", "SIL2_FWHM", "SIL2_ASYM"]
        # fmt: on
        if pheader.get("SIL1_CENTER"):
            self.g21_best_fit = {}
            for curkey in G21_keys:
                if pheader.get(curkey):
                    self.g21_best_fit[curkey] = float(pheader.get("%s" % curkey))

        # get the G21 p50 +unc -unc fit parameters if they exist
        if pheader.get("SIL1_CENTER_p50"):
            self.g21_p50_fit = {}
            for bkey in G21_keys:
                if pheader.get(f"{bkey}_p50"):
                    val = float(pheader.get(f"{bkey}_p50"))
                    punc = float(pheader.get(f"{bkey}_punc"))
                    munc = float(pheader.get(f"{bkey}_munc"))
                    self.g21_p50_fit[bkey] = (val, punc, munc)

    def _get_ext_ytitle(self, ytype=None):
        """
        Format the extinction type nicely for plotting

        Returns
        -------
        ptype : string
            Latex formatted string for plotting
        """
        if not ytype:
            ytype = self.type

        if isinstance(self.type_rel_band, str):
            relband = self.type_rel_band.replace("_", "")
        else:
            relband = f"{self.type_rel_band}"

        if ytype == "elx":
            return rf"$E(\lambda - {relband})$"
        elif ytype == "alax":
            return rf"$A(\lambda)/A({relband})$"
        elif ytype == "elv":
            return r"$E(\lambda - V)$"
        elif ytype == "elvebv":
            return r"$E(\lambda - V)/E(B - V)$"
        elif ytype == "alav":
            return r"$A(\lambda)/A(V)$"
        else:
            return "%s (not found)" % ytype

    def plot(
        self,
        pltax,
        color=None,
        alpha=None,
        alax=False,
        wavenum=False,
        exclude=[],
        normval=1.0,
        yoffset=0.0,
        rebin_fac=None,
        annotate_key=None,
        annotate_wave_range=None,
        annotate_text=None,
        annotate_rotation=0.0,
        annotate_yoffset=0.0,
        annotate_color="k",
        legend_key=None,
        legend_label=None,
        fontsize=None,
        model=False,
    ):
        """
        Plot an extinction curve

        Parameters
        ----------
        pltax : matplotlib plot object

        color : matplotlib color [default=None]
            color to use for all the data

        alpha : float [default=None]
            transparency value (0=transparent, 1=opaque)

        alax : boolean [default=False]
            convert from E(lambda-X) using A(X), if necessary
            plot A(lambda)/A(X)

        wavenum : boolean [default=False]
            plot x axis as 1/wavelength as is standard for UV extinction curves

        exclude : list of strings [default=[]]
            List of data type(s) to exclude from the plot (e.g., "IRS", "IRAC1",...)

        normval : float [default=1.0]
            Normalization value

        yoffset : float [default=0.0]
            additive offset for the data

        rebin_fac : int [default=None]
            factor by which to rebin the extinction curve

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

        legend_label : string [default=None]
            label to use for legend

        fontsize : int [default=None]
            fontsize for plot

        model : boolean
            if set and the model exists, plot it
        """
        if alax:
            # transform the extinctions from E(lambda-V) to A(lambda)/A(V)
            self.trans_elv_alav()

        for curtype in self.waves.keys():
            # do not plot the excluded data type(s)
            if curtype in exclude:
                continue
            # replace extinction values by NaNs for wavelength regions that need to be excluded from the plot
            if np.sum(self.npts[curtype] == 0) > 0:
                self.exts[curtype][self.npts[curtype] == 0] = np.nan
            x = self.waves[curtype].to(u.micron).value
            y = self.exts[curtype]
            yu = self.uncs[curtype]

            y = y / normval + yoffset
            yu = yu / normval

            if wavenum:
                x = 1.0 / x

            if (legend_key is not None) and (legend_key == curtype):
                clabel = legend_label
            else:
                clabel = None

            if curtype == "BAND":
                # do not plot the excluded band(s)
                for i, bandname in enumerate(self.names[curtype]):
                    if bandname in exclude:
                        y[i] = np.nan
                # plot band data as points with errorbars
                pltax.errorbar(
                    x,
                    y,
                    yerr=yu,
                    fmt="o",
                    color=color,
                    alpha=alpha,
                    mfc="white",
                    label=clabel,
                )
            else:
                if rebin_fac is not None:
                    x, y = _rebin(x, y, rebin_fac)

                pltax.plot(x, y, "-", color=color, alpha=alpha, label=clabel)

            if curtype == annotate_key:
                (ann_indxs,) = np.where(
                    (x >= annotate_wave_range[0].value)
                    & (x <= annotate_wave_range[1].value)
                )
                ann_val = np.nanmedian(y[ann_indxs])
                ann_val += (annotate_yoffset,)
                ann_xval = 0.5 * np.sum(annotate_wave_range.value)

                pltax.text(
                    ann_xval,
                    ann_val,
                    annotate_text,
                    color=annotate_color,
                    horizontalalignment="left",
                    rotation=annotate_rotation,
                    rotation_mode="anchor",
                    fontsize=fontsize,
                )

            # plot the model if desired
            if model:
                x = self.model["waves"]
                if wavenum:
                    x = 1.0 / x
                y = self.model["exts"]

                y = y / normval + yoffset

                pltax.plot(x, y, "-", color=color, alpha=alpha)

            if wavenum:
                xtitle = r"$1/\lambda$ $[\mu m^{-1}]$"
            else:
                xtitle = r"$\lambda$ $[\mu m]$"
            pltax.set_xlabel(xtitle)
            pltax.set_ylabel(self._get_ext_ytitle())

    def fit_band_ext(self):
        """
        Fit the observed NIR extinction curve with a powerlaw model, based on the band data between 1 and 40 micron

        Returns
        -------
        Updates self.model["waves", "exts", "residuals", "params"] and self.columns["AV"] with the fitting results:
            - waves: np.ndarray with the wavelengths used in the fitting
            - exts: np.ndarray with the fitted powerlaw model to the extinction curve
            - residuals: np.ndarray with the fractional residuals, i.e. (data-fit)/fit
            - params: tuple with the parameters (amplitude, alpha) if data in A(lambda)/A(V) or (amplitude, alpha, A(V)) if data in E(lambda-V)
        """
        # retrieve the band data to be fitted
        ftype = "BAND"
        gbool = np.all(
            [
                (self.npts[ftype] > 0),
                (self.waves[ftype] > 1.0 * u.micron),
                (self.waves[ftype] < 40.0 * u.micron),
            ],
            axis=0,
        )
        waves = self.waves[ftype][gbool].value
        exts = self.exts[ftype][gbool]

        # fit the data points with a powerlaw function (function must take the independent variable as the first argument and the parameters to fit as separate remaining arguments)
        if self.type == "alav":

            def alav_powerlaw(x, a, alpha):
                return a * x**-alpha

            func = alav_powerlaw
        else:

            def elx_powerlaw(x, a, alpha, c):
                return a * x**-alpha - c

            func = elx_powerlaw
        fit_result = curve_fit(func, waves, exts)

        # save the fitting results
        self.model["waves"] = waves
        self.model["exts"] = func(waves, *fit_result[0])
        self.model["residuals"] = exts - self.model["exts"]
        self.model["params"] = tuple(fit_result[0])
        if self.type != "alav":
            self.columns["AV"] = (fit_result[0][2], 0.0)

    def fit_spex_ext(
        self, amp_bounds=(-1.5, 1.5), index_bounds=(0.0, 5.0), AV_bounds=(0.0, 6.0)
    ):
        """
        Fit the observed NIR extinction curve with a powerlaw model, based on the SpeX spectra

        Parameters
        ----------
        amp_bounds : tuple [default=(-1.5,1.5)]
            Model amplitude bounds to be used in the fitting

        index_bounds : tuple [default=(0.0,5.0)]
            Powerlaw index bounds to be used in the fitting

        AV_bounds : tuple [default=(0.0,6.0)]
            A(V) bounds to be used in the fitting

        Returns
        -------
        Updates self.model["waves", "exts", "residuals", "params"] and self.columns["AV"] with the fitting results:
            - waves: np.ndarray with the wavelengths used in the fitting
            - exts: np.ndarray with the fitted powerlaw model to the extinction curve
            - residuals: np.ndarray with the fractional residuals, i.e. (data-fit)/fit
            - params: tuple with the parameters (amplitude, alpha) if data in A(lambda)/A(V) or (amplitude, alpha, A(V)) if data in E(lambda-V)
        """
        # retrieve the SpeX data, and sort the curve from short to long wavelengths
        (waves, exts, exts_unc) = self.get_fitdata(["SpeX_SXD", "SpeX_LXD"])
        indx = np.argsort(waves)
        waves = waves[indx].value
        exts = exts[indx]
        exts_unc = exts_unc[indx]

        # fit a powerlaw to the spectrum
        if self.type == "alav":
            func = PowerLaw1D(
                fixed={"x_0": True},
                bounds={"amplitude": amp_bounds, "alpha": index_bounds},
            )
        else:
            func = PowerLaw1D(
                fixed={"x_0": True},
                bounds={"amplitude": amp_bounds, "alpha": index_bounds},
            ) | AxAvToExv(bounds={"Av": AV_bounds})

        fit = LevMarLSQFitter()
        fit_result = fit(func, waves, exts, weights=1 / exts_unc)

        # save the fitting results
        self.model["waves"] = waves
        self.model["exts"] = fit_result(waves)
        self.model["residuals"] = exts - self.model["exts"]
        if self.type == "alav":
            self.model["params"] = (fit_result.amplitude.value, fit_result.alpha.value)
        else:  # in this case, fitted amplitude has to be multiplied by A(V) to get the "combined" amplitude
            self.model["params"] = (
                fit_result.amplitude_0.value * fit_result.Av_1.value,
                fit_result.alpha_0.value,
                fit_result.Av_1.value,
            )
            self.columns["AV"] = (fit_result.Av_1.value, 0.0)
