import numpy as np
import astropy.units as u

from dust_extinction.shapes import _curve_F99_method
from dust_extinction.parameter_averages import G23

from measure_extinction.stardata import StarData, BandData, SpecData

__all__ = ["ModelData"]


class ModelData(object):
    """
    Provide stellar atmosphere model "observed" data given input stellar, gas,
    and dust extinction parameters.

    Parameters
    ----------
    modelfiles: string array
        set of model files to use

    path : string, optional
        path for model files

    band_names : string array, optional
        bands to use
        default = ['U', 'B', 'V', 'J', 'H', 'K']

    spectra_names : string array, optional
        origin of the spectra to use
        default = ['STIS']

    Attributes
    ----------
    n_models : int
        number of stellar atmosphere models
    model_files : string array
        filenames for the models

    temps : float array
        log10(effective temperatures)
    gravs : float array
        log10(surface gravities)
    mets : float array
        log10(metallicities)
    vturbs : float array
        microturbulance values [km/s]

    n_bands : int
        number of photometric bands
    band_names : string array
        names of the photometric bands

    n_spectra : int
        number of different types of spectra
    spectra_names : string array
        identifications for the spectra data (includes band data)
    waves : n_spectra dict
        wavelengths for the spectra
    fluxes : n_spectra dict
        fluxes in the bands
    flux_uncs : n_spectra list
        flux uncertainties in the bands
    """

    def __init__(
        self,
        modelfiles,
        path="./",
        band_names=["U", "B", "V", "J", "H", "K"],
        spectra_names=["BAND", "STIS"],
    ):

        self.n_models = len(modelfiles)
        self.model_files = np.array(modelfiles)

        # physical parameters of models
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)
        self.vturb = np.zeros(self.n_models)

        # photometric band data
        self.n_bands = len(band_names)
        self.band_names = band_names

        # photometric and spectroscopic data
        self.n_spectra = len(spectra_names) + 1
        self.spectra_names = spectra_names
        self.waves = {}
        self.fluxes = {}
        self.flux_uncs = {}

        for cspec in self.spectra_names:
            self.fluxes[cspec] = None
            self.flux_uncs[cspec] = None

        # initialize the BAND dictonary entry as the number of elements
        # is set by the desired bands, not the bands in the files
        self.waves["BAND"] = np.zeros((self.n_bands))
        self.fluxes["BAND"] = np.zeros((self.n_models, self.n_bands))
        self.flux_uncs["BAND"] = np.zeros((self.n_models, self.n_bands))

        # read and store the model data
        for k, cfile in enumerate(modelfiles):
            moddata = StarData(cfile, path=path)

            # model parameters
            self.temps[k] = np.log10(float(moddata.model_params["Teff"]))
            self.gravs[k] = float(moddata.model_params["logg"])
            self.mets[k] = np.log10(float(moddata.model_params["Z"]))
            self.vturb[k] = float(moddata.model_params["vturb"])

            # spectra
            for cspec in self.spectra_names:
                # initialize the spectra vectors
                if self.fluxes[cspec] is None:
                    self.waves[cspec] = moddata.data[cspec].waves
                    self.fluxes[cspec] = np.zeros(
                        (self.n_models, len(moddata.data[cspec].fluxes))
                    )
                    self.flux_uncs[cspec] = np.zeros(
                        (self.n_models, len(moddata.data[cspec].fluxes))
                    )

                # photometric bands
                if cspec == "BAND":
                    for i, cband in enumerate(self.band_names):
                        band_flux = moddata.data["BAND"].get_band_flux(cband)
                        self.waves[cspec][i] = band_flux[2]
                        self.fluxes[cspec][k, i] = band_flux[0]
                        self.flux_uncs[cspec][k, i] = band_flux[1]
                else:
                    # get the spectral data
                    self.fluxes[cspec][k, :] = moddata.data[cspec].fluxes
                    self.flux_uncs[cspec][k, :] = moddata.data[cspec].uncs

        # add units
        self.waves["BAND"] = self.waves["BAND"] * u.micron

        # provide the width in model space for each parameter
        #   used in calculating the nearest neighbors
        self.n_nearest = 11

        self.temps_min = min(self.temps)
        self.temps_max = max(self.temps)
        self.temps_width2 = (self.temps_max - self.temps_min) ** 2
        if self.temps_width2 == 0:
            self.temps_width2 == 1.0

        self.gravs_min = min(self.gravs)
        self.gravs_max = max(self.gravs)
        self.gravs_width2 = (self.gravs_max - self.gravs_min) ** 2
        if self.gravs_width2 == 0:
            self.gravs_width2 == 1.0

        self.mets_min = min(self.mets)
        self.mets_max = max(self.mets)
        self.mets_width2 = (self.mets_max - self.mets_min) ** 2
        if self.mets_width2 == 0:
            self.mets_width2 = 1.0
        # self.mets_width2 *= 4.0

    def stellar_sed(self, params, velocity=None):
        """
        Compute the stellar SED given model parameters

        Parameters
        ----------
        params : float array
            stellar atmosphere parameters [logT, logg, logZ]

        velocity : float
            stellar velocity in km/s

        Returns
        -------
        sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # compute the distance between model params and grid points
        #    probably a better way using a kdtree
        dist2 = (
            (params[0] - self.temps) ** 2 / self.temps_width2
            + (params[1] - self.gravs) ** 2 / self.gravs_width2
            + (params[2] - self.mets) ** 2 / self.mets_width2
        )
        sindxs = np.argsort(dist2)
        gsindxs = sindxs[0 : self.n_nearest]

        # generate model SED form nearest neighbors
        #   should handle the case where dist2 has an element that is zero
        #   i.e., one of the precomputed models exactly matches the request
        weights = 1.0 / np.sqrt(dist2[gsindxs])
        weights /= np.sum(weights)

        sed = {}
        for cspec in self.fluxes.keys():
            # dot product does the multiplication and sum
            sed[cspec] = np.dot(weights, self.fluxes[cspec][gsindxs, :])

            sed[cspec][sed[cspec] == 0] = np.nan
            # shift spectrum if velocity given
            if velocity is not None:
                cwaves = self.waves[cspec]
                sed[cspec] = np.interp(
                    cwaves, (1.0 + velocity / 2.998e5) * cwaves, sed[cspec]
                )

        return sed

    def dust_extinguished_sed(self, params, sed, fit_range="all", velocity=0.0):
        """
        Dust extinguished sed given the extinction parameters

        Parameters
        ----------
        params : float array
            dust extinction parameters [Av, Rv, c2, b3, c4, gamma, x0]

        sed : dict
            fluxes for each spectral piece

        fit_range : string, optional
            keyword to toggle SED fitting to be done with G23 only or to also include curve_F99_method

        velocity : float, optional
            velocity of dust

        Returns
        -------
        extinguished sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        Rv = params[1]
        g23mod = G23(Rv=Rv)

        # create the extinguished sed
        ext_sed = {}
        if fit_range.lower() == "g23":
            for cspec in self.fluxes.keys():
                shifted_waves = (1.0 - velocity / 2.998e5) * self.waves[cspec]
                axav = g23mod(shifted_waves)
                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * params[0]))

        elif fit_range.lower() == "all":
            optnir_axav_x = np.flip(1.0 / (np.arange(0.35, 30.0, 0.1) * u.micron))
            optnir_axav_y = g23mod(optnir_axav_x)

            # updated F04 C1-C2 correlation
            C1 = 2.18 - 2.91 * params[2]

            for cspec in self.fluxes.keys():
                # get the dust extinguished SED (account for the
                #  systemic velocity of the galaxy [opposite regular sense])
                shifted_waves = (1.0 - velocity / 2.998e5) * self.waves[cspec]

                # convert to 1/micron as _curve_F99_method does not do this (as of Nov 2024)
                with u.add_enabled_equivalencies(u.spectral()):
                    shifted_waves_imicron = u.Quantity(
                        shifted_waves, 1.0 / u.micron, dtype=np.float64
                    )

                axav = _curve_F99_method(
                    shifted_waves_imicron.value,
                    Rv,
                    C1,
                    params[2],  # C2
                    params[3],  # B3
                    params[4],  # C4
                    xo=params[5],  # xo
                    gamma=params[6],  # gamma
                    optnir_axav_x=optnir_axav_x.value,
                    optnir_axav_y=optnir_axav_y,
                    fm90_version="B3",
                )

                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * params[0]))

        else:
            raise ValueError(
                "Incorrect input for fit_range argument in dust_extinguished_sed(). Available options are: g23, all"
            )

        return ext_sed

    def dust_extinguished_sed_FM04(self, params, sed, velocity=0.0):
        """
        Dust extinguished sed given the extinction parameters

        Parameters
        ----------
        params : float array
            dust extinction parameters [Av, Rv, c2, c3, c4, gamma, x0]

        sed : dict
            fluxes for each spectral piece

        velocity : float, optional
            velocity of dust

        Returns
        -------
        extinguished sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        Rv = params[1]

        # updated F04 C1-C2 correlation
        C1 = 2.18 - 2.91 * params[2]

        # spline points
        opt_axav_x = 10000.0 / np.array([6000.0, 5470.0, 4670.0, 4110.0])
        # **Use NIR spline x values in FM07, clipped to K band for now
        nir_axav_x = np.array([0.50, 0.75, 1.0])
        optnir_axav_x = np.concatenate([nir_axav_x, opt_axav_x])

        # **Keep optical spline points from F99:
        #    Final optical spline point has a leading "-1.208" in Table 4
        #    of F99, but that does not reproduce Table 3.
        #    Additional indication that this is not correct is from
        #    fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        opt_axebv_y = np.array(
            [
                -0.426 + 1.0044 * Rv,
                -0.050 + 1.0016 * Rv,
                0.701 + 1.0016 * Rv,
                1.208 + 1.0032 * Rv - 0.00033 * (Rv**2),
            ]
        )
        # updated NIR curve from F04, note R dependence
        nir_axebv_y = (0.63 * Rv - 0.84) * nir_axav_x**1.84

        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # create the extinguished sed
        ext_sed = {}
        for cspec in self.fluxes.keys():
            # get the dust extinguished SED (account for the
            #  systemic velocity of the galaxy [opposite regular sense])
            shifted_waves = (1.0 - velocity / 2.998e5) * self.waves[cspec]
            axav = _curve_F99_method(
                shifted_waves,
                Rv,
                C1,
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
                optnir_axav_x,
                optnir_axebv_y / Rv,
                [0.2, 11.0],
                "F04_measure_extinction",
            )
            ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * params[0]))

        return ext_sed

    def hi_abs_sed(self, params, hi_velocities, sed):
        """
        HI abs sed given the HI columns

        Parameters
        ----------
        params : float array
            hi columns [log(HI_MW), log(HI_gal)]

        hi_velocities : float array
            hi velocities in km/sec [vel_MW, vel_gal]

        sed : dict
            fluxes for each spectral piece

        Returns
        -------
        hi absorbed sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # wavelengths of HI lines
        #     only use Ly-alpha right now - others useful later
        h_lines = (
            np.array(
                [
                    1215.0,
                    1025.0,
                    972.0,
                    949.0,
                    937.0,
                    930.0,
                    926.0,
                    923.0,
                    920,
                    919.0,
                    918.0,
                ]
            )
            * u.angstrom
        )
        # width overwhich to compute the HI abs
        h_width = 100.0 * u.angstrom

        hi_sed = {}
        for cspec in self.fluxes.keys():
            hi_sed[cspec] = np.copy(sed[cspec])
            (indxs,) = np.where(
                np.absolute((self.waves[cspec] - h_lines[0]) <= h_width)
            )
            if len(indxs) > 0:
                for i, cvel in enumerate(hi_velocities):
                    # compute the Ly-alpha abs: from Bohlin et al. (197?)
                    abs_wave = (1.0 + (cvel / 3e5)) * h_lines[0].to(u.micron).value
                    phi = 4.26e-20 / (
                        (1e4 * (self.waves[cspec][indxs].to(u.micron).value - abs_wave))
                        ** 2
                        + 6.04e-10
                    )

                    nhi = 10 ** params[i]
                    hi_sed[cspec][indxs] = hi_sed[cspec][indxs] * np.exp(
                        -1.0 * nhi * phi
                    )

        return hi_sed

    def SED_to_StarData(self, sed):
        """
        Convert the model created SED into a StarData object.
        Needed to plug into generating an ExtData object.

        Parameters
        ----------
        sed : object
            SED of each component
        """
        sd = StarData(None)

        for cspec in sed.keys():
            if cspec == "BAND":
                # populate the BAND info
                sd.data["BAND"] = BandData("BAND")
                sd.data["BAND"].fluxes = sed["BAND"] * (
                    u.erg / ((u.cm**2) * u.s * u.angstrom)
                )
                for k, cband in enumerate(self.band_names):
                    sd.data["BAND"].band_fluxes[cband] = (sed["BAND"][k], 0.0)
                sd.data["BAND"].get_band_mags_from_fluxes()

            else:
                # populate the spectral info
                sd.data[cspec] = SpecData(cspec)
                sd.data[cspec].fluxes = sed[cspec] * (
                    u.erg / ((u.cm**2) * u.s * u.angstrom)
                )

            sd.data[cspec].waves = self.waves[cspec]
            sd.data[cspec].n_waves = len(sd.data[cspec].waves)
            sd.data[cspec].uncs = 0.0 * sd.data[cspec].fluxes
            sd.data[cspec].npts = np.full((sd.data[cspec].n_waves), 1.0)
            sd.data[cspec].wave_range = [
                min(sd.data[cspec].waves),
                max(sd.data[cspec].waves),
            ]

        return sd
