import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import G23
from dust_extinction.shapes import _curve_F99_method


class MEParameter(object):
    """
    Provide parameter info in a flexible format.
    Inspired by astropy modeling.
    """

    def __init__(self, value=0.0, bounds=(False, False), prior=None, fixed=False):
        self.value = value
        self.bounds = bounds
        self.prior = prior
        self.fixed = fixed


class MEModel(object):
    """
    Model
        Provide all the parameters for the measure_extinction fitting.
    Inspired by astropy modeling.
    """

    # fmt: off
    paramnames = ["logTeff", "logg", "logZ", "velocity",
                  "Av", "Rv", "C2", "B3", "C4", "xo", "gamma",
                  ]
    # fmt: on
    nparams = len(paramnames)

    # stellar
    logTeff = MEParameter(value=4.0, bounds=(0.0, False))
    logg = MEParameter(value=3.0, bounds=(0.0, False))
    logZ = MEParameter(value=0.0, bounds=(0.0, False))
    velocity = MEParameter(value=0.0, bounds=[-1000.0, 1000.0], fixed=True)  # km/s

    # dust - defaults based on VCG04 and FM07 MW samples (expect Av)
    Av = MEParameter(value=0.5, bounds=(0.0, False))
    Rv = MEParameter(value=3.0, bounds=(0.0, False))
    C2 = MEParameter(value=0.73, bounds=(0.0, False))
    B3 = MEParameter(value=3.6, bounds=(0.0, False))
    C4 = MEParameter(value=3.6, bounds=(0.0, False))
    xo = MEParameter(value=0.4, bounds=(0.0, False))
    gamma = MEParameter(value=0.89, bounds=(0.0, False))

    # gas
    vel_MW = MEParameter(value=0.0, bounds=(-300.0, 300.0))
    logHI_MW = MEParameter(value=20.0, bounds=(16., 24.))
    vel_exgal = MEParameter(value=0.0, bounds=(-300.0, 1000.0), fixed=True)
    logHI_exgal = MEParameter(value=16.0, bounds=(16., 24.), fixed=True)

    def parameters(self):
        """
        Give all the parameters values in a vector (fixed or not).
        
        Returns
        -------
        params : np array
            parameters values
        """
        vals = []
        for cname in self.paramnames:
            vals.append(getattr(self, cname).value)
        return np.array(vals)

    def parameters_to_fit(self):
        """
        Give the non-fixed parameters values in a vector.  Needed for most fitters/samplers.
        
        Returns
        -------
        params : np array
            non-fixed parameters values
        """
        vals = []
        for cname in self.paramnames:
            if not getattr(self, cname).fixed:
                vals.append(getattr(self, cname).value)
        return np.array(vals)

    def fit_to_parameters(self, fit_params):
        """
        Set the parameter values based on a vector of the non-fixed values.
        Needed for most fitters/samplers.
        
        Parameters
        ----------
        fit_params : np array
            non-fixed parameters values
        """
        vals = np.zeros(self.nparams)
        i = 0
        for k, cname in enumerate(self.paramnames):
            if not getattr(self, cname).fixed:
                setattr(self, cname, fit_params[i])
                i += 1

    def stellar_sed(self, moddata):
        """
        Compute the stellar SED from the model parameters

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

        Returns
        -------
        sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # compute the distance between model params and grid points
        #    probably a better way using a kdtree
        dist2 = (
            (self.logTeff.value - moddata.temps) ** 2 / moddata.temps_width2
            + (self.logg.value - moddata.gravs) ** 2 / moddata.gravs_width2
            + (self.logZ.value - moddata.mets) ** 2 / moddata.mets_width2
        )
        sindxs = np.argsort(dist2)
        gsindxs = sindxs[0 : moddata.n_nearest]

        # generate model SED form nearest neighbors
        #   should handle the case where dist2 has an element that is zero
        #   i.e., one of the precomputed models exactly matches the request
        weights = 1.0 / np.sqrt(dist2[gsindxs])
        weights /= np.sum(weights)

        sed = {}
        for cspec in moddata.fluxes.keys():
            # dot product does the multiplication and sum
            sed[cspec] = np.dot(weights, moddata.fluxes[cspec][gsindxs, :])

            sed[cspec][sed[cspec] == 0] = np.nan
            # shift spectrum if velocity given
            if self.velocity is not None:
                cwaves = moddata.waves[cspec]
                sed[cspec] = np.interp(
                    cwaves, (1.0 + self.velocity.value / 2.998e5) * cwaves, sed[cspec]
                )

        return sed

    def dust_extinguished_sed(self, moddata, sed, fit_range="all"):
        """
        Dust extinguished sed given the extinction parameters

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

        sed : dict
            fluxes for each spectral piece

        fit_range : string, optional
            keyword to toggle SED fitting to be done with G23 only or to also include curve_F99_method

        Returns
        -------
        extinguished sed : dict
            SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        g23mod = G23(Rv=self.Rv.value)

        # create the extinguished sed
        ext_sed = {}
        if fit_range.lower() == "g23":
            for cspec in moddata.fluxes.keys():
                shifted_waves = (1.0 - self.velocity.value / 2.998e5) * self.waves[cspec]
                axav = g23mod(shifted_waves)
                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * self.Av))

        elif fit_range.lower() == "all":
            optnir_axav_x = np.flip(1.0 / (np.arange(0.35, 30.0, 0.1) * u.micron))
            optnir_axav_y = g23mod(optnir_axav_x)

            # updated F04 C1-C2 correlation
            C1 = 2.18 - 2.91 * self.C2.value

            for cspec in moddata.fluxes.keys():
                # get the dust extinguished SED (account for the
                #  systemic velocity of the galaxy [opposite regular sense])
                shifted_waves = (1.0 - self.velocity.value / 2.998e5) * moddata.waves[cspec]

                # convert to 1/micron as _curve_F99_method does not do this (as of Nov 2024)
                with u.add_enabled_equivalencies(u.spectral()):
                    shifted_waves_imicron = u.Quantity(
                        shifted_waves, 1.0 / u.micron, dtype=np.float64
                    )

                axav = _curve_F99_method(
                    shifted_waves_imicron.value,
                    self.Rv.value,
                    C1,
                    self.C2.value,
                    self.B3.value,
                    self.C4.value,
                    xo=self.xo.value,
                    gamma=self.gamma.value,
                    optnir_axav_x=optnir_axav_x.value,
                    optnir_axav_y=optnir_axav_y,
                    fm90_version="B3",
                )

                ext_sed[cspec] = sed[cspec] * (10 ** (-0.4 * axav * self.Av.value))
        else:
            raise ValueError(
                "Incorrect input for fit_range argument in dust_extinguished_sed(). Available options are: g23, all"
            )

        # update the BAND fluxes by integrating the reddened MODEL_FULL spectrum
        if "BAND" in moddata.fluxes.keys():
            band_sed = np.zeros(moddata.n_bands)
            for k, cband in enumerate(moddata.band_names):
                gvals = np.isfinite(ext_sed["MODEL_FULL_LOWRES"])
                iwave = (1.0 - self.velocity.value / 2.998e5) * moddata.waves["MODEL_FULL_LOWRES"][
                    gvals
                ]
                iflux = ext_sed["MODEL_FULL_LOWRES"][gvals]
                iresp = moddata.band_resp[cband](iwave)
                inttop = np.trapezoid(iwave * iresp * iflux, iwave)
                intbot = np.trapezoid(iwave * iresp, iwave)
                band_sed[k] = inttop / intbot
            ext_sed["BAND"] = band_sed

        return ext_sed

    def hi_abs_sed(self, moddata, sed):
        """
        HI abs sed given the HI columns

        Parameters
        ----------
        moddata : ModelData object
            all the information about the model spectra

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
        for cspec in moddata.fluxes.keys():
            hi_sed[cspec] = np.copy(sed[cspec])
            (indxs,) = np.where(
                np.absolute((self.waves[cspec] - h_lines[0]) <= h_width)
            )
            if len(indxs) > 0:
                logHI_vals = [self.logHI_MW, self.logHI_exgal]
                for i, cvel in enumerate([self.vel_MW, self.vel_exgal]):
                    # compute the Ly-alpha abs: from Bohlin et al. (197?)
                    abs_wave = (1.0 + (cvel / 3e5)) * h_lines[0].to(u.micron).value
                    phi = 4.26e-20 / (
                        (1e4 * (moddata.waves[cspec][indxs].to(u.micron).value - abs_wave))
                        ** 2
                        + 6.04e-10
                    )

                    nhi = 10 ** logHI_vals[i]
                    hi_sed[cspec][indxs] = hi_sed[cspec][indxs] * np.exp(
                        -1.0 * nhi * phi
                    )

        return hi_sed

    def lnlike(self, params, obsdata, modeldata, fit_range="all"):
        """
        Compute the natural log of the likelihood that the data
        fits the model.

        Parameters
        ``
        ----------
        params : floats
            parameters of the model
            params = [logT, logg, logZ, Av, Rv, C2, C3, C4, x0, gamma, HI_gal, HI_mw]

        obsdata : StarData object
            observed data for a reddened star

        moddata : ModelData object
            all the information about the model spectra
        """
        # intrinsic sed
        modsed = modeldata.stellar_sed(params[0:3], velocity=self.velocities[0])

        # dust_extinguished sed
        ext_modsed = modeldata.dust_extinguished_sed(
            params[3:10], modsed, fit_range=fit_range, velocity=self.velocities[0]
        )

        # hi_abs sed
        hi_ext_modsed = modeldata.hi_abs_sed(params[10:12], self.velocities, ext_modsed)

        # compute the normalization factors for the model and observed data
        # model data normalized to the observations using the ratio
        #   weighted average of the averages of each type of data (photometry or specific spectrum)
        #   allows for all the data to contribute to the normalization
        #   weighting by number of points in each type of data to achieve the highest S/N in
        #     the normalization
        norm_mod = []
        norm_dat = []
        norm_npts = []
        for cspec in obsdata.data.keys():
            gvals = (self.weights[cspec] > 0) & (np.isfinite(hi_ext_modsed[cspec]))
            norm_npts.append(np.sum(gvals))
            norm_mod.append(np.average(hi_ext_modsed[cspec][gvals]))
            norm_dat.append(np.average(obsdata.data[cspec].fluxes[gvals].value))
        norm_model = np.average(norm_mod, weights=norm_npts)
        norm_data = np.average(norm_dat, weights=norm_npts)

        lnl = 0.0
        for cspec in obsdata.data.keys():
            try:
                gvals = (self.weights[cspec] > 0) & (np.isfinite(hi_ext_modsed[cspec]))
            except ValueError:
                raise ValueError(
                    "Oops! The model data and reddened star data did not match.\n Hint: Make sure that the BAND name in the .dat files match."
                )
            chiarr = np.square(
                (
                    (
                        obsdata.data[cspec].fluxes[gvals].value
                        - (hi_ext_modsed[cspec][gvals] * (norm_data / norm_model))
                    )
                    * self.weights[cspec][gvals]
                )
            )
            lnl += -0.5 * np.sum(chiarr)

        return lnl

    def lnprior(self, params):
        """
        Compute the natural log of the priors

        Parameters
        ----------
        params : floats
            parameters of the model
        """
        # make sure the parameters are within the limits
        for k, cplimit in enumerate(self.parameter_limits):
            if (params[k] < cplimit[0]) | (params[k] > cplimit[1]):
                # print('param limits excedded', k, params[k], cplimit)
                # exit()
                return lnp_bignnum

        # now use any priors
        lnp = 0.0
        if self.parameter_priors is not None:
            for ptype in self.parameter_priors.keys():
                pk = self.parameter_names.index(ptype)
                lnp += (
                    -0.5
                    * (
                        (
                            (params[pk] - self.parameter_priors[ptype][0])
                            / self.parameter_priors[ptype][1]
                        )
                    )
                    ** 2
                )

        return lnp

    @staticmethod
    def lnprob(params, obsdata, modeldata, fitinfo, fit_range="all"):
        """
        Compute the natural log of the probability

        Parameters
        ----------
        params : floats
            parameters of the model

        obsdata : StarData object
            observed data for a reddened star

        moddata : ModelData object
            all the information about the model spectra

        fitinfo : FitInfo object
            information about the fitting
        """
        lnp = fitinfo.lnprior(params)
        # print(params)
        # print(lnp, fitinfo.lnlike(params, obsdata, modeldata))
        if lnp == lnp_bignnum:
            return lnp
        else:
            return lnp + fitinfo.lnlike(params, obsdata, modeldata, fit_range=fit_range)

    def check_param_limits(self, params):
        """
        Check the parameters are within the parameter limits

        Parameters
        ----------
        params : floats
            parameters of the model
        """
        # make sure the parameters are within the limits
        for k, cplimit in enumerate(self.parameter_limits):
            if (params[k] < cplimit[0]) | (params[k] > cplimit[1]):
                print(
                    "param limits excedded", self.parameter_names[k], params[k], cplimit
                )
