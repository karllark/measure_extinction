#!/usr/bin/env python
# fit a stellar model atmosphere model + dust extinction model to
# observed spectra and photometry

import glob
import argparse

import numpy as np
import scipy.optimize as op

import matplotlib.pyplot as plt
import matplotlib as mpl

import astropy.units as u

import emcee

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.modeldata import ModelData
from measure_extinction.utils.helpers import get_full_starfile


# lnp_bignnum = -1e20
lnp_bignnum = -np.inf


class FitInfo(object):
    """
    Fitting constraints, likelihood functions, etc.
    Easier setup for passing information to likelihood functions.

    Parameters
    ----------
    parameter_names : strings
        names of the parameters

    parameter_limits : 2xn array (n=number of parameters)
        min/max parameter limits

    weights : dict of float arrays
        weights for each spectral region
        each entry is n wavelengths in length

    parameter_priors :
        priors on select parameters

    stellar_velocitiy : float, optional
        velocity of star (also assumed to be gal HI velocity)
    """

    def __init__(
        self,
        parameter_names,
        parameter_limits,
        weights,
        parameter_priors=None,
        stellar_velocity=0.0,
    ):

        self.parameter_names = parameter_names
        self.parameter_limits = parameter_limits
        self.parameter_priors = parameter_priors
        self.weights = weights
        self.velocities = np.array([stellar_velocity, 0.0])

    def lnlike(self, params, obsdata, modeldata, fit_range="all"):
        """
        Compute the natural log of the likelihood that the data
        fits the model.

        Parameters
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


def fit_model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of star")
    parser.add_argument(
        "--spteff",
        metavar=("logteff", "logteff_unc"),
        type=float,
        nargs=2,
        help="Spectral type prior: log(Teff) sigma_log(Teff)",
    )
    parser.add_argument(
        "--splogg",
        metavar=("logg", "logg_unc"),
        type=float,
        nargs=2,
        help="Spectral type prior: log(g) sigma_log(g)",
    )
    parser.add_argument(
        "--splogz",
        metavar=("logz", "logz_unc"),
        type=float,
        nargs=2,
        help="Spectral type prior: log(Z) sigma_log(Z)",
    )
    parser.add_argument("--velocity", type=float, default=0.0, help="Stellar velocity")
    parser.add_argument(
        "--relband", default="V", help="Band to use for relative extinction measurement"
    )
    parser.add_argument(
        "-f", "--fast", help="Use minimal step debug code", action="store_true"
    )
    parser.add_argument("--path", help="path to star/model files", default="./")
    parser.add_argument("--emcee", help="run EMCEE MCMC fitting", action="store_true")
    parser.add_argument(
        "--png", help="save the plot as a png file", action="store_true"
    )
    return parser


def get_best_fit_params(sampler):
    """
    Determine the best fit parameters given an emcee sampler object
    """
    max_lnp = -1e30
    nwalkers = len(sampler.lnprobability)
    for k in range(nwalkers):
        tmax_lnp = np.max(sampler.lnprobability[k])
        if tmax_lnp > max_lnp:
            max_lnp = tmax_lnp
            (indxs,) = np.where(sampler.lnprobability[k] == tmax_lnp)
            fit_params_best = sampler.chain[k, indxs[0], :]

    # ndim = len(fit_params_best)
    # params_best = np.zeros((ndim + 3))
    # params_best[0:ndim] = fit_params_best
    # params_best[ndim] = params_best[3] / params_best[4]
    # params_best[ndim + 1] = (10 ** params_best[10]) / params_best[3]
    # params_best[ndim + 2] = (10 ** params_best[10]) / params_best[ndim]

    return fit_params_best


def get_percentile_params(samples):
    """
    Determine the 50p plus/minus 33p vlaues
    """

    # add in E(B-V) and N(HI)/A(V) and N(HI)/E(B-V)
    samples_shape = samples.shape
    ndim = samples_shape[1]
    new_samples_shape = (samples_shape[0], samples_shape[1] + 3)
    new_samples = np.zeros(new_samples_shape)
    new_samples[:, 0:ndim] = samples
    new_samples[:, ndim] = samples[:, 3] / samples[:, 4]
    new_samples[:, ndim + 1] = (10 ** samples[:, 10]) / samples[:, 3]
    new_samples[:, ndim + 2] = (10 ** samples[:, 10]) / new_samples[:, ndim]

    per_params = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(new_samples, [16, 50, 84], axis=0)),
    )

    return per_params


if __name__ == "__main__":
    # commandline parser
    parser = fit_model_parser()
    args = parser.parse_args()
    out_basename = args.starname

    # get the full starfilename and path
    fstarname, file_path = get_full_starfile(args.starname)

    # get the observed reddened star data
    reddened_star = StarData(fstarname, path=file_path)
    band_names = reddened_star.data["BAND"].get_band_names()
    spectra_names = reddened_star.data.keys()

    # override for now
    print("possible", spectra_names)
    # spectra_names = ['BAND', 'STIS_Opt']
    # spectra_names = ['BAND', 'STIS_Opt']
    print("only using", spectra_names)

    # override for now
    # band_names = ['U', 'B', 'V']
    # band_names = ['U', 'B', 'V', 'J', 'H', 'K']
    print(band_names)

    # get just the filenames
    print("reading in the model spectra")
    tlusty_models_fullpath = glob.glob("{}/Models/tlusty_*v10.dat".format(file_path))
    # tlusty_models_fullpath = tlusty_models_fullpath[0:10]
    tlusty_models = [
        tfile[tfile.rfind("/") + 1 : len(tfile)] for tfile in tlusty_models_fullpath
    ]

    # get the models with just the reddened star band data and spectra
    modinfo = ModelData(
        tlusty_models,
        path="{}/Models/".format(file_path),
        band_names=band_names,
        spectra_names=spectra_names,
    )

    # parameters
    pnames = [
        "logT",
        "logg",
        "logZ",
        "Av",
        "Rv",
        "C2",
        "C3",
        "C4",
        "x0",
        "gamma",
        "HI_gal",
        "HI_mw",
    ]
    params = [4.3, 2.09, 0.2, 0.75, 3.7, 2.5, 0.65, 0.26, 4.66, 0.86, 22.0, 19.0]
    plimits = [
        [modinfo.temps_min, modinfo.temps_max],
        [modinfo.gravs_min, modinfo.gravs_max],
        [modinfo.mets_min, modinfo.mets_max],
        [0.0, 4.0],
        [2.0, 6.0],
        [-0.1, 5.0],
        [0.0, 2.5],
        [0.0, 1.0],
        [4.5, 4.9],
        [0.6, 1.5],
        [17.0, 24.0],
        [17.0, 22.0],
    ]
    ppriors = {}
    if args.spteff:
        ppriors["logT"] = args.spteff
        params[0] = args.spteff[0]
    if args.splogg:
        ppriors["logg"] = args.splogg
        params[1] = args.splogg[0]
    if args.splogz:
        ppriors["logZ"] = args.splogz
        params[2] = args.splogz[0]

    # cropping info for weights
    #  bad regions are defined as those were we know the models do not work
    #  or the data is bad
    ex_regions = [
        [8.23 - 0.1, 8.23 + 0.1],  # geocoronal line
        [8.7, 10.0],  # bad data from STIS
        [3.55, 3.6],
        [3.80, 3.90],
        [4.15, 4.3],
        [6.4, 6.6],
        [7.1, 7.3],
        [7.45, 7.55],
        [7.65, 7.75],
        [7.9, 7.95],
        [8.05, 8.1],
    ] / u.micron
    weights = {}
    for cspec in spectra_names:
        # based on observed uncs
        weights[cspec] = np.full(len(reddened_star.data[cspec].fluxes), 0.0)
        gvals = reddened_star.data[cspec].npts > 0
        weights[cspec][gvals] = 1.0 / reddened_star.data[cspec].uncs[gvals].value

        x = 1.0 / reddened_star.data[cspec].waves
        for cexreg in ex_regions:
            weights[cspec][np.logical_and(x >= cexreg[0], x <= cexreg[1])] = 0.0

    # make the photometric bands have higher weight
    weights["BAND"] *= 1000.0

    fitinfo = FitInfo(
        pnames,
        plimits,
        weights,
        parameter_priors=ppriors,
        stellar_velocity=args.velocity,
    )

    # find a good starting point via standard minimization
    # print('finding minimum')
    #
    def nll(*args):
        return -fitinfo.lnprob(*args)

    result = op.minimize(
        nll, params, method="Nelder-Mead", args=(reddened_star, modinfo, fitinfo)
    )
    params = result["x"]
    fit_params = params
    params_best = params
    pnames_extra = pnames

    # now run emcee to explore the region around the min solution
    if args.emcee:
        out_basename += "_emcee"
        print("starting point")
        print(params)

        print("exploring with emcee")
        p0 = params
        ndim = len(p0)
        if args.fast:
            nwalkers = 2 * ndim
            nsteps = 50
            burn = 50
        else:
            nwalkers = 100
            nsteps = 500
            burn = 500

        # setting up the walkers to start "near" the inital guess
        p = [p0 * (1 + 0.01 * np.random.normal(0, 1.0, ndim)) for k in range(nwalkers)]

        # setup the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, fitinfo.lnprob, args=(reddened_star, modinfo, fitinfo)
        )

        # burn in the walkers
        pos, prob, state = sampler.run_mcmc(p, burn)

        # rest the sampler
        sampler.reset()

        # do the full sampling
        pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)

        # create the samples variable for later use
        samples = sampler.chain.reshape((-1, ndim))

        # get the best fit values
        pnames_extra = pnames + ["E(B-V)", "N(HI)/A(V)", "N(HI)/E(B-V)"]
        params_best = get_best_fit_params(sampler)
        fit_params = params_best
        print("best params")
        print(params_best)

        # get the 16, 50, and 84 percentiles
        params_per = get_percentile_params(samples)
        print("percentile params")

        # save the best fit and p50 +/- uncs values to a file
        # save as a single row table to provide a uniform format
        f = open(out_basename + "_fit_params.dat", "w")
        f.write("# best fit, p50, +unc, -unc\n")
        for k, val in enumerate(params_per):
            print(
                "{} {} {} {} # {}".format(
                    params_best[k], val[0], val[1], val[2], pnames_extra[k]
                )
            )
            f.write(
                "{} {} {} {} # {}\n".format(
                    params_best[k], val[0], val[1], val[2], pnames_extra[k]
                )
            )

    else:
        f = open(out_basename + "_fit_params.dat", "w")
        f.write("# best fit\n")
        print(params_best)
        print(pnames_extra)
        for k, val in enumerate(params_best):
            print("{} # {}".format(val, pnames_extra[k]))
            f.write("{} # {}\n".format(val, pnames_extra[k]))

    f.close()

    # create the p50 parameters with symmetric error bars
    # params_50p = np.zeros(len(params_per))
    # params_50p_uncs = np.zeros(len(params_per))
    # for k, cval in enumerate(params_per):
    #     params_50p[k] = cval[0]
    #     params_50p_uncs[k] = 0.5*(cval[1] + cval[2])

    # *******************
    # plot

    # intrinsic sed
    modsed = modinfo.stellar_sed(fit_params[0:3], velocity=args.velocity)

    # dust_extinguished sed
    ext_modsed = modinfo.dust_extinguished_sed(fit_params[3:10], modsed)

    # hi_abs sed
    hi_ext_modsed = modinfo.hi_abs_sed(
        fit_params[10:12], [args.velocity, 0.0], ext_modsed
    )

    # create a StarData object for the best fit SED
    modsed_stardata = modinfo.SED_to_StarData(modsed)

    # create an extincion curve and save it
    extdata = ExtData()
    extdata.calc_elx(reddened_star, modsed_stardata, rel_band=args.relband)
    col_info = {"av": fit_params[3], "rv": fit_params[4]}
    extdata.save(out_basename + "_ext.fits", column_info=col_info)

    # plot the SEDs
    norm_model = np.average(hi_ext_modsed["BAND"])
    norm_data = np.average(reddened_star.data["BAND"].fluxes)

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=1)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    # setup the plot
    fig, ax = plt.subplots(figsize=(13, 10))

    # plot the bands and all spectra for this star
    for cspec in modinfo.fluxes.keys():
        if cspec == "BAND":
            ptype = "o"
        else:
            ptype = "-"

        # ax.plot(reddened_star.data[cspec].waves,
        #        weights[cspec], 'k-')

        ax.plot(
            reddened_star.data[cspec].waves,
            reddened_star.data[cspec].fluxes / norm_data,
            "k" + ptype,
            label="data",
        )

        # print(reddened_star.data[cspec].waves)
        # print(modinfo.waves[cspec])

        ax.plot(
            modinfo.waves[cspec], modsed[cspec] / norm_model, "b" + ptype, label=cspec
        )
        ax.plot(
            modinfo.waves[cspec],
            ext_modsed[cspec] / norm_model,
            "r" + ptype,
            label=cspec,
        )
        ax.plot(
            modinfo.waves[cspec],
            hi_ext_modsed[cspec] / norm_model,
            "g" + ptype,
            label=cspec,
        )

    # finish configuring the plot
    ax.set_ylim(8e4 / norm_model, 2e9 / norm_model)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=1.3 * fontsize)
    ax.set_ylabel(r"$F(\lambda)$ [$ergs\ cm^{-2}\ s\ \AA$]", fontsize=1.3 * fontsize)
    ax.tick_params("both", length=10, width=2, which="major")
    ax.tick_params("both", length=5, width=1, which="minor")

    # ax.legend()

    # use the whitespace better
    fig.tight_layout()

    if args.png:
        fig.savefig(out_basename + "_mod_obs_spec.png")
    else:
        plt.show()
