import argparse
import pickle
import emcee
import matplotlib.pyplot as plt
import numpy as np

from measure_extinction.model import MEModel
from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of star")
    parser.add_argument(
        "--plttype",
        help="Type of plot",
        choices=["chains", "corner", "bestfit"],
        default="bestfit",
    )
    parser.add_argument(
        "--residrange", help="residual range in percentage", default=10.0, type=float
    )
    parser.add_argument("--burnfrac", help="burn fraction", default=0.5, type=float)
    parser.add_argument(
        "--obspath",
        help="path to observed data",
        default="/home/kgordon/Python/extstar_data/MW/",
    )
    parser.add_argument(
        "--picmodname", help="name of pickled model", default="tlusty_z100_modinfo.p"
    )
    parser.add_argument(
        "--bands", help="only use these observed bands", nargs="+", default=None
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # base filename
    outname = f"figs/{args.starname}_mefit"

    # get the observed data
    fstarname = f"{args.starname}.dat"
    reddened_star = StarData(fstarname, path=f"{args.obspath}", only_bands=args.bands)

    # get the modeling info
    modinfo = pickle.load(open(args.picmodname, "rb"))

    # setup the ME model
    memod = MEModel(obsdata=reddened_star, modinfo=modinfo)

    # get the extinction curve
    tstr = outname.replace("figs", "exts")
    extfile = f"{tstr}_ext.fits"
    ext = ExtData(filename=extfile)
    ptab = ext.fit_params["MCMC"]
    for k, cname in enumerate(ptab["name"]):
        param = getattr(memod, cname)
        param.value = ptab["value"][k]
        if ptab["fixed"][k] > 0.1:
            param.fixed = True
        else:
            param.fixed = False
        if ptab["prior"][k] > 0.1:
            param.prior = (ptab["prior_val"][k], ptab["prior_unc"][k])
        else:
            param.prior = None

    # weights
    memod.fit_weights(reddened_star)

    # get the MCMC chains
    sampfile = f"{tstr}_.h5"
    reader = emcee.backends.HDFBackend(sampfile)
    samples = reader.get_chain()
    nsteps = samples.shape[0]

    # analyze chains for convergence
    tau = reader.get_autocorr_time(quiet=True)
    print("taus = ", tau)
    # burnin = int(2 * np.max(tau))
    # thin = int(0.5 * np.min(tau))
    # samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    # log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
    # log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

    # print("burn-in: {0}".format(burnin))
    # print("thin: {0}".format(thin))
    # print("flat chain shape: {0}".format(samples.shape))
    # print("flat log prob shape: {0}".format(log_prob_samples.shape))
    # print("flat log prior shape: {0}".format(log_prior_samples.shape))

    flat_samples = reader.get_chain(discard=int(args.burnfrac * nsteps), flat=True)

    # get the 50 percentile and +/- uncertainties
    params_per = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(flat_samples, [16, 50, 84], axis=0)),
    )

    # now package the fit parameters into two vectors, averaging the +/- uncs
    n_params = samples.shape[2]
    params_p50 = np.zeros(n_params)
    params_unc = np.zeros(n_params)
    for k, val in enumerate(params_per):
        params_p50[k] = val[0]
        params_unc[k] = 0.5 * (val[1] + val[2])

    # set the best fit parameters in the output model
    memod.fit_to_parameters(params_p50, uncs=params_unc)
    print(f"p50 parameters, burnfrac={args.burnfrac}")
    memod.pprint_parameters()

    if args.plttype == "chains":
        fig = memod.plot_sampler_chains(reader)
        save_str = "_mefit_chains"
    elif args.plttype == "corner":
        fig = memod.plot_sampler_corner(flat_samples)
        save_str = "_mefit_corner"
    else:
        fig = memod.plot(
            reddened_star, modinfo, lyaplot=True, resid_range=args.residrange
        )
        save_str = "_mefit_mcmc"

    # plot or save to a file
    if args.png:
        fig.savefig(f"{args.starname}{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{args.starname}{save_str}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
