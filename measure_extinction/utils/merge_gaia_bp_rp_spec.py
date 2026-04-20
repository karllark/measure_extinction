import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import QTable
import astropy.units as u

from measure_extinction.merge_obsspec import obsspecinfo, merge_gen_obsspec


fluxunit = u.erg / (u.cm * u.cm * u.s * u.angstrom)


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star (filebase)")
    parser.add_argument(
        "--inpath",
        help="path where original data files are stored",
        default="./",
    )
    parser.add_argument(
        "--outpath",
        help="path where merged spectra will be stored",
        default="./",
    )
    parser.add_argument("--outname", help="Output filebase")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    sfilename = f"{args.inpath}{args.starname}.fits"
    sfiles = glob.glob(sfilename)
    cdatas = QTable.read(sfiles[0])
    for row in cdatas:
        for xp in ['bp', 'rp']:
            stable = []

            cdata_xp = QTable()
            cdata_xp["WAVELENGTH"] = row[f"wl_{xp}"] * 0.001 * u.micron
            cdata_xp["FLUX"] = row[f"flux_{xp}"] * 100 * fluxunit
            cdata_xp["ERROR"] = row[f"flux_err_{xp}"] * 100 * fluxunit
            cdata_xp["NPTS"] = np.full((len(cdata_xp["FLUX"])), 1.0)
            cdata_xp["NPTS"][cdata_xp["FLUX"] == 0.0] = 0.0
            stable.append(cdata_xp)

            cres, crange = obsspecinfo[f"gaia_{xp}"]
            rb_xp = merge_gen_obsspec(stable, crange, cres)
            if args.outname:
                outname = args.outname
            else:
                outname = row['ALS'].lower().replace('ls ', 'ls').replace(' ', '-')
                outname = outname.replace('gl', 'al')
            xp_file = f"{outname}_gaia_{xp}.fits"
            rb_xp.write(f"{args.outpath}/{xp_file}", overwrite=True)

            # plot the original and merged Spectra
            fontsize = 14
            font = {"size": fontsize}
            plt.rc("font", **font)
            plt.rc("lines", linewidth=2)
            plt.rc("axes", linewidth=2)
            plt.rc("xtick.major", width=2)
            plt.rc("ytick.major", width=2)

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5.5))

            for ctable in stable:
                gvals = ctable["NPTS"] > 0
                cfluxes = (
                    ctable["FLUX"]
                    .to(fluxunit, equivalencies=u.spectral_density(ctable["WAVELENGTH"]))
                    .value
                )
                ax.plot(
                    ctable["WAVELENGTH"][gvals],
                    cfluxes[gvals],
                    "k-",
                    alpha=0.5,
                    label="orig",
                )
            gvals = rb_xp["NPTS"] > 0
            cfluxes = (
                rb_xp["FLUX"]
                .to(fluxunit, equivalencies=u.spectral_density(ctable["WAVELENGTH"]))
                .value
            )

            ax.plot(
                rb_xp["WAVELENGTH"][gvals].to(u.micron),
                rb_xp["FLUX"][gvals],
                "b-",
                alpha=0.5,
                label="merged",
            )

            ax.set_xlabel(r"$\lambda$ [$\AA$]")
            ax.set_ylabel(r"F($\lambda$)")

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.legend()
            fig.tight_layout()

            fname = xp_file.replace(".fits", "")
            if args.png:
                fig.savefig(f"{fname}.png")
            elif args.eps:
                fig.savefig(f"{fname}.eps")
            else:
                fig.savefig(f"{fname}.pdf")

            plt.close()
