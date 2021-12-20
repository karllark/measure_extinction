import numpy as np
from astropy import uncertainty as unc
import astropy.units as u

from dust_extinction.parameter_averages import F19

if __name__ == "__main__":
    # simulate distributions for E(lam-V) for 4 bands
    # assume reddened/comparison star uncs add in quadrature
    #  V, J, H, K
    band_waves = np.array([0.45, 0.55, 1.1, 1.6, 2.2]) * u.micron
    redband_uncs = np.array([0.02, 0.02, 0.02, 0.025, 0.03])  # in mags
    compband_uncs = np.array([0.02, 0.02, 0.02, 0.025, 0.03])
    vindx = 1
    kindx = 4
    nbands = len(band_waves)

    # get alav and al values
    extmod = F19()
    alav = extmod(band_waves)
    av = 2.0
    # convert to elv
    elv = (alav - 1) * av
    elvuncs = np.sqrt(redband_uncs ** 2 + compband_uncs ** 2)

    n_samples = 1000
    reddist = []
    compdist = []
    elvdist = []
    for k in range(nbands):
        reddist.append(
            unc.normal(alav[k] * av, std=redband_uncs[k], n_samples=n_samples)
        )
        # just assume the comparion has zero mag in all bands
        # then reddened is just A(lam)
        compdist.append(unc.normal(0.0, std=compband_uncs[k], n_samples=n_samples))

    evdist = reddist[vindx] - compdist[vindx]
    for k in range(nbands):
        elvdist.append(reddist[k] - compdist[k] - evdist)

    avdist = -1.14 * elvdist[3]
    avmean = avdist.pdf_mean()
    avunc = avdist.pdf_std()
    print("av", avdist.pdf_mean(), avdist.pdf_std())
    alavdist = []
    for k in range(nbands):
        alavdist.append(elvdist[k] / avdist + 1.0)
        print(
            band_waves[k],
            elvdist[k].pdf_mean(),
            elvdist[k].pdf_std(),
            alavdist[k].pdf_mean(),
            alavdist[k].pdf_std(),
        )

    # compute the alav uncs based on error propagation and not simulation
    alavuncs = (
        np.sqrt(
            np.square(elvuncs)
            + np.square(elv * avunc / avmean)
        )
        / avmean
    )
    print(alavuncs)
    print(avunc / avmean)
