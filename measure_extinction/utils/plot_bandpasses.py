from matplotlib import pyplot as plt
from synphot import SpectralElement

from measure_extinction.utils.helpers import get_datapath


def plot_bandpasses(bands):
    # create the figure
    plt.figure(figsize=(15, 5))

    # plot all response curves
    for band in bands:
        bp = SpectralElement.from_file(f"{band_path}/{band}.dat")
        if "MIPS" in band:
            wavelengths = bp.waveset * 10**4
        else:
            wavelengths = bp.waveset
        plt.plot(wavelengths, bp(bp.waveset), label=band)

    plt.xlabel(r"$\lambda$ [$\AA$]", size=15)
    plt.ylabel("transmission", size=15)
    plt.legend(ncol=2, loc=1)
    plt.show()


if __name__ == "__main__":
    # path for band response curves
    band_path = f"{get_datapath()}/Band_RespCurves/"

    # define the different bandpasses
    bands = [
        "JohnU",
        "JohnB",
        "JohnV",
        "JohnR",
        "JohnI",
        "JohnJ",
        "2MASSJ",
        "JohnH",
        "2MASSH",
        "JohnK",
        "JohnKs",
        "2MASSKs",
        "WISE1",
        "AAOL",
        "IRAC1",
        "AAOLprime",
        "IRAC2",
        "WISE2",
        "AAOM",
    ]
    plot_bandpasses(bands)
