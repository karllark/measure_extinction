import os

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.utils.calc_ext import calc_extinction, calc_ave_ext


def test_calc_extinction():
    redstarname = "HD229238"
    compstarname = "HD204172"

    # get the location of the data files
    data_path = get_datapath()

    # calculate the extinction curve
    calc_extinction(redstarname, compstarname, data_path, savepath=data_path)

    # check if an extinction curve has been calculated and saved to a fits file
    assert os.path.isfile(
        f"{data_path}/{redstarname.lower()}_{compstarname.lower()}_ext.fits"
    ), (
        "No FITS file has been created with the extinction curve of reddened star "
        + redstarname
        + " with comparison star "
        + compstarname
    )


def test_calc_ave_ext():
    # get the location of the data files
    data_path = get_datapath()

    # list the same starpair 3 times so that an average curve will be calculated (it needs at least 3 sightlines)
    starpair_list = ["HD229238_HD204172", "HD229238_HD204172", "HD229238_HD204172"]

    # calculate the average extinction curve
    # this is actually not very useful if there is only one starpair, but at least the function will be run
    calc_ave_ext(starpair_list, data_path)

    # check if the average extinction curve has been calculated and saved to a fits file
    assert os.path.isfile(
        f"{data_path}/average_ext.fits"
    ), "No FITS file has been created with the average extinction curve"
