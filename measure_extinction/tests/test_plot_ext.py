import os
import warnings
import pytest

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.plotting.plot_ext import (
    plot_multi_extinction,
    plot_extinction,
    plot_average,
)


@pytest.mark.skip(reason="failing due to changes in matplotlib")
def test_plot_extinction():
    # get the location of the data files
    data_path = get_datapath()
    starpair = "HD229238_HD204172"

    # plot the extinction curve
    # with the default settings
    plot_extinction(starpair, data_path, pdf=True)

    # as A(lambda)/A(V) (i.e. alax=True)
    plot_extinction(starpair, data_path, alax=True, pdf=True)

    # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
    plot_extinction(starpair, data_path, range=[0.7, 6], pdf=True)

    # test several other plotting options
    # note: verifying the existence of the pdf file with the plot is not sufficient to ensure that the different plotting options work as expected. However, these tests at least make sure that the corresponding functions run without errors.

    # with a fitted model overplotted, this should issue a warning
    with warnings.catch_warnings(record=True) as w:
        plot_extinction(starpair, data_path, fitmodel=True, pdf=True)
        assert issubclass(w[-1].category, UserWarning)
        assert "There is no fitted model available to plot." == str(w[-1].message)

    # with Milky Way extinction curve models overplotted
    plot_extinction(starpair, data_path, extmodels=True, pdf=True)

    # with band data excluded from the plot
    plot_extinction(starpair, data_path, exclude=["BAND"], pdf=True)

    # with HI-lines indicated
    plot_extinction(starpair, data_path, HI_lines=True, pdf=True)

    # in one figure, although this is actually only relevant if there is more than one curve
    starpair_list = ["HD229238_HD204172"]
    plot_multi_extinction(starpair_list, data_path, pdf=True)
    plot_multi_extinction(starpair_list, data_path, alax=True, pdf=True)
    plot_multi_extinction(
        starpair_list, data_path, alax=True, average=True, pdf=True
    )  # This needs to be checked for alax=False as well TODO
    plot_multi_extinction(starpair_list, data_path, HI_lines=True, pdf=True)
    plot_multi_extinction(starpair_list, data_path, range=[0.7, 6], pdf=True)
    plot_multi_extinction(starpair_list, data_path, exclude=["BAND"], pdf=True)

    with warnings.catch_warnings(record=True) as w:
        plot_multi_extinction(starpair_list, data_path, fitmodel=True, pdf=True)
    assert issubclass(w[-1].category, UserWarning)
    assert "There is no fitted model available to plot." == str(w[-1].message)

    # this option should issue a warning
    with warnings.catch_warnings(record=True) as w:
        plot_multi_extinction(starpair_list, data_path, extmodels=True, pdf=True)
        assert issubclass(w[-1].category, UserWarning)
        assert (
            "Overplotting Milky Way extinction curve models on a figure with multiple observed extinction curves in E(lambda-V) units is disabled, because the model curves in these units are different for every star, and would overload the plot. Please, do one of the following if you want to overplot Milky Way extinction curve models: 1) Use the flag --alax to plot ALL curves in A(lambda)/A(V) units, OR 2) Plot all curves separately by removing the flag --onefig."
            == str(w[-1].message)
        )

    # check if the expected pdf files were created
    message = (
        "Plotting the extinction curve of reddened star "
        + starpair.split("_")[0]
        + " with comparison star "
        + starpair.split("_")[1]
        + ", "
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_elx.pdf"), (
        message + "with the default settings, has failed."
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_alax.pdf"), (
        message + "in A(lambda)/A(V), has failed."
    )

    assert os.path.isfile(data_path + starpair.lower() + "_ext_elx_zoom.pdf"), (
        message + "in a specific wavelength range, has failed."
    )

    assert os.path.isfile(data_path + "all_ext_elx.pdf"), (
        message + "in one figure, with the default settings, has failed."
    )

    assert os.path.isfile(data_path + "all_ext_alax.pdf"), (
        message + "in one figure, in A(lambda)/A(V), has failed."
    )

    assert os.path.isfile(data_path + "all_ext_elx_zoom.pdf"), (
        message + "in one figure, in a specific wavelength range, has failed."
    )


def test_plot_average():
    # get the location of the data files
    data_path = get_datapath()

    # plot the average extinction curve in a separate figure
    # with the default settings
    plot_average(data_path, pdf=True)

    # with extinction models
    plot_average(data_path, extmodels=True, pdf=True)

    # with HI_lines
    plot_average(data_path, HI_lines=True, pdf=True)

    # zoomed in on a specific wavelength region (i.e. range=[0.7,6])
    plot_average(data_path, range=[0.7, 6], pdf=True)

    # with band data excluded from the plot
    plot_average(data_path, exclude=["BAND"], pdf=True)

    # with the wavelengths on a log-scale
    plot_average(data_path, log=True, pdf=True)

    # check if the expected pdf file was created
    assert os.path.isfile(
        data_path + "average_ext.pdf"
    ), "Plotting the average extinction curve has failed."
