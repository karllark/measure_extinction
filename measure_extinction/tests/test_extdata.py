import astropy.units as u
import numpy as np

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData, _hierarch_keywords, _get_column_val


def test_calc_ext():
    # get the location of the data files
    data_path = get_datapath()

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # test that the quantities have units (or not as appropriate)
    for cursrc in ext.waves.keys():
        assert isinstance(ext.waves[cursrc], u.Quantity)
        assert not isinstance(ext.exts[cursrc], u.Quantity)
        assert not isinstance(ext.uncs[cursrc], u.Quantity)
        assert not isinstance(ext.npts[cursrc], u.Quantity)

    # check that the wavelengths can be converted to microns
    for cursrc in ext.waves.keys():
        twave = ext.waves[cursrc].to(u.micron)
        assert twave.unit == u.micron


def test_get_fitdata():

    data_path = get_datapath()

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # once wavelenth units saved, update FITS file and use this line instead
    # of the 4 lines above

    # ext = ExtData(filename=data_path + "hd283809_hd064802_ext.fits")

    wave, y, unc = ext.get_fitdata(
        ["BAND", "IUE"], remove_uvwind_region=True, remove_lya_region=True
    )

    # fitting routines often cannot handle units, make sure none are present
    for cursrc in ext.waves.keys():
        assert isinstance(wave, u.Quantity)
        assert not isinstance(y, u.Quantity)
        assert not isinstance(unc, u.Quantity)


def test_calc_AV_RV():
    # get the location of the data files
    data_path = get_datapath()

    # read in the observed data of the stars
    redstar = StarData("hd229238.dat", path=data_path)
    compstar = StarData("hd204172.dat", path=data_path)

    # calculate the extinction curve
    ext = ExtData()
    ext.calc_elx(redstar, compstar)

    # calculate A(V)
    ext.calc_AV()
    np.testing.assert_almost_equal(ext.columns["AV"][0], 2.5626900237367805)

    # calculate R(V)
    ext.calc_RV()
    np.testing.assert_almost_equal(ext.columns["RV"][0], 2.614989769244703)


def test_hierarch_keyword():
    # input and expected keywords
    inkeys = ["AKEY", "AAAAKEY", "AAAAAKEY", "AAAAAAAKEY"]
    expkeys = ["AKEY", "AAAAKEY", "HIERARCH AAAAAKEY", "HIERARCH AAAAAAAKEY"]
    # out keywords
    outkeys = _hierarch_keywords(inkeys)
    for ekey, okey in zip(expkeys, outkeys):
        assert ekey == okey


def test_get_column_val():
    # single float value
    np.testing.assert_almost_equal(_get_column_val(3.0), 3.0)
    # tuple
    np.testing.assert_almost_equal(_get_column_val((3.0, 1.0, 2.0)), 3.0)


def test_fit_band_ext():  # only for alax=False (for now)
    # get the location of the data files
    data_path = get_datapath()

    # read in the extinction curve data
    extdata = ExtData(f"{data_path}/hd229238_hd204172_ext.fits")

    # fit the extinction curve with a powerlaw based on the band data
    extdata.fit_band_ext()

    # test the fitting results
    waves, exts, res = np.loadtxt(
        f"{data_path}/fit_band_ext_result_hd229238_hd204172.txt", unpack=True
    )
    np.testing.assert_almost_equal(extdata.model["waves"], waves)
    np.testing.assert_almost_equal(extdata.model["exts"], exts)
    np.testing.assert_almost_equal(extdata.model["residuals"], res)
    np.testing.assert_almost_equal(
        extdata.model["params"],
        (0.7593262393303228, 1.345528276482045, 2.6061368004634025),
    )
    np.testing.assert_almost_equal(extdata.columns["AV"][0], 2.6061368004634025)


def test_fit_spex_ext():  # only for alax=False (for now)
    # get the location of the data files
    data_path = get_datapath()

    # read in the extinction curve data
    extdata = ExtData(f"{data_path}/hd229238_hd204172_ext.fits")

    # fit the extinction curve with a powerlaw based on the SpeX data
    extdata.fit_spex_ext()

    # test the fitting results
    waves, exts, res = np.loadtxt(
        f"{data_path}/fit_spex_ext_result_hd229238_hd204172.txt", unpack=True
    )
    np.testing.assert_almost_equal(extdata.model["waves"], waves)
    np.testing.assert_almost_equal(extdata.model["exts"], exts)
    np.testing.assert_almost_equal(extdata.model["residuals"], res)
    np.testing.assert_almost_equal(
        extdata.model["params"],
        (0.8680132704511972, 2.023865293614347, 2.5626900237367805),
    )
    np.testing.assert_almost_equal(extdata.columns["AV"][0], 2.5626900237367805)
