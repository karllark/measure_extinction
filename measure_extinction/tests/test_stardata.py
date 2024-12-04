import astropy.units as u

from measure_extinction.utils.helpers import get_datapath
from measure_extinction.stardata import StarData


def test_load_stardata():
    # get the location of the data files
    data_path = get_datapath()

    # read in the observed data on the star
    star = StarData("hd229238.dat", path=data_path)

    assert "BAND" in star.data.keys()
    assert "IUE" in star.data.keys()


def test_units_stardata():
    # get the location of the data files
    data_path = get_datapath()

    # read in the observed data of the star
    star = StarData("hd229238.dat", path=data_path)

    # test that the quantities have units
    for cursrc in star.data.keys():
        assert isinstance(star.data[cursrc].waves, u.Quantity)
        assert isinstance(star.data[cursrc].wave_range, u.Quantity)
        assert isinstance(star.data[cursrc].fluxes, u.Quantity)
        assert isinstance(star.data[cursrc].uncs, u.Quantity)

    fluxunit = u.erg / ((u.cm**2) * u.s * u.angstrom)
    # check that the wavelengths can be converted to microns and the
    # flux units can be converted to spectral density
    for cursrc in star.data.keys():
        twave = star.data[cursrc].waves.to(u.micron)
        assert twave.unit == u.micron

        tflux = star.data[cursrc].fluxes.to(
            fluxunit, equivalencies=u.spectral_density(twave)
        )
        assert tflux.unit == fluxunit
