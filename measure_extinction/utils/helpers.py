import importlib.resources as importlib_resources

import astropy.units as u
from astropy.table import QTable

__all__ = ["get_datapath", "read_bohlin"]


def get_datapath():
    """
    Determine the location of the data distributed along with the package
    """
    # get the location of the data files
    ref = importlib_resources.files("measure_extinction") / "data"
    with importlib_resources.as_file(ref) as cdata_path:
        data_path = str(cdata_path)
    return data_path


def read_bohlin(sfilename):
    """
    Read in a Ralph Bohlin custom formatted table.

    Parameters
    ----------
    sfilename : str
        filename of table

    Returns
    -------

    """
    # determine the line for the 1st data (can vary between files)
    f = open(sfilename, "r")
    k = 0
    for x in f:
        if "###" in x:
            dstart = k + 1
        else:
            k += 1
    f.close()

    # read
    t1 = QTable.read(
        sfilename,
        format="ascii",
        data_start=dstart,
        names=[
            "WAVELENGTH",
            "COUNT-RATE",
            "FLUX",
            "STAT-ERROR",
            "SYS-ERROR",
            "NPTS",
            "TIME",
            "QUAL",
        ],
    )

    t1["WAVELENGTH"] *= u.angstrom
    t1["FLUX"] *= u.erg / (u.s * u.cm * u.cm * u.angstrom)
    t1["ERROR"] = t1["STAT-ERROR"] * u.erg / (u.s * u.cm * u.cm * u.angstrom)
    t1["STAT-ERROR"] *= u.erg / (u.s * u.cm * u.cm * u.angstrom)

    return t1
