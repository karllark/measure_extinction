import os
import importlib.resources as importlib_resources

__all__ = ["get_datapath"]


def get_datapath():
    """
    Determine the location of the data distributed along with the package
    """
    # get the location of the data files
    ref = importlib_resources.files("measure_extinction") / "data"
    with importlib_resources.as_file(ref) as cdata_path:
        data_path = str(cdata_path)
    return data_path
