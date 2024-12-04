import os
import importlib.resources as importlib_resources

__all__ = ["get_full_starfile", "get_datapath"]


def get_full_starfile(starname):
    """
    Get full path to the DAT file for the specified star
    """
    if "dat" not in starname:
        fstarname = "%s.dat" % starname
    else:
        fstarname = starname

    if not os.path.isfile(fstarname):
        # not an installable package yet
        # def_path = pkg_resources.resource_filename(
        #    'extstar_data', '')
        def_path = "/home/kgordon/Python/extstar_data/"
        if not os.path.isfile("{}/DAT_files/{}".format(def_path, fstarname)):
            print("{} file not found".format(fstarname))
        else:
            fstarname = "DAT_files/{}".format(fstarname)
    else:
        def_path = "./"

    return fstarname, def_path


def get_datapath():
    """
    Determine the location of the data distributed along with the package
    """
    # get the location of the data files
    ref = importlib_resources.files("measure_extinction") / "data"
    with importlib_resources.as_file(ref) as cdata_path:
        data_path = str(cdata_path)
    return data_path
