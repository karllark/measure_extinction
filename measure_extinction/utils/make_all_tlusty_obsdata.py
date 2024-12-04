import glob
import argparse

from measure_extinction.utils.make_obsdata_from_model import make_obsdata_from_model


def decode_params_pre2022(filename):
    """
    Decode the tlusty filenames for the model parameters
    """
    model_params = {}

    slashpos = filename.rfind("/")
    periodpos = filename.rfind(".flux")

    # translation of metallicity code
    met_code = {"C": 2.0, "G": 1.0, "L": 0.5, "S": 0.2, "T": 0.1}
    met_char = filename[slashpos + 1 : slashpos + 2]
    T_start = slashpos + 1
    if met_char == "B":
        met_char = filename[slashpos + 2 : slashpos + 3]
        T_start += 1
        model_params["origin"] = "Bstar"
    else:
        model_params["origin"] = "Ostar"
    model_params["Z"] = met_code[met_char]

    gpos = filename.find("g", slashpos)
    vpos = filename.find("v", slashpos)

    model_params["Teff"] = float(filename[T_start + 1 : gpos])
    model_params["logg"] = float(filename[gpos + 1 : vpos]) * 0.01
    model_params["vturb"] = float(filename[vpos + 1 : periodpos])

    return model_params


def decode_params(filename):
    """
    Decode the tlusty filenames for the model parameters
    """
    model_params = {}

    slashpos = filename.rfind("/")
    periodpos = filename.rfind(".spec")

    zpos = filename.find("z", slashpos)
    tpos = filename.find("t", slashpos)
    gpos = filename.find("g", slashpos)
    vpos = filename.find("v", slashpos)

    if tpos - zpos > 4:
        model_params["Z"] = float(filename[zpos + 1 : tpos]) * 0.001
    else:
        model_params["Z"] = float(filename[zpos + 1 : tpos]) * 0.01
    model_params["Teff"] = float(filename[tpos + 1 : gpos])
    model_params["logg"] = float(filename[gpos + 1 : vpos]) * 0.01
    model_params["vturb"] = float(filename[vpos + 1 : periodpos])

    return model_params


def decode_params_wd(filename):
    """
    Decode the tlusty filenames for the model parameters
    """
    model_params = {}

    slashpos = filename.rfind("/")
    periodpos = filename.rfind(".spec")

    tpos = filename.find("t", slashpos)
    gpos = filename.find("g", slashpos)

    model_params["Teff"] = float(filename[tpos + 1 : gpos])
    model_params["logg"] = float(filename[gpos + 1 : periodpos - 1]) * 0.01
    model_params["vturb"] = 0.0
    model_params["Z"] = 1.0   # ratio to solar

    if model_params["Teff"] < 10000.0:
        model_params["Teff"] *= 100.0

    return model_params


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        choices=["tlusty", "wd_hubeny"],
        default="tlusty",
        help="Grid to use",
    )
    parser.add_argument(
        "--vturb",
        choices=["v2", "v5", "v10"],
        default="v2",
        help="Microturbulent velocity (only applies to tlusty grid)",
    )

    args = parser.parse_args()

    if args.grid == "wd_hubeny":
        mfilestr = "/home/kgordon/Python/extstar_data/Models/WD_Hubeny/*.spec"
        decodefunc = decode_params_wd
        outbase = "wd_hubeny"
    else:
        mfilestr = f"/home/kgordon/Python/extstar_data/Models/Tlusty_2023/*{args.vturb}.spec.gz"
        decodefunc = decode_params
        outbase = "tlusty"

    tlusty_models = glob.glob(mfilestr)

    for cfname in tlusty_models:
        # parse the filename to get the model parameters
        model_params = decodefunc(cfname)

        # get the base filename for the output files
        slashpos = cfname.rfind("/")
        periodpos = cfname.rfind(".spec")

        basename = f"{outbase}_{cfname[slashpos + 1 : periodpos]}"

        print(cfname)
        print(basename)
        print(model_params)

        make_obsdata_from_model(
            cfname,
            model_type="tlusty",
            output_filebase=basename,
            output_path="/home/kgordon/Python/extstar_data",
            model_params=model_params,
        )
