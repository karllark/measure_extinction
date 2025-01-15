import numpy as np
import astropy.units as u
from synphot import SpectralElement
import stsynphot as STS


from measure_extinction.stardata import StarData, BandData, SpecData
from measure_extinction.utils.helpers import get_datapath

__all__ = ["ModelData"]


class ModelData(object):
    """
    Provide stellar atmosphere model "observed" data given input stellar, gas,
    and dust extinction parameters.

    Parameters
    ----------
    modelfiles: string array
        set of model files to use

    path : string, optional
        path for model files

    band_names : string array, optional
        bands to use
        default = ['U', 'B', 'V', 'J', 'H', 'K']

    spectra_names : string array, optional
        origin of the spectra to use
        default = ['STIS']

    Attributes
    ----------
    n_models : int
        number of stellar atmosphere models
    model_files : string array
        filenames for the models

    temps : float array
        log10(effective temperatures)
    gravs : float array
        log10(surface gravities)
    mets : float array
        log10(metallicities)
    vturbs : float array
        microturbulance values [km/s]

    n_bands : int
        number of photometric bands
    band_names : string array
        names of the photometric bands

    n_spectra : int
        number of different types of spectra
    spectra_names : string array
        identifications for the spectra data (includes band data)
    waves : n_spectra dict
        wavelengths for the spectra
    fluxes : n_spectra dict
        fluxes in the bands
    flux_uncs : n_spectra list
        flux uncertainties in the bands
    """

    def __init__(
        self,
        modelfiles,
        path="./",
        band_names=["U", "B", "V", "J", "H", "K"],
        spectra_names=["BAND", "STIS"],
    ):

        self.n_models = len(modelfiles)
        self.model_files = np.array(modelfiles)

        # physical parameters of models
        self.temps = np.zeros(self.n_models)
        self.gravs = np.zeros(self.n_models)
        self.mets = np.zeros(self.n_models)
        self.vturb = np.zeros(self.n_models)

        # photometric band data
        self.n_bands = len(band_names)
        self.band_names = band_names
        # path for non-HST band response curves
        band_resp_path = f"{get_datapath()}/Band_RespCurves/"

        # photometric and spectroscopic data +2 for "BANDS" and "MODEL_FULL"
        self.n_spectra = len(spectra_names) + 2
        # add in special "model_full" spectra for use in computing the reddened band fluxes
        self.spectra_names = spectra_names + ["MODEL_FULL_LOWRES"]
        self.waves = {}
        self.fluxes = {}
        self.flux_uncs = {}

        for cspec in self.spectra_names:
            self.fluxes[cspec] = None
            self.flux_uncs[cspec] = None

        if "BAND" in spectra_names:
            # initialize the BAND dictionary entry as the number of elements
            # is set by the desired bands, not the bands in the files
            self.waves["BAND"] = np.zeros((self.n_bands))
            self.fluxes["BAND"] = np.zeros((self.n_models, self.n_bands))
            self.flux_uncs["BAND"] = np.zeros((self.n_models, self.n_bands))
            self.band_resp = {}

        # read and store the model data
        for k, cfile in enumerate(modelfiles):
            moddata = StarData(cfile, path=path)

            # model parameters
            self.temps[k] = np.log10(float(moddata.model_params["Teff"]))
            self.gravs[k] = float(moddata.model_params["logg"])
            self.mets[k] = np.log10(float(moddata.model_params["Z"]))
            self.vturb[k] = float(moddata.model_params["vturb"])

            # spectra
            for cspec in self.spectra_names:
                # initialize the spectra vectors
                if self.fluxes[cspec] is None:
                    self.waves[cspec] = moddata.data[cspec].waves
                    self.fluxes[cspec] = np.zeros(
                        (self.n_models, len(moddata.data[cspec].fluxes))
                    )
                    self.flux_uncs[cspec] = np.zeros(
                        (self.n_models, len(moddata.data[cspec].fluxes))
                    )

                # photometric bands
                if cspec == "BAND":
                    for i, cband in enumerate(self.band_names):
                        band_flux = moddata.data["BAND"].get_band_flux(cband)
                        self.waves[cspec][i] = band_flux[2]
                        self.fluxes[cspec][k, i] = band_flux[0]
                        self.flux_uncs[cspec][k, i] = band_flux[1]

                        # read in the band response functions for determining the reddened photometry
                        if "ACS" in cband:
                            bp_info = cband.split("_")
                            bp = STS.band(f"ACS,WFC1,{bp_info[1]}")
                        elif "WFPC2" in cband:
                            bp_info = cband.split("_")
                            bp = STS.band(f"WFPC2,4,{bp_info[1]}")
                        elif "WFC3" in cband:
                            bp_info = cband.split("_")
                            if bp_info[1] in ["F110W", "F160W"]:
                                bp_cam = "IR"
                            else:
                                bp_cam = "UVIS1"
                            bp = STS.band(f"WFC3,{bp_cam},{bp_info[1]}")
                        else:
                            band_filename = f"John{cband}.dat"
                            bp = SpectralElement.from_file(
                                f"{band_resp_path}/{band_filename}"
                            )
                        self.band_resp[cband] = bp
                else:
                    # get the spectral data
                    self.fluxes[cspec][k, :] = moddata.data[cspec].fluxes
                    self.flux_uncs[cspec][k, :] = moddata.data[cspec].uncs

        if "BAND" in spectra_names:
            # add units
            self.waves["BAND"] = self.waves["BAND"] * u.micron

        # provide the width in model space for each parameter
        #   used in calculating the nearest neighbors
        self.n_nearest = 11

        self.temps_min = min(self.temps)
        self.temps_max = max(self.temps)
        self.temps_width2 = (self.temps_max - self.temps_min) ** 2
        if self.temps_width2 == 0.0:
            self.temps_width2 = 1.0

        self.gravs_min = min(self.gravs)
        self.gravs_max = max(self.gravs)
        self.gravs_width2 = (self.gravs_max - self.gravs_min) ** 2
        if self.gravs_width2 == 0.0:
            self.gravs_width2 = 1.0

        self.mets_min = min(self.mets)
        self.mets_max = max(self.mets)
        self.mets_width2 = (self.mets_max - self.mets_min) ** 2
        if self.mets_width2 == 0.0:
            self.mets_width2 = 1.0

    def SED_to_StarData(self, sed):
        """
        Convert the model created SED into a StarData object.
        Needed to plug into generating an ExtData object.

        Parameters
        ----------
        sed : object
            SED of each component
        """
        sd = StarData(None)

        for cspec in sed.keys():
            if cspec == "BAND":
                # populate the BAND info
                sd.data["BAND"] = BandData("BAND")
                sd.data["BAND"].fluxes = sed["BAND"] * (
                    u.erg / ((u.cm**2) * u.s * u.angstrom)
                )
                for k, cband in enumerate(self.band_names):
                    sd.data["BAND"].band_fluxes[cband] = (sed["BAND"][k], 0.0)
                sd.data["BAND"].get_band_mags_from_fluxes()

            else:
                # populate the spectral info
                sd.data[cspec] = SpecData(cspec)
                sd.data[cspec].fluxes = sed[cspec] * (
                    u.erg / ((u.cm**2) * u.s * u.angstrom)
                )

            sd.data[cspec].waves = self.waves[cspec]
            sd.data[cspec].n_waves = len(sd.data[cspec].waves)
            sd.data[cspec].uncs = 0.0 * sd.data[cspec].fluxes
            sd.data[cspec].npts = np.full((sd.data[cspec].n_waves), 1.0)
            sd.data[cspec].wave_range = [
                min(sd.data[cspec].waves),
                max(sd.data[cspec].waves),
            ]

        return sd
