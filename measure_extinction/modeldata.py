import numpy as np
import astropy.units as u

from dust_extinction.shapes import _curve_F99_method

from measure_extinction.stardata import StarData


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
    def __init__(self, modelfiles,
                 path='./',
                 band_names=['U', 'B', 'V', 'J', 'H', 'K'],
                 spectra_names=['STIS']):

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

        # photometric and spectroscopic data
        self.n_spectra = len(spectra_names) + 1
        self.spectra_names = ['BAND'] + spectra_names
        self.waves = {}
        self.fluxes = {}
        self.flux_uncs = {}

        for cspec in self.spectra_names:
            self.fluxes[cspec] = None
            self.flux_uncs[cspec] = None

        # initialize the BAND dictonary entry as the number of elements
        # is set by the desired bands, not the bands in the files
        self.waves['BAND'] = np.zeros((self.n_bands))
        self.fluxes['BAND'] = np.zeros((self.n_models, self.n_bands))
        self.flux_uncs['BAND'] = np.zeros((self.n_models, self.n_bands))

        # read and store the model data
        for k, cfile in enumerate(modelfiles):
            moddata = StarData(cfile, path=path)

            # model parameters
            self.temps[k] = np.log10(float(moddata.model_params['Teff']))
            self.gravs[k] = float(moddata.model_params['logg'])
            self.mets[k] = np.log10(float(moddata.model_params['Z']))
            self.vturb[k] = float(moddata.model_params['vturb'])

            # spectra
            for cspec in self.spectra_names:
                # initialize the spectra vectors
                if self.fluxes[cspec] is None:
                    self.waves[cspec] = moddata.data[cspec].waves
                    self.fluxes[cspec] = \
                        np.zeros((self.n_models,
                                  len(moddata.data[cspec].fluxes)))
                    self.flux_uncs[cspec] = \
                        np.zeros((self.n_models,
                                  len(moddata.data[cspec].fluxes)))

                # photometric bands
                if cspec == 'BAND':
                    for i, cband in enumerate(self.band_names):
                        band_flux = moddata.data['BAND'].get_band_flux(cband)
                        self.waves[cspec][i] = band_flux[2]
                        self.fluxes[cspec][k, i] = band_flux[0]
                        self.flux_uncs[cspec][k, i] = band_flux[1]
                else:
                    # get the spectral data
                    self.fluxes[cspec][k, :] = \
                        moddata.data[cspec].fluxes
                    self.flux_uncs[cspec][k, :] = \
                        moddata.data[cspec].uncs

        # provide the width in model space for each parameter
        #   used in calculating the nearest neighbors
        self.n_nearest = 11

        self.temps_min = min(self.temps)
        self.temps_max = max(self.temps)
        self.temps_width2 = (self.temps_max - self.temps_min)**2
        # self.temp_width2 = 1.0

        self.gravs_min = min(self.gravs)
        self.gravs_max = max(self.gravs)
        self.gravs_width2 = (self.gravs_max - self.gravs_min)**2

        self.mets_min = min(self.mets)
        self.mets_max = max(self.mets)
        self.mets_width2 = (self.mets_max - self.mets_min)**2
        # self.mets_width2 *= 4.0

    def get_stellar_sed(self,
                        params):
        """
        Compute the stellar SED given model parameters

        Parameters
        ----------
        params : float array
            stellar atmosphere parameters [logT, logg, logZ]

        Returns
        -------
        sed : dict
            stellar SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        # compute the distance between model params and grid points
        #    probably a better way using a kdtree
        dist2 = ((params[0] - self.temps)**2/self.temps_width2
                 + (params[1] - self.gravs)**2/self.gravs_width2
                 + (params[2] - self.mets)**2/self.mets_width2)
        sindxs = np.argsort(dist2)
        gsindxs = sindxs[0:self.n_nearest]

        # generate model SED form nearest neighbors
        #   should handle the case where dist2 has an element that is zero
        #   i.e., one of the precomputed models exactly matches the request
        weights = (1.0/np.sqrt(dist2[gsindxs]))
        weights /= np.sum(weights)

        sed = {}
        for cspec in self.fluxes.keys():
            # dot product does the multiplication and sum
            sed[cspec] = np.dot(weights,
                                self.fluxes[cspec][gsindxs, :])

        return sed

    def get_dust_extinguished_sed(self,
                                  params,
                                  sed):
        """
        Dust extinguished sed given the extinction parameters

        Parameters
        ----------
        params : float array
            dust extinction parameters [Av, Rv, c2, c3, c4, gamma, x0]

        sed : dict
            fluxes for each spectral piece

        Returns
        -------
        extinguished sed : dict
            stellar SED with {'bands': band_sed, 'spec': spec_sed, ...}
        """
        Rv = params[1]

        # updated F04 C1-C2 correlation
        C1 = 2.18 - 2.91*params[2]

        # spline points
        opt_axav_x = 10000./np.array([6000.0, 5470.0,
                                      4670.0, 4110.0])
        # **Use NIR spline x values in FM07, clipped to K band for now
        nir_axav_x = np.array([0.50, 0.75, 1.0])
        optnir_axav_x = np.concatenate([nir_axav_x, opt_axav_x])

        # **Keep optical spline points from F99:
        #    Final optical spline point has a leading "-1.208" in Table 4
        #    of F99, but that does not reproduce Table 3.
        #    Additional indication that this is not correct is from
        #    fm_unred.pro
        #    which is based on FMRCURVE.pro distributed by Fitzpatrick.
        opt_axebv_y = np.array([-0.426 + 1.0044*Rv,
                                -0.050 + 1.0016*Rv,
                                0.701 + 1.0016*Rv,
                                1.208 + 1.0032*Rv - 0.00033*(Rv**2)])
        # updated NIR curve from F04, note R dependence
        nir_axebv_y = (0.63*Rv - 0.84)*nir_axav_x**1.84

        optnir_axebv_y = np.concatenate([nir_axebv_y, opt_axebv_y])

        # create the extinguished sed
        ext_sed = {}
        for cspec in self.fluxes.keys():
            axav = _curve_F99_method(self.waves[cspec]*u.micron,
                                     Rv, C1, params[2], params[3], params[4],
                                     params[5], params[6],
                                     optnir_axav_x, optnir_axebv_y/Rv,
                                     [0.3, 10.0], 'F04_measure_extinction')
            ext_sed[cspec] = sed[cspec]*(10**(-0.4*axav*params[0]))

        return ext_sed