
=================
Code Capabilities
=================

The code is built around two classes.
StarData is for photometry and spectroscopy of a single star.
ExtData is for calculating and storing the extinction curve for a single
sightline.

For both classes, the data is stored by the source of origin of the data.  Possible
origins are band photometry ("BAND") or spectroscopy from the
International Ultraviolet Explorer ("IUE"),
Far-Ultraviolet Spectroscopic Explorer ("FUSE"),
Hubble Space Telescope Imaging Spectrograph ("STIS"),
NASA's Infrared Telescope SpeX Instrument ("SpeX_SXD", "SpeX_LXD"),
and Spitzer Infrared Spectrograph ("IRS")
The strings in "" give the dictionary key.

This package assumes the spectral data from a specific source is on the
same wavelength grid for all stars/sightlines.
This simplifies the extinction calculations.
Code to do this for some data sources can be found in
`measure_extinction.merge_obsspec`.

StarData
========

The data stored as part of `measure_extinction.stardata.StarData`
is the photometry and spectroscopy for a
single star, each dataset stored in a class itself.
The photometry is stored in the dedicated BandData class to
provide specific capabilities needed for handling photometric data.
The spectroscopy is stored in SpecData class.

The details of the data stored include:

* file: name of DAT file
* path: path of DAT file
* data: dictionary of BandData and SpecData objects containing the stellar data
* photonly: only photometry used (all spectroscopic data ignored)
* sptype: spectral type of star from DAT file
* use_corfac: boolean for determining if corfacs should be used
* corfac: dictionary of correction factors for spectral data
* dereddened: boolean if the data has been dereddened
* dereddenParams: dictionary of FM90 and CCM89 dereddening parameters
* model_parameters: stellar atmosphere model parameters (for model based data)

Member Functions
----------------

The member functions of StarData include:

* read: read in the data on a star based on a DAT formatted file
* deredden: deredden the data based on FM90 (UV) and CCM89 (optical/NIR)
* get_flat_data_arrays: get waves, fluxes, uncs as vectors combining all the data
* plot: plot the stellar data given a matplotlib plot object

BandData
========

Member Functions
----------------

SpecData
========

A spectrum is stored as part of `measure_extinction.stardata.SpecData`.
The details of the data stored are:

* waves: wavelengths with units
* n_waves: number of wavelengths
* wave_range: min/max of waves with units
* fluxes: fluxes versus wavelengths with units
* uncs: flux uncertainties versus wavelength with units
* npts: number of measurements versus wavelength

Member Functions
----------------

The member functions of SpecData include:

* read_spectra: generic read for a spectrum in a specifically formatted FITS file
* read_fuse: read a FUSE spectrum
* read_iue: read an IUE spectrum (includes cutting data > 3200 A)
* read_stis: read a Hubble/STIS spectrum (either UV or optical range)
* read_spex: read a IRTF/SpeX spectrum (includes scaling by corfacs)
* read_irs: read a Spitzer/IRS spectrum (includes scaling by corfacs, cutting above some wavelength)
* read_nircam_ss: read in NIRCam/slitless spectrum
* read_miri_lrs: read in Webb/MIRI LRS spectrum
* read_miri_ifu: read in Webb/MIRI MRS spectrum
* rebin_constres: rebin spectrum to a constant input resolution

ExtData
=======

The data that is stored as part of `measure_extinction.extdata.ExtData`
is the extinction curve from each source and associated details.
Specific details stored include (where src = data source),

* type: type of extinction measurement (e.g., elx, elxebv, alax)
* type_rel_band: photometric band that is used for the relative extinction measurement (usually V band)
* waves[src]: wavelengths
* exts[src]: extinction versus wavelength
* uncs[src]: uncertainty in the extinction versus wavelength
* npts[src]: number of measurements at each wavelength
* names[src]: names of the photometric bands (only present for names["BAND"])
* columns: dictionary of gas or dust column measurements (e.g., A(V), R(V), N(HI))
* red_file: filename of the reddened star
* comp_file: filename of the comparison star

Member Functions
----------------

The member functions of ExtData include:

* calc_elx: calculate E(lambda-X) for all data sources present in both reddened and comparison StarData objects
* calc_elx_bands: calculate E(lambda-X) for the "BAND" data
* calc_elx_spectra: calculate E(lambda-X) for a single spectral data source
* calc_EBV: determine E(B-V) based on E(lambda-V) "BAND" data
* calc_AV: determine A(V) based on E(lambda-V) "BAND" data
* calc_RV: determine R(V) using calc_EBV and calc_AV as needed
* trans_elv_elvebv: calculate E(lambda-V)/E(B-V) based on E(lambda-V) and E(B-V)
* trans_elv_alav: calculate A(lambda)/A(V) based on E(lambda-V) and A(V)
* rebin_constres: rebin one data source extinction data to a constant input resolution
* get_fitdata: gets a tuple of vectors with (wavelengths, exts, uncs) that is useful for fitting
* save: save the extinction curve to a FITS file
* read: read an extinction curve from a FITS file
* plot: plot the extinction curve given a matplotlib plot object

.. note::
   ExtData partially supports extinction curves relative to an arbitrary band.
   Some member functions only support extinction curve relative to V band.
   Work continues to update the code to allow arbitrary bands for all functions.
