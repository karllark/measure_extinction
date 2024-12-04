.. _data_formats:

############
Data Formats
############

The data for each stars is stored in a combination of an ASCII "DAT" file
and FITS "spectra" files.

DAT file
========

The ASCII file is the only required file.  This file gives the photometry
and links to the FITS files that contain the spectroscopy.  Many dat files
for stars used in literature dust extinction curves are provided in the 
`extstar_data repository <https://github.com/karllark/extstar_data>_`.

An example of
such a file (for a star from Valencic et al. 2004) that gives the Johnson photometry, 
location of IUE spectrum, and spectral types is below.  This includes the
photometry and associated uncertainties in Vega magnitudes.  Comments taking
an entire line are allowed if they start with a "#" character.

::

    # data file for observations of HD 38023
    V = 8.870 +/- 0.020
    B = 9.190 +/- 0.030
    U = 8.870 +/- 0.030
    J = 8.080 +/- 0.020
    H = 7.990 +/- 0.060
    K = 7.900 +/- 0.020
    IUE = IUE_Data/hd038023_iue.fits
    sptype = B4V
    uvsptype = B4IV

A more complicated example giving photometry and spectroscopy from different sources
is below (from a star from Gordon et al. 2021; Decleir et al. 2022 among other studies).
The Johnson photometry is provide mainly as colors in this example.  Photometry
rom IRAC/IRS/MIPS is provide in mJy.  The different photometry units are converted to the
measure_extinction internal units when read into StarData.  In addition, there are comments
provide after data using the ";" character.

::

    # data file for observations of HD 29647
    V = 8.31 +/- 0.017 ; UBVRI ref = Slutskij, Stalbovskij & Shevchenko 1980 (1980SvAL....6..397S)
    (U-B) = 0.47 +/- 0.030
    (B-V) = 0.91 +/- 0.019
    (V-R) = 0.96 +/- 0.024
    (V-I) = 1.71 +/- 0.030
    J = 5.960 +/- 0.030 ; JHK ref = IRSA 2MASS All-Sky Point Source Catalog
    H = 5.593 +/- 0.021
    K = 5.363 +/- 0.020
    # IRAC 1 and 2 seem off
    IRAC1 = 2.23E+03 +/- 4.45E+01 mJy
    IRAC2 = 1.50E+03 +/- 3.00E+01 mJy
    IRAC3 = 1.06E+03 +/- 2.16E+01 mJy
    IRAC4 = 5.42E+02 +/- 1.09E+01 mJy
    IRS15 = 3.14E+02 +/- 6.51E+00 mJy
    MIPS24 = 6.76E+01 +/- 1.89E+00 mJy
    # WISE have extended source contamination, contaminated by scattered moonlight
    # WISE 1, 2 and 3 spurious detection of scattered light halo
    # WISE 1 and 2 seem off, have saturated pixels
    WISE1 = 5.179 +/- 0.150 ; WISE ref = IRSA AllWISE Source Catalog
    WISE2 = 4.905 +/- 0.068
    WISE3 = 4.882 +/- 0.023
    WISE4 = 3.821 +/- 0.033
    IUE = IUE_Data/hd029647_iue.fits
    SpeX_SXD = SpeX_Data/hd029647_SXD_spex.fits
    SpeX_LXD = SpeX_Data/hd029647_LXD_spex.fits
    IRS = IRS_Data/hd029647_irs.fits
    sptype = B7IV ; sptype ref = Murakawa, Tamura & Nagata 2000 (2000ApJS..128..603M)
    uvsptype = B8III ; uvsptype ref = Valencic+2004 (2004ApJ...616..912V)
   

Spectra
=======

The spectra are stored in FITS files with standard columns and wavelength grids.
The standard columns are "WAVELENGTH", "FLUX", "SIGMA", and "NPTS".  These files
are created using the "merge_xxx_spec" functions provide in
:class:`~measure_extinction.merge_obsspec`.  These ensure that all the spectra
from a specific source (e.g., IUE) have the same wavelength grid, units, and standard
columns.

For some of the types of spectra (e.g., IUE, STIS, SpeX, NIRCam SS), 
there are commandline code in the `measure_extinction/utils` subdirectory to
create such files from the commandline.

When using a stellar model for the comparison, each type of spectra supported is 
simulated/mocked using code in the `measure_extinction/utils` subdirectory. 
See :ref:`model standards`.
