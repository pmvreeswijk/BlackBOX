
import os
import argparse
import re
from datetime import datetime
from dateutil.tz import gettz
import itertools
from random import choice
from string import ascii_uppercase
import glob
import sys

# set up log
import logging
import time
logfmt = ('%(asctime)s.%(msecs)03d [%(levelname)s, %(process)s] %(message)s '
          '[%(funcName)s, line %(lineno)d]')
datefmt = '%Y-%m-%dT%H:%M:%S'
logging.basicConfig(level='INFO', format=logfmt, datefmt=datefmt)
logFormatter = logging.Formatter(logfmt, datefmt)
logging.Formatter.converter = time.gmtime #convert time in logger to UTC
log = logging.getLogger()

from zogy import find_stars, haversine, get_par, read_hdulist, get_psfoptflux
from zogy import get_airmass, apply_zp, mem_use, get_Xchan_bool, mini2back
from zogy import get_index_around_xy, get_matches
from blackbox import get_path, pool_func
import set_zogy
set_zogy.verbose=False
import set_blackbox as set_bb

import numpy as np

import astropy.io.fits as fits
from astropy.coordinates import Angle, SkyOffsetFrame, SkyCoord
from astropy.table import Table, hstack, vstack
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS

from fitsio import FITS

MLBG_fields = Table.read(set_bb.mlbg_fieldIDs)

cols_fullsource = ['NUMBER', 'X_POS', 'Y_POS', 'XVAR_POS', 'YVAR_POS',
                   'XYCOV_POS', 'RA', 'DEC', 'ELONGATION', 'FWHM', 'CLASS_STAR', 
                   'FLAGS', 'FLAGS_MASK', 'BACKGROUND',
                   'MAG_APER_R0.66xFWHM', 'MAGERR_APER_R0.66xFWHM',
                   'MAG_APER_R1.5xFWHM', 'MAGERR_APER_R1.5xFWHM',
                   'MAG_APER_R5xFWHM', 'MAGERR_APER_R5xFWHM',
                   'E_FLUX_OPT', 'E_FLUXERR_OPT', 'MAG_OPT', 'MAGERR_OPT']

cols_trans = ['NUMBER', 'X_PEAK', 'Y_PEAK',
              'RA_PEAK', 'DEC_PEAK', 'SNR_ZOGY',
              'E_FLUX_ZOGY', 'E_FLUXERR_ZOGY', 'MAG_ZOGY', 'MAGERR_ZOGY',
              'X_POS_SCORR', 'Y_POS_SCORR',
              'RA_SCORR', 'DEC_SCORR', 'ELONG_SCORR',
              'FLAGS_SCORR', 'FLAGS_MASK_SCORR',
              'X_PSF_D', 'XERR_PSF_D', 'Y_PSF_D', 'YERR_PSF_D',
              'RA_PSF_D', 'DEC_PSF_D', 'MAG_PSF_D', 'MAGERR_PSF_D', 
              'CHI2_PSF_D',
              'X_GAUSS_D', 'XERR_GAUSS_D', 'Y_GAUSS_D', 'YERR_GAUSS_D',
              'RA_GAUSS_D', 'DEC_GAUSS_D', 
              'FWHM_GAUSS_D', 'ELONG_GAUSS_D', 'CHI2_GAUSS_D', 'CLASS_REAL',
              'THUMBNAIL_RED', 'THUMBNAIL_REF', 'THUMBNAIL_D', 'THUMBNAIL_SCORR']

cols_ref = ['NUMBER', 'X_POS', 'Y_POS',
            'XVAR_POS', 'YVAR_POS', 'XYCOV_POS', 
            'RA', 'DEC',
            'CXX', 'CYY', 'CXY', 'A', 'B', 'THETA',
            'ELONGATION', 'FWHM', 'CLASS_STAR',
            'FLAGS', 'FLAGS_MASK',
            'BACKGROUND',
            'MAG_APER', 'MAGERR_APER',  
            'MAG_AUTO', 'MAGERR_AUTO', 'KRON_RADIUS',
            'MAG_ISO', 'MAGERR_ISO', 'ISOAREA',
            'MU_MAX', 'FLUX_RADIUS',
            'MAG_PETRO', 'MAGERR_PETRO', 'PETRO_RADIUS',
            'E_FLUX_OPT', 'E_FLUXERR_OPT', 'MAG_OPT', 'MAGERR_OPT']
# add _REF to the [cols_ref] column names to avoid duplicating
# full-source columns
for i in range(len(cols_ref)):
    cols_ref[i] = '{}_REF'.format(cols_ref[i])


#try:
#    from fitsio import FITS
#except:
#    log.info ('import of fitsio.FITS failed; using astropy.io.fits instead')

__version__ = '0.8'


################################################################################

def force_phot (radec_images_dict, trans=True, ref=True, fullsource=False,
                nsigma=5, use_catalog_mags=False, sep_max=3, catcols2add=None,
                catcols2add_dtypes=None, keys2add=None, keys2add_dtypes=None,
                bkg_global=True, ncpus=1):

    """Forced photometry on MeerLICHT/BlackGEM images at the input
       coordinates provided, or alternatively, the extraction of
       full-source and/or transient catalog magnitudes near the input
       coordinates. The results are returned in a single astropy
       Table.

       Based on an input dictionary [radec_images_dict] with (RA,DEC)
       coordinate tuples as keys and lists of MeerLICHT or BlackGEM
       reduced image basenames as values, this function performs
       forced photometry on the list of images at the corresponding
       coordinates (RA,DEC). Depending on whether [trans] and/or
       [fullsource] is set to True, this is done for the transient
       and/or full-source catalogs.

       In the transient case, the transient magnitude (MAG_ZOGY) and
       corresponding error (MAGERR_ZOGY), its signal-to-noise ratio
       (SNR_ZOGY) and the [nsigma]-sigma transient limiting magnitude
       (TRANS_LIMMAG_[nsigma]SIGMA) are determined at the input
       coordinates and added as columns to the output table.

       In the fullsource case, the optimal magnitude (MAG_OPT) and
       corresponding error (MAGERR_OPT), its signal-to-noise ratio
       (SNR_OPT) and the [nsigma]-sigma limiting magnitude
       (TRANS_LIMMAG_[nsigma]SIGMA) are determined at the input
       coordinates and added as columns to the output table.

       If [use_catalog_mags] is True, a source in the transient or
       full-source catalog corresponding to the input image is
       searched for within [sep_max] arcseconds from the input
       position; if a match is found, the magnitude, error,
       signal-to-noise ratio and limiting magnitude values are
       replaced by the corresponding quantities of the catalog source.

       Any header keyword listed in [keys2add] with the corresponding
       data type listed in [keys2add_dtypes] will be added as an
       output column, where the values are searched for in the input
       image headers.

       To speed up the processing, [ncpus] can be increased to the
       number of CPUs available.

   
    Parameters:
    -----------

    radec_image_dict: dictionary (no default) with (RA,DEC) coordinate
                      tuples as keys and lists of MeerLICHT or
                      BlackGEM reduced image basenames as values; RA
                      and DEC need to be in decimal degrees and the
                      image basenames are as follows: [full
                      path]/[tel]_yyyymmdd_hhmmss, e.g.
                      ['/idia/projects/meerlicht/ML1/red/2022/01/21/
                      ML1_20220120_225746']

    trans: boolean (default=True), if True the transient magnitudes
           and limits will be extracted
    
    ref: boolean (default=True), if True the reference magnitudes and
         limits will be extracted

    fullsource: boolean (default=False), if True the full-source
                magnitudes and limits will be extracted

    nsigma: float (default=5), the significance level at which the
            limiting magnitudes will be determined; this values will
            be indicated in the relevant output table column names

    use_catalog_mags: boolean (default=False); if True, a search is
                      done in the existing full-source or transient
                      catalog for an object within [sep_max]
                      arcseconds of the input position; if an object
                      is found, its magnitudes, signal-to-noise ratio
                      and limiting magnitude at its position is
                      reported in the output table instead of the
                      values at the input RA,DEC.

    sep_max: float (default=3 arcseconds); maximum separation in
             arcseconds between the input RA,DEC and a catalog source

    catcols2add: list of strings (default=None); additional
                 full-source, transient and/or reference catalog columns 
                 to add to the output table; add \'_REF\' to any reference 
                 column name, to separate them from the often-identical
                 full-source column names

    catcols2add_dtypes: list of dtypes (default=None); corresponding
                        dtypes of the catalog columns provided in
                        [catcols2add]

    keys2add: list of strings (default=None); header keywords that
              will be added as columns to the output table

    keys2add_dtypes: list of dtypes (default=None); corresponding
                     dtypes of the header keywords provided in
                     [keys2add]
                     
    ncpus: int (default=1); number of processes/tasks to use

    """
    

    # no point continuing if input [trans] and [fullsource] are both
    # set to False
    if not trans and not fullsource:
        log.error ('input parameters [trans] and [fullsource] are both set to '
                   'False; no data will be extracted')
        return None

    
    # to add keys or not
    if keys2add is not None and keys2add_dtypes is not None:
        add_keys = True
    else:
        add_keys = False


    # initialize output table with several columns
    names = ['RA_IN', 'DEC_IN', 'FILENAME', 'X_POS_IN', 'Y_POS_IN']
    dtypes = [float, float, 'U30', float, float]


    # initialize keyword columns
    if add_keys:
        for ikey, key in enumerate(keys2add):
            names.append(key)
            dtype_str = keys2add_dtypes[ikey]
            if dtype_str in ['float', 'int', 'bool']:
                dtype = eval(dtype_str)
            else:
                dtype = dtype_str

            dtypes.append(dtype)

            # if QC-FLAG is a requested keyword, while TQC-FLAG is
            # not, then add the latter if trans is True
            if key == 'QC-FLAG' and 'TQC-FLAG' not in keys2add and trans:
                names += ['TQC-FLAG']
                dtypes += ['U6']


    # add FLAGS_MASK, which is determined irrespective of a match with
    # a full-source catalog source
    if 'FLAGS_MASK' not in names:
        names += ['FLAGS_MASK']
        dtypes += ['uint8']

        
    # initialize columns to be determined below
    if fullsource:

        names_fullsource = ['MAG_OPT', 'MAGERR_OPT', 'SNR_OPT',
                            'LIMMAG_{}SIGMA'.format(nsigma)]
        names += names_fullsource
        dtypes += [float, float, float, float]

        # add separation between input coordinates and potential
        # matching source in full-source catalog
        if use_catalog_mags:
            names += ['SEP_FULLSOURCE']
            dtypes += [float]

        # add additional full-source catalog columns specified in
        # [args.catcols2add]
        if use_catalog_mags and catcols2add is not None:
            for icol, col in enumerate(catcols2add):
                if col in cols_fullsource and col not in names:
                    names += [col]
                    dtypes += [catcols2add_dtypes[icol]]


    if trans:

        names_trans = ['MAG_ZOGY', 'MAGERR_ZOGY', 'SNR_ZOGY',
                       'TRANS_LIMMAG_{}SIGMA'.format(nsigma)]
        names += names_trans
        dtypes += [float, float, float, float]

        # add separation between input coordinates and potential
        # matching source in transient catalog
        if use_catalog_mags:
            names += ['SEP_TRANS']
            dtypes += [float]

        # add additional transient catalog columns specified in
        # [args.catcols2add]
        if use_catalog_mags and catcols2add is not None:
            for icol, col in enumerate(catcols2add):
                if col in cols_trans and col not in names:
                    names += [col]
                    dtypes += [catcols2add_dtypes[icol]]


    if ref:

        # add pixelcoordinates corresponding to input RA/DEC to table
        names += ['X_POS_IN_REF', 'Y_POS_IN_REF']
        dtypes += [float, float]
        

        # add FLAGS_MASK for the reference image
        if 'FLAGS_MASK_REF' not in names:
            names += ['FLAGS_MASK_REF']
            dtypes += ['uint8']


        # magnitude, snr and limiting magnitude columns
        names_ref = ['MAG_OPT_REF', 'MAGERR_OPT_REF', 'SNR_OPT_REF',
                     'LIMMAG_{}SIGMA_REF'.format(nsigma),
                     'MAG_ZOGY_PLUSREF']
        names += names_ref
        dtypes += [float, float, float, float, float]


        # add separation between input coordinates and potential
        # matching source in full-source catalog
        if use_catalog_mags:
            names += ['SEP_REF']
            dtypes += [float]


        # add additional full-source catalog columns specified in
        # [args.catcols2add]
        if use_catalog_mags and catcols2add is not None:
            for icol, col in enumerate(catcols2add):
                if col in cols_ref and col not in names:
                    names += [col]
                    dtypes += [catcols2add_dtypes[icol]]

                    
    # convert input dictionary {(RA1,DEC1): imagelist1, (RA1,DEC1):
    # imagelist2, ...}  to {image1: RADEClist, image2: RADEClist2,
    # ...}, so that if there are multiple sources in the same image,
    # it only needs to be read in once, and it will be faster
    log.info ('converting dictionary from '
              '{(RA1,DEC1): imagelist1, (RA1,DEC1): imagelist2, ...} '
              'to {image1: RADEClist, image2: RADEClist2, ...}')
    image_radecs_dict = {}
    for k, vlist in radec_images_dict.items():
        for v in vlist:
            image_radecs_dict[v] = image_radecs_dict.get(v, []) + [k]


    # convert [image_radecs_dict] to a list so it is more easily
    # processed by pool_func
    log.info ('converting image_radecs_dict to a list')
    image_radecs_list = []
    for k, v in image_radecs_dict.items():
        image_radecs_list.append([k,v])


    # pick maximum number (***CHECK!!!***)
    #image_radecs_list = image_radecs_list[0:100]


    # check if there are any images to process at all
    nimages = len(image_radecs_list)
    log.info ('effective number of images from which to extract magnitudes: {}'
              .format(nimages))
    if nimages == 0:
        log.critical ('no images could be found matching the input coordinates ')


    # use pool_func and function [get_rows] to multi-process list of
    # basenames
    rows = pool_func (get_rows, image_radecs_list, trans, ref, fullsource,
                      nsigma, use_catalog_mags, sep_max, catcols2add, keys2add,
                      add_keys, names, dtypes, bkg_global, nproc=ncpus)
    rows = list(itertools.chain.from_iterable(rows))


    # remove None entries, e.g. due to coordinates off the field
    while True:
        try:
            rows.pop(rows.index(None))
        except:
            break

    # finished multi-processing basenames
    if len(rows) > 0:
        # add rows to table
        table = Table(rows=rows, names=names, dtype=dtypes)
    else:
        return None


    # sort in time
    index_sort = np.argsort(table['FILENAME'])

    mem_use('at end of [force_phot]')

    # return table
    return table[index_sort]


################################################################################

def remove_empty (list_in):

    while True:
        try:
            i = list_in.index('')
            log.warning ('removing empty string from list')
            list_in.pop(i)
        except:
            break


################################################################################

def get_rows (image_radecs, trans, ref, fullsource, nsigma, use_catalog_mags,
              sep_max, catcols2add, keys2add, add_keys, names, dtypes,
              bkg_global):


    # extract basenames and coordinates from input tuple [image_radecs]
    basename, radecs = image_radecs
    ras_deg, decs_deg = zip(*radecs)
    ras_deg = np.array(ras_deg)
    decs_deg = np.array(decs_deg)


    # infer telescope name from basename
    tel = basename.split('/')[-1][0:3]


    # read header; data is read further below when necessary
    fits_red = '{}_red.fits.fz'.format(basename)
    fits_cat = '{}_red_cat.fits'.format(basename)
    fits_trans = '{}_red_trans.fits'.format(basename)
    # try to read transient catalog header, as it is more complete
    # than the full-source catalog header
    if os.path.exists(fits_trans):
        fits_2read = fits_trans
    elif os.path.exists(fits_cat):
        fits_2read = fits_cat
    elif os.path.exists(fits_red):
        fits_2read = fits_red
    else:
        log.warning ('reduced image, full-source and transient catalog all '
                     'do not exist for {}; skipping its extraction'
                     .format(basename))
        return [None]


    # read header
    #log.info ('reading header of {}'.format(fits_2read))
    #header = read_hdulist (fits_2read, get_data=False, get_header=True)
    header = FITS(fits_2read)[-1].read_header()


    # read FWHM from the header
    if 'PSF-FWHM' in header:
        fwhm = header['PSF-FWHM']
    elif 'S-FWHM' in header:
        fwhm = header['S-FWHM']
    else:
        fwhm = 5
        log.warning ('keywords PSF-FWHM nor S-FWHM present in the header '
                     'for {}; assuming fwhm=5 pix'.format(basename))


    # data_shape
    data_shape = get_par(set_zogy.shape_new,tel)
    ysize, xsize = data_shape


    # convert input [ras_deg] and [decs_deg] to pixel coordinates
    xcoords, ycoords = WCS(header).all_world2pix(ras_deg, decs_deg, 1)
    
    # indices of pixel coordinates
    x_indices = (xcoords-0.5).astype(int)
    y_indices = (ycoords-0.5).astype(int)


    # make sure xcoords and ycoords are valid numbers and on the image
    mask_finite = (np.isfinite(xcoords) & np.isfinite(ycoords))
    if np.sum(mask_finite)==0:
        log.warning ('inferred pixel coordinates are not finite numbers '
                     'for {}; skipping its extraction'.format(basename))
        log.warning ('xcoords: {}'.format(xcoords))
        log.warning ('ycoords: {}'.format(ycoords))
        return [None]


    # and that at least one them is on the image
    dpix_edge = 10
    mask_on = ((xcoords > dpix_edge) & (xcoords < xsize-dpix_edge) &
               (ycoords > dpix_edge) & (ycoords < ysize-dpix_edge))
    if np.sum(mask_on)==0:
        log.warning ('inferred pixel coordinates are off the image for {}; '
                     'skipping its extraction'.format(basename))
        return [None]


    # take out entries that were not finite or off the image
    mask_finite_on = mask_finite & mask_on
    if np.sum(~mask_finite_on) != 0:
        xcoords = xcoords[mask_finite_on]
        ycoords = ycoords[mask_finite_on]
        ras_deg = ras_deg[mask_finite_on]
        decs_deg = decs_deg[mask_finite_on]


    # create a zero-valued table with shape ncoords x number of column
    # names, with the names and dtypes set by the corresponding input
    # parameters
    ncoords = len(xcoords)
    table = Table(np.zeros((ncoords,len(names))), names=names, dtype=dtypes)
    colnames = table.colnames

    # start to fill in table
    table['RA_IN'] = ras_deg
    table['DEC_IN'] = decs_deg
    table['FILENAME'] = '{}_red'.format(basename.split('/')[-1])
    table['X_POS_IN'] = xcoords
    table['Y_POS_IN'] = ycoords

    
    # add header keywords to output table
    if add_keys:
        for key in keys2add:

            try:
                table[key] = header[key]
            except:
                table[key] = None
                log.warning ('key {} not in header of {}'
                             .format(key, basename))

            if key=='QC-FLAG' and 'TQC-FLAG' not in keys2add and trans:

                try:
                    table['TQC-FLAG'] = header['TQC-FLAG']
                except:
                    table['TQC-FLAG'] = None
                    log.warning ('TQC-FLAG not in header of {}'
                                 .format(basename))


    # read data mask here; is needed in case of fullsource and/or trans
    fits_mask = '{}_mask.fits.fz'.format(basename)
    #data_mask = read_hdulist (fits_mask, dtype='uint8')
    data_mask = FITS(fits_mask)[-1]

    # add combined FLAGS_MASK column to output table using
    # [get_flags_mask_comb]
    table['FLAGS_MASK'] = get_flags_mask_comb (
        data_mask, xcoords, ycoords, fwhm, xsize, ysize)


    # infer the image zeropoint
    keys = ['EXPTIME', 'FILTER', 'DATE-OBS']
    exptime, filt, obsdate = [header[key] for key in keys]
    # get zeropoint from [header]
    if 'PC-ZP' in header:
        zp = header['PC-ZP']
    else:
        log.warning ('keyword PC-ZP not in header of {}_red; skipping it'
                     .format(basename))
        return [list(table[i]) for i in range(len(table))]

    
    # determine object airmass, unless input image is a combined
    # image
    if 'R-V' in header or 'R-COMB-M' in header:
        airmass = 1.0
    else:
        lat = get_par(set_zogy.obs_lat,tel)
        lon = get_par(set_zogy.obs_lon,tel)
        height = get_par(set_zogy.obs_height,tel)
        airmass = get_airmass(table['RA_IN'], table['DEC_IN'], obsdate,
                              lat, lon, height)

    # extinction coefficient
    ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]       
    


    # full-source; determining optimal flux
    # -------------------------------------
    nsigma_orig = 5
    if fullsource:              

        # define PSFEx binary fits table to use
        psfex_bintable = '{}_red_psf.fits'.format(basename)

        if not os.path.exists(psfex_bintable):
            log.warning ('{} not found; skipping extraction of magnitudes '
                         'on {}'.format(psfex_bintable, basename))
            return [list(table[i]) for i in range(len(table))]

        
        # background STD
        fits_bkg_std = '{}_red_bkg_std.fits.fz'.format(basename)
        if os.path.exists(fits_bkg_std):
            #data_bkg_std = read_hdulist (fits_bkg_std, dtype='float32')
            data_bkg_std = FITS(fits_bkg_std)[-1]
        else:
            # if it does not exist, create it from the background mesh
            fits_bkg_std_mini = '{}_red_bkg_std_mini.fits'.format(basename)
            data_bkg_std_mini, header_mini = read_hdulist (
                fits_bkg_std_mini, get_header=True, dtype='float32')

            if 'BKG-SIZE' in header_mini:
                bkg_size = header_mini['BKG-SIZE']
            else:
                bkg_size = get_par(set_zogy.bkg_boxsize,tel)


            if ncoords == 1:
                # determine scalar bkg_std value from mini image at
                # xcoord, ycoord
                x_indices_mini = ((xcoords-0.5).astype(int)/bkg_size).astype(int)
                y_indices_mini = ((ycoords-0.5).astype(int)/bkg_size).astype(int)
                [data_bkg_std] = data_bkg_std_mini[y_indices_mini, x_indices_mini]
                
            else:
                # determine full bkg_std image from mini image
                
                # determine whether interpolation is allowed across different
                # channels in [mini2back] using function get_Xchan_bool
                chancorr = get_par(set_zogy.MLBG_chancorr,tel)
                interp_Xchan_std = get_Xchan_bool (tel, chancorr, 'new',
                                                   std=True)
                data_bkg_std = mini2back (
                    data_bkg_std_mini, data_shape, order_interp=1,
                    bkg_boxsize=bkg_size, interp_Xchan=interp_Xchan_std,
                    timing=get_par(set_zogy.timing,tel))



        # object mask - not part of the standard zogy products at the
        # moment, so not available (yet); for the time being, let this
        # depend on input parameter [bkg_global]
        if bkg_global:
            # if True, global background is used, i.e. any local flux
            # due to nearby sources or galaxy is not taken into
            # account
            objmask = np.ones (data_shape, dtype=bool)
        else:
            # if False, a circular annulus around each object is used
            # to estimate the sky background
            objmask = np.zeros (data_shape, dtype=bool)


        # read reduced image; need to use astropy method, as otherwise
        # this will lead to an exception in [get_psfoptflux] as
        # (probably) the shape attribute is not available when data is
        # read through fitsio.FITS
        data = read_hdulist (fits_red)
        
        try:
            # determine optimal fluxes at pixel coordinates
            flux_opt, fluxerr_opt = get_psfoptflux (
                psfex_bintable, data, data_bkg_std**2, data_mask, xcoords, ycoords,
                imtype='new', fwhm=fwhm, D_objmask=objmask, set_zogy=set_zogy,
                tel=tel)

        except Exception as e:
            log.error ('exception was raised while executing [get_psfoptflux]; '
                       'skipping extraction of magnitudes from {}: {}'
                       .format(basename, e))
            return [list(table[i]) for i in range(len(table))]


        # infer calibrated magnitudes using the zeropoint
        mag_opt, magerr_opt = apply_zp (np.abs(flux_opt), zp, airmass, exptime,
                                        filt, ext_coeff, fluxerr=fluxerr_opt)


        # read limiting magnitude at pixel coordinates
        fits_limmag = '{}_red_limmag.fits.fz'.format(basename)
        if os.path.exists(fits_limmag):

            #data_limmag = read_hdulist(fits_limmag)
            #limmags = data_limmag[y_indices, x_indices]
            data_limmag = FITS(fits_limmag)[-1]
            limmags = get_fitsio_values (data_limmag, y_indices, x_indices)

            # convert limmag from number of sigma listed
            # in the image header to input [nsigma]
            if ('NSIGMA' in header and
                isinstance(header['NSIGMA'], (float, int)) and
                header['NSIGMA'] != 0):
                nsigma_orig = header['NSIGMA']
                    
            if nsigma_orig != nsigma:
                limmags += -2.5*np.log10(nsigma/nsigma_orig)

        else:
            log.warning ('{} not found; no limiting magnitudes available'
                         .format(fits_limmag))
            limmags = None


        # check if [use_catalog_mags] is True
        if use_catalog_mags:

            # look for nearby source in full-source catalog
            fits_cat = '{}_red_cat.fits'.format(basename)
            table_cat = Table.read(fits_cat)
            colnames_cat = table_cat.colnames

            if len(table_cat) > 0:

                # find matches between pixel coordinates (xcoords,
                # ycoords) and sources in full-source catalog;
                # return their respective indices and separation
                i_coords, i_cat, sep, __, __ = get_matches (
                    table['RA_IN'].quantity.value,
                    table['DEC_IN'].quantity.value,
                    table_cat['RA'].quantity.value,
                    table_cat['DEC'].quantity.value,
                    dist_max=sep_max, return_offsets=True)


                mag_opt[i_coords] = np.array(table_cat['MAG_OPT'][i_cat])
                magerr_opt[i_coords] = np.array(table_cat['MAGERR_OPT'][i_cat])
                flux_opt[i_coords] = np.array(table_cat['E_FLUX_OPT'][i_cat])
                fluxerr_opt[i_coords] = np.array(table_cat['E_FLUXERR_OPT'][i_cat])


                # update existing FLAGS_MASK column
                table['FLAGS_MASK'][i_coords] = np.array(
                    table_cat['FLAGS_MASK'][i_cat])


                # add separation
                table['SEP_FULLSOURCE'][i_coords] = sep


                # add additional catalog columns if present
                for col in catcols2add:
                    if col in colnames_cat and col in colnames:
                        table[col][i_coords] = table_cat[col][i_cat]



        # calculate signal-to-noise ratio; applies to either the SNR
        # of the limit or the matched source in the catalog
        snr_opt = np.zeros(flux_opt.shape)
        mask_nonzero = (fluxerr_opt != 0)
        snr_opt[mask_nonzero] = flux_opt[mask_nonzero] / fluxerr_opt[mask_nonzero]


        # update table
        table['MAG_OPT'] = mag_opt
        table['MAGERR_OPT'] = magerr_opt
        table['SNR_OPT'] = snr_opt
        table['LIMMAG_{}SIGMA'.format(nsigma)] = limmags



    # transient; extracting ZOGY fluxes
    # ---------------------------------
    nsigma_trans_orig = 6
    if trans:

        # read PSF flux image
        fits_Fpsf = '{}_red_Fpsf.fits.fz'.format(basename)
        if not os.path.exists(fits_Fpsf):

            log.warning ('{} not found; skipping extraction of transient '
                         'magnitude'.format(fits_Fpsf))
            return [list(table[i]) for i in range(len(table))]


        # read flux values at xcoords, ycoords
        #data_Fpsf = read_hdulist (fits_Fpsf, dtype='float32')
        #Fpsf = data_Fpsf[y_indices, x_indices]
        data_Fpsf = FITS(fits_Fpsf)[-1]
        Fpsf = get_fitsio_values (data_Fpsf, y_indices, x_indices)


        # get transient limiting magnitude at xcoord, ycoord
        # and convert it back to Fpsferr

        # read limiting magnitude at pixel coordinates
        fits_tlimmag = '{}_red_trans_limmag.fits.fz'.format(basename)
        if os.path.exists(fits_tlimmag):
            #data_tlimmag = read_hdulist (fits_tlimmag)
            #tlimmags = data_tlimmag[y_indices, x_indices]
            data_tlimmag = FITS(fits_tlimmag)[-1]
            tlimmags = get_fitsio_values (data_tlimmag, y_indices, x_indices)

            
            # convert limmag from number of sigma listed
            # in the image header to input [nsigma]
            fits_trans = '{}_red_trans.fits'.format(basename)
            #header_trans = read_hdulist (fits_trans, get_data=False,
            #                             get_header=True)
            header_trans = FITS(fits_trans)[-1].read_header()

            if ('NSIGMA' in header_trans and
                isinstance(header_trans['NSIGMA'], (float, int)) and 
                header_trans['NSIGMA'] != 0):
                nsigma_trans_orig = header_trans['NSIGMA']

            if nsigma_trans_orig != nsigma:
                tlimmags += -2.5*np.log10(nsigma/nsigma_trans_orig)

        else:
            log.warning ('{} not found; no limiting magnitudes available'
                         .format(fits_tlimmag))
            tlimmags = np.zeros(ncoords)


        # zp, object airmass, ext_coeff and exptime were
        # determined above; for conversion from transient
        # limiting magnitude to Fpsferr the airmass at image
        # centre was used
        airmassc = header['AIRMASSC']
        Fpsferr = (10**(-0.4*(tlimmags - zp + airmassc * ext_coeff))
                   * exptime / get_par(set_zogy.transient_nsigma,tel))


        # read off transient S/N from Scorr image
        fits_Scorr = '{}_red_Scorr.fits.fz'.format(basename)
        #data_Scorr = read_hdulist (fits_Scorr)
        #snr_zogy = data_Scorr[y_indices, x_indices]
        data_Scorr = FITS(fits_Scorr)[-1]
        snr_zogy = get_fitsio_values (data_Scorr, y_indices, x_indices)


        # infer calibrated magnitudes using the zeropoint
        mag_zogy, magerr_zogy = apply_zp (np.abs(Fpsf), zp, airmass, exptime,
                                          filt, ext_coeff, fluxerr=Fpsferr)


        # check if [use_catalog_mags] is True
        if use_catalog_mags:
            
            # look for nearby source in transient catalog
            fits_trans = '{}_red_trans.fits'.format(basename)
            table_trans = Table.read(fits_trans)
            colnames_trans = table_trans.colnames

            if len(table_trans) > 0:

                # find matches between pixel coordinates (xcoords,
                # ycoords) and sources in transient catalog;
                # return their respective indices and separation
                i_coords, i_trans, sep, __, __ = get_matches (
                    table['RA_IN'].quantity.value,
                    table['DEC_IN'].quantity.value,
                    table_trans['RA_PSF_D'].quantity.value,
                    table_trans['DEC_PSF_D'].quantity.value,
                    dist_max=sep_max, return_offsets=True)


                mag_zogy[i_coords] = np.array(table_trans['MAG_ZOGY'][i_trans])
                magerr_zogy[i_coords] = np.array(table_trans['MAGERR_ZOGY'][i_trans])
                snr_zogy[i_coords] = np.array(table_trans['SNR_ZOGY'][i_trans])
                    

                # with match, position slightly changed, so update
                # the tlimmag for the coordinates with matches
                x_indices[i_coords] = np.array(table_trans['X_PEAK'][i_trans])-1
                y_indices[i_coords] = np.array(table_trans['Y_PEAK'][i_trans])-1
                #tlimmags = data_tlimmag[y_indices, x_indices]
                tlimmags = get_fitsio_values (data_tlimmag, y_indices, x_indices)
                if nsigma_trans_orig != nsigma:
                    tlimmags += -2.5*np.log10(nsigma/nsigma_trans_orig)


                # add separation
                table['SEP_TRANS'][i_coords] = sep


                # add additional catalog columns if present
                for col in catcols2add:
                    if col in colnames_trans and col in colnames:
                        table[col][i_coords] = table_trans[col][i_trans]



        # update table
        table['MAG_ZOGY'] = mag_zogy
        table['MAGERR_ZOGY'] = magerr_zogy
        table['SNR_ZOGY'] = snr_zogy
        table['TRANS_LIMMAG_{}SIGMA'.format(nsigma)] = tlimmags



    # reference; determine optimal fluxes
    # -----------------------------------
    if ref:

        # infer path to ref folder
        ref_dir = get_par(set_bb.ref_dir,tel)
        # read field ID from header
        obj, filt = header['OBJECT'], header['FILTER']
        # reference image and catalog names including full path
        basename_ref = '{}/{}/{}_{}'.format(ref_dir, obj, tel, filt)
        fits_red_ref = '{}_red.fits.fz'.format(basename_ref)
        fits_cat_ref = '{}_red_cat.fits'.format(basename_ref)
        

        # read reference image header
        header_ref = FITS(fits_red_ref)[-1].read_header()
        

        # read FWHM from the header
        if 'PSF-FWHM' in header_ref:
            fwhm_ref = header_ref['PSF-FWHM']
        elif 'S-FWHM' in header_ref:
            fwhm_ref = header_ref['S-FWHM']
        else:
            fwhm_ref = 5
            log.warning ('keywords PSF-FWHM nor S-FWHM present in the header '
                         'for {}; assuming fwhm_ref=5 pix'.format(basename_ref))


        # data_shape from header
        data_shape_ref = (header_ref['ZNAXIS2'], header_ref['ZNAXIS1'])
        ysize_ref, xsize_ref = data_shape_ref

        
        # convert input [ras_deg] and [decs_deg] to pixel coordinates
        # on the reference image
        xcoords_ref, ycoords_ref = (WCS(header_ref).
                                    all_world2pix(ras_deg, decs_deg, 1))

        
        # indices of pixel coordinates
        x_indices_ref = (xcoords_ref-0.5).astype(int)
        y_indices_ref = (ycoords_ref-0.5).astype(int)


        # define PSFEx binary fits table to use
        psfex_bintable = '{}_red_psf.fits'.format(basename_ref)

        if not os.path.exists(psfex_bintable):
            log.warning ('{} not found; skipping extraction of reference '
                         'magnitudes on {}'.format(psfex_bintable,
                                                   basename_ref))
            return [list(table[i]) for i in range(len(table))]

        
        # background STD
        fits_bkg_std = '{}_red_bkg_std.fits.fz'.format(basename_ref)
        if os.path.exists(fits_bkg_std):
            #data_bkg_std = read_hdulist (fits_bkg_std, dtype='float32')
            data_bkg_std = FITS(fits_bkg_std)[-1]
        else:
            # if it does not exist, create it from the background mesh
            fits_bkg_std_mini = '{}_red_bkg_std_mini.fits'.format(basename_ref)
            data_bkg_std_mini, header_mini = read_hdulist (
                fits_bkg_std_mini, get_header=True, dtype='float32')

            if 'BKG-SIZE' in header_mini:
                bkg_size = header_mini['BKG-SIZE']
            else:
                bkg_size = get_par(set_zogy.bkg_boxsize,tel)


            if ncoords == 1:
                # determine scalar bkg_std value from mini image at
                # xcoord, ycoord
                x_indices_mini = (((xcoords_ref-0.5).astype(int)/bkg_size)
                                  .astype(int))
                y_indices_mini = (((ycoords_ref-0.5).astype(int)/bkg_size)
                                  .astype(int))
                [data_bkg_std] = data_bkg_std_mini[y_indices_mini,
                                                   x_indices_mini]

            else:
                # determine full bkg_std image from mini image
                
                # determine whether interpolation is allowed across different
                # channels in [mini2back] using function get_Xchan_bool
                chancorr = get_par(set_zogy.MLBG_chancorr,tel)
                interp_Xchan_std = get_Xchan_bool (tel, chancorr, 'ref',
                                                   std=True)
                data_bkg_std = mini2back (
                    data_bkg_std_mini, data_shape_ref, order_interp=1,
                    bkg_boxsize=bkg_size, interp_Xchan=interp_Xchan_std,
                    timing=get_par(set_zogy.timing,tel))



        # object mask - not part of the standard zogy products at the
        # moment, so not available (yet); for the time being, let this
        # depend on input parameter [bkg_global]
        if bkg_global:
            # if True, global background is used, i.e. any local flux
            # due to nearby sources or galaxy is not taken into
            # account
            objmask = np.ones (data_shape_ref, dtype=bool)
        else:
            # if False, a circular annulus around each object is used
            # to estimate the sky background
            objmask = np.zeros (data_shape_ref, dtype=bool)


        # read reduced image
        data = read_hdulist (fits_red_ref)

        # read mask image
        fits_mask_ref = '{}_mask.fits.fz'.format(basename_ref)
        #data_mask = read_hdulist (fits_mask_ref, dtype='uint8')
        data_mask = FITS(fits_mask_ref)[-1]


        # add combined FLAGS_MASK_REF column to output table using
        # [get_flags_mask_comb]
        table['FLAGS_MASK_REF'] = get_flags_mask_comb (
            data_mask, xcoords_ref, ycoords_ref, fwhm_ref, xsize_ref, ysize_ref)
        
        
        try:
            # determine optimal fluxes at pixel coordinates
            flux_opt, fluxerr_opt = get_psfoptflux (
                psfex_bintable, data, data_bkg_std**2, data_mask,
                xcoords_ref, ycoords_ref, imtype='ref', fwhm=fwhm_ref,
                D_objmask=objmask, set_zogy=set_zogy, tel=tel)

        except Exception as e:
            log.error ('exception was raised while executing [get_psfoptflux]; '
                       'skipping extraction of magnitudes from {}: {}'
                       .format(basename_ref, e))
            return [list(table[i]) for i in range(len(table))]


        # infer some reference image header keywords
        keys = ['EXPTIME', 'PC-ZP', 'AIRMASS']
        exptime, zp, airmass = [header_ref[key] for key in keys]

        # infer calibrated magnitudes using the zeropoint
        mag_opt, magerr_opt = apply_zp (np.abs(flux_opt), zp, airmass, exptime,
                                        filt, ext_coeff, fluxerr=fluxerr_opt)


        # read limiting magnitude at pixel coordinates
        fits_limmag = '{}_red_limmag.fits.fz'.format(basename_ref)
        if os.path.exists(fits_limmag):

            #data_limmag = read_hdulist(fits_limmag)
            #limmags = data_limmag[y_indices_ref, x_indices_ref]
            data_limmag = FITS(fits_limmag)[-1]
            limmags = get_fitsio_values (data_limmag,
                                         y_indices_ref, x_indices_ref)

            # convert limmag from number of sigma listed
            # in the image header to input [nsigma]
            if ('NSIGMA' in header and
                isinstance(header['NSIGMA'], (float, int)) and
                header['NSIGMA'] != 0):
                nsigma_orig = header['NSIGMA']

            if nsigma_orig != nsigma:
                limmags += -2.5*np.log10(nsigma/nsigma_orig)

        else:
            log.warning ('{} not found; no limiting magnitudes available'
                         .format(fits_limmag))
            limmags = None


        # check if [use_catalog_mags] is True
        if use_catalog_mags:

            # look for nearby source in full-source catalog
            table_cat = Table.read(fits_cat_ref)
            colnames_cat = table_cat.colnames

            if len(table_cat) > 0:

                # find matches between pixel coordinates (xcoords,
                # ycoords) and sources in full-source catalog;
                # return their respective indices and separation
                i_coords, i_cat, sep, __, __ = get_matches (
                    table['RA_IN'].quantity.value,
                    table['DEC_IN'].quantity.value,
                    table_cat['RA'].quantity.value,
                    table_cat['DEC'].quantity.value,
                    dist_max=sep_max, return_offsets=True)


                mag_opt[i_coords] = np.array(table_cat['MAG_OPT'][i_cat])
                magerr_opt[i_coords] = np.array(table_cat['MAGERR_OPT'][i_cat])
                flux_opt[i_coords] = np.array(table_cat['E_FLUX_OPT'][i_cat])
                fluxerr_opt[i_coords] = np.array(table_cat['E_FLUXERR_OPT'][i_cat])


                if False:
                    # update existing FLAGS_MASK column
                    table['FLAGS_MASK'][i_coords] = np.array(
                        table_cat['FLAGS_MASK'][i_cat])


                # add separation
                table['SEP_REF'][i_coords] = sep


                # add additional catalog columns if present
                for col in catcols2add:
                    if col in colnames_cat and col in colnames:
                        table[col][i_coords] = table_cat[col][i_cat]



        # calculate signal-to-noise ratio; applies to either the SNR
        # of the limit or the matched source in the catalog
        snr_opt = np.zeros(flux_opt.shape)
        mask_nonzero = (fluxerr_opt != 0)
        snr_opt[mask_nonzero] = flux_opt[mask_nonzero] / fluxerr_opt[mask_nonzero]


        # update table
        # add these to table
        table['X_POS_IN_REF'] = xcoords_ref
        table['Y_POS_IN_REF'] = ycoords_ref
        table['MAG_OPT_REF'] = mag_opt
        table['MAGERR_OPT_REF'] = magerr_opt
        table['SNR_OPT_REF'] = snr_opt
        table['LIMMAG_{}SIGMA_REF'.format(nsigma)] = limmags


        # if there is a significant source in the reference image with
        # positive flux, correct MAG_ZOGY for it and add the result in
        # the column MAG_ZOGY_PLUSREF
        if 'mag_zogy' in locals() and 'snr_zogy' in locals():

            # start off with MAG_ZOGY
            table['MAG_ZOGY_PLUSREF'] = mag_zogy

            # transient flux in arbitrary flux units; mag_zogy is
            # always positive, even in case of a negative transient,
            # i.e. when the flux in the reference image was higher, so
            # include sign of [snr_zogy]
            flux_zogy = np.sign(snr_zogy) * 10**(-0.4*mag_zogy)

            # reference flux in arbitrary units; [snr_opt] is required
            # to be positive, so [flux_ref] will be positive
            flux_ref = 10**(-0.4*mag_opt)

            # corrected flux and magnitude
            flux_corr = (flux_zogy + flux_ref)
            mag_corr = -2.5 * np.log10(np.abs(flux_corr))

            # add to table only for signficant detections in the
            # reference image
            mask_sign = (snr_opt >= nsigma)
            table['MAG_ZOGY_PLUSREF'][mask_sign] = mag_corr[mask_sign]
                
            


    mem_use('at end of [get_rows]')

    return [list(table[i]) for i in range(len(table))]


################################################################################

def get_fitsio_values (data, y_indices=None, x_indices=None):
    
    if y_indices is None or x_indices is None:
        values = data[:,:]       
    else:
        nvalues = len(y_indices)
        values = np.zeros(nvalues)
        for i in range(nvalues):
            values[i] = data[y_indices[i]:y_indices[i]+1,
                             x_indices[i]:x_indices[i]+1]
            
    return values


################################################################################

def get_flags_mask_comb (data_mask, xcoords, ycoords, fwhm, xsize, ysize):

    # identify mask pixels within 2xFWHM of the pixel
    # coordinate; full size of window around coordinates,
    # make sure it is even
    size_4fwhm = int(4*fwhm+0.5)
    if size_4fwhm % 2 != 0:
        size_4fwhm += 1

    hsize = int(size_4fwhm/2)

    # define meshgrid
    xy = range(1, size_4fwhm+1)
    xx, yy = np.meshgrid(xy, xy, indexing='ij')

    # initialize flags_mask_comb
    ncoords = len(xcoords)
    flags_mask_comb = np.zeros(ncoords, dtype='uint')

    # loop coordinates
    for m in range(ncoords):

        # get index around x,y position using function
        # [zogy.get_index_around_xy]
        index_full, index_tn = (get_index_around_xy(ysize, xsize, ycoords[m],
                                                    xcoords[m], size_4fwhm))

        if np.sum(data_mask[index_full]) != 0:

            # create zero-valued temporary thumbnail data_mask
            data_mask_tn = np.zeros((size_4fwhm, size_4fwhm), dtype='uint')

            # fill temporary thumbnail with values from full data_mask
            data_mask_tn[index_tn] = data_mask[index_full]

            # define mask of pixels within 2*fwhm of the pixel coordinates
            mask_central = (np.sqrt(
                (xx - (xcoords[m] - int(xcoords[m]) + hsize))**2 +
                (yy - (ycoords[m] - int(ycoords[m]) + hsize))**2) < 2*fwhm)

            # add sum of unique values in the thumbnail data mask
            # to the output array flags_mask_comb
            flags_mask_comb[m] = np.sum(np.unique(data_mask_tn[mask_central]))



    return flags_mask_comb


################################################################################

def coords2field(ra, dec):
    '''
    Returns the BlackGEM / MeerLICHT field IDS for a given position.
    Parameters
    ----------
    ra : float
        Right ascension of the position in degrees.
    dec : float
        Declination in degrees
        
    Returns
    -------
    fields : ndarray
        An array of ints with the ids of the fields.
    '''
    side = 1.64
    hside = side/2.
    
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    centers = SkyCoord(MLBG_fields["ra_c"]*u.deg,
                       MLBG_fields["dec_c"]*u.deg, frame="icrs")
    centers.transform_to(SkyOffsetFrame(origin=centers))
    target_centers = target.transform_to(SkyOffsetFrame(origin=centers))
    xi, eta = target_centers.lon, target_centers.lat
       
    contains_mask = (np.abs(xi) < hside*u.deg) * (np.abs(eta)< hside*u.deg)

    #return fields[contains_mask]
    return MLBG_fields[contains_mask]['field_id']


################################################################################

def radec_images (radec_list, mjds_obs, dtime_max, basenames, radecs_cntr):

    """function to return the input coordinates as a tuple and the
    corresponding list of images"""

    # list to return
    radec_list_out = []

    # loop input list
    nfiles = mjds_obs.size
    delta_sep = 1
    hside = 1.655/2
    for radec in radec_list:

        # depending on the length of [radec], extract
        # the ra, dec and possibly also mjd
        mask_dtime = np.ones(nfiles, dtype=bool)
        if len(radec)==2:

            (ra_in, dec_in) = radec
        
        elif len(radec)==3:

            # in case mjd is also provided
            (ra_in, dec_in, mjd_in) = radec

            # determine mask of filenames that satisfies MJD window
            if mjd_in is not None:
                mask_dtime &= (np.abs(mjds_obs - mjd_in) <= dtime_max/24)


        # limit [radecs_cntr] to [delta_sep] degrees from [radec]
        mask_radec = (np.abs(radecs_cntr[:,1] - dec_in) <= delta_sep)


        # calculate separation between radec and radecs_cntr array; use
        # subset of radecs_cntr defined by mask_radec to limit the size of
        # the array on which haversine is run
        sep = haversine(ra_in, dec_in,
                        radecs_cntr[:,0][mask_radec],
                        radecs_cntr[:,1][mask_radec])
        # fill the True elements of mask_radec with the mask (sep < delta_sep)
        mask_radec[mask_radec] = (sep < delta_sep)


        # could determine whether radec is actually on which of
        # images[mask_radec], but whether the pixel coordinates are on
        # the image is quickly determined in [get_rows], so don't
        # really need to that here; but if [args.nepochs_min] is
        # larger than 1, then this is relevant.
        if True:
            target = SkyCoord(ra=ra_in, dec=dec_in, unit='deg')
            centers = SkyCoord(ra=radecs_cntr[:,0][mask_radec],
                               dec=radecs_cntr[:,1][mask_radec], unit='deg')
            centers.transform_to(SkyOffsetFrame(origin=centers))
            target_centers = target.transform_to(SkyOffsetFrame(origin=centers))
            xi, eta = target_centers.lon, target_centers.lat
            mask_on = (np.abs(xi) < hside*u.deg) * (np.abs(eta)< hside*u.deg)
            mask_radec[mask_radec] = mask_on


        # combined mask
        mask_files = mask_dtime & mask_radec
        if False and np.sum(mask_files) > 0:

            info_str = ('{} filename(s) with matching field IDs for object at '
                        'RA: {}, DEC: {}'
                        .format(np.sum(mask_files), ra_in, dec_in))

            if len(radec)==3:
                info_str = ('{}, within {} hours of MJD {:.4f}'
                            .format(info_str, dtime_max, mjd_in))
            
            log.info (info_str)


        basenames_radec = list(np.array(basenames)[mask_files].astype(str))


        if len(basenames_radec) > 0:
            radec_list_out += (ra_in, dec_in), basenames_radec


    return radec_list_out
    

################################################################################

def file2fullpath (filenames, set_zogy=None, set_bb=None):

    """returns a list of basenames for a list of MeerLICHT or BlackGEM
    [filenames] (can be a mix of telescopes) including their full
    path; i.e. [full path]/[tel]_yyyymmdd_hhmmss is returned for each
    item in [filenames]

    """

    # determine dictionary with UTC offsets for the different
    # telecopes
    UTC_offset = {}
    for tel_tmp in ['ML1', 'BG2', 'BG3', 'BG4']:
        UTC_offset[tel_tmp] = (datetime.now().replace(
            tzinfo=gettz(get_par(set_zogy.obs_timezone,tel_tmp)))
                               .utcoffset().total_seconds()/3600)

    # define basename of filename, including full path, i.e.  /[path
    # to reduced folder]/[tel]_yyyymmdd_hhmmss_red, without fits or
    # fits.fz extension
    fullnames = []
    for filename in filenames:
        
        # infer basename [tel]_yyyymmdd_hhmmss
        basename = filename.split('/')[-1][0:19]

        # infer date of observation in isot format from basename
        [tel_tmp, date_tmp, time_tmp] = basename.split('_')
        date_obs = '{}-{}-{}T{}:{}:{}'.format(
            date_tmp[0:4], date_tmp[4:6], date_tmp[6:8],
            time_tmp[0:2], time_tmp[2:4], time_tmp[4:6])

        # path to the reduced folder; needs to be inside loop because
        # input filenames can be from different telescopes
        red_dir = get_par(set_bb.red_dir,tel_tmp)

        # determine evening date and yyyy/mm/dd subfolder
        date_eve = (Time(int(Time(date_obs).jd+UTC_offset[tel_tmp]/24),
                         format='jd').strftime('%Y%m%d'))
        date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])
        
        # add basename with its full path to output list
        fullnames.append('{}/{}/{}'.format(red_dir, date_dir, basename))


    return fullnames


################################################################################

def get_headkeys (basenames):

    nfiles = len(basenames)
    objects = np.zeros(nfiles, dtype=int)
    mjds_obs = np.zeros(nfiles)
    filts = np.zeros(nfiles, dtype=str)
    
    for nfile, basename in enumerate(basenames):
        
        # read header
        #with fits.open('{}_red_hdr.fits'.format(basename)) as hdulist:
        #    header = hdulist[-1].header
        header = FITS('{}_red_hdr.fits'.format(basename))[-1].read_header()

                
        objects[nfile] = int(header['OBJECT'])
        mjds_obs[nfile] = header['MJD-OBS']
        filts[nfile] = header['FILTER'].strip()
        

    return objects, mjds_obs, filts


################################################################################

def get_mjd_mask (mjds, date_start, date_end, date_format):

    mask_mjd = np.ones(mjds.size, dtype=bool)
    if date_start is not None:
        mjd_start = Time(date_start, format=date_format).mjd
        mask_mjd &= (mjds >= mjd_start)

    if date_end is not None:
        mjd_end = Time(date_end, format=date_format).mjd
        mask_mjd &= (mjds <= mjd_end)


    return mask_mjd


################################################################################

# from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Perform (transient) forced '
                                     'photometry on MeerLICHT/BlackGEM data')


    parser.add_argument('radecs', type=str,
                        help='comma-separated list of RA,DEC coordinates '
                        '(ra1,dec1,ra2,dec2,...) or the name of a file with '
                        'format [radecs_file_format] containing the RA and DEC '
                        'coordinates with the column names [ra_col] and '
                        '[dec_col] and optionally also a date of observation '
                        'with the column name [date_col] and astropy Time format '
                        '[date_format]; RA and DEC can either be both in decimal '
                        'degrees or in colon-separated sexagesimal hours and '
                        'degrees for RA and DEC, respectively')

    parser.add_argument('fits_out', type=str,
                        help='output fits table with resulting magnitudes')

    parser.add_argument('--radecs_file_format', type=str, default='fits',
                        help='astropy file format of [radecs] in case it points '
                        'to a file, e.g. ascii or csv; default=fits')

    parser.add_argument('--ra_col', type=str, default='RA',
                        help='name of input RA column in [radecs]; default=RA')

    parser.add_argument('--dec_col', type=str, default='DEC',
                        help='name of input DEC column in [radecs]; default=DEC')

    parser.add_argument('--date_col', type=str, default='DATE-OBS',
                        help='name of input date of observation in [radecs]; '
                        'default=DATE-OBS')

    parser.add_argument('--date_format', type=str, default='isot',
                        help='astropy.time.Time format of [date_col]; '
                        'default=isot')

    parser.add_argument('--input_cols2copy', type=str, default=None,
                        help='comma-separated list of input column names to '
                        'add to the output fits table (optional); the columns '
                        '[ra_col], [dec_col] and [date_col] - the latter only '
                        'if it is present in [radecs] - are copied by default; '
                        'default=None')

    parser.add_argument('--filenames', type=str,
                        default='/idia/projects/meerlicht/Headers/ML1_headers_cat.fits',
                        help='comma-separated list with the filenames of the '
                        'reduced images to be used or the name of a file with '
                        'format [filenames_file_format] containing the filenames '
                        'with the column name [filenames_col]; '
                        'default=/idia/projects/meerlicht/Headers/ML1_headers_cat.fits')

    parser.add_argument('--filenames_col', type=str, default='FILENAME',
                        help='name of input filename column in [filenames]; '
                        'default=FILENAME')

    parser.add_argument('--filenames_file_format', type=str, default='fits',
                        help='astropy file format of [filenames] in case it '
                        'points to a file; default=fits')

    parser.add_argument('--filters', type=str, default='ugqriz',
                        help='consider images in these filters only; '
                        'default=\'ugqriz\'')

    parser.add_argument('--date_start', type=str, default=None,
                        help='starting UTC date of observation in format '
                        '[date_format] for measurements to consider; '
                        'default=None')

    parser.add_argument('--date_end', type=str, default=None,
                        help='ending UTC date of observation in format '
                        '[date_format] for measurements to consider; '
                        'default=None')

    parser.add_argument('--trans', type=str2bool, default=True,
                        help='extract transient magnitudes?; default=True')

    parser.add_argument('--ref', type=str2bool, default=True,
                        help='extract reference magnitudes?; default=True')

    parser.add_argument('--fullsource', type=str2bool, default=False,
                        help='extract full-source magnitudes?; default=False')

    parser.add_argument('--bkg_global', type=str2bool, default=False,
                        help='for full-source case only: use global background '
                        'estimate (T) or estimate local background from annulus '
                        'around the coordinates (F); default=True')

    parser.add_argument('--nsigma', type=int, default=3,
                        help='significance threshold for a detection; default=3')

    parser.add_argument('--use_catalog_mags', type=str2bool, default=False,
                        help='use magnitudes from nearest catalog source '
                        'within [sep_max] instead of forced photometry; '
                        'reverts to forced photometry if no source is detected; '
                        'default=False')

    parser.add_argument('--sep_max', type=float, default=2,
                        help='[arcsec] maximum separation of catalog source '
                        'from the input coordinates; only relevant if '
                        '[use_catalog_mags] is set to True; default=2')

    parser.add_argument('--dtime_max', type=float, default=1,
                        help='[hr] maximum time difference between the input '
                        'date of observation in [radecs] and the filename date '
                        'of observation; [dtime_max] is not used if input date '
                        'of observation is not provided; default=1')

    parser.add_argument('--nepochs_min', type=int, default=1,
                        help='minimum number of epochs required for a set of '
                        'coordinates to be processed and feature in the output '
                        'table; default=1')

    par_default = ('E_FLUX_OPT,E_FLUXERR_OPT,RA,DEC,ELONGATION,FWHM,'
                   'CLASS_STAR,FLAGS,MAG_APER_R1.5xFWHM,MAGERR_APER_R1.5xFWHM,'
                   'MAG_APER_R5xFWHM,MAGERR_APER_R5xFWHM,'
                   'E_FLUX_ZOGY,E_FLUXERR_ZOGY,RA_PSF_D,DEC_PSF_D,MAG_PSF_D,'
                   'MAGERR_PSF_D,CHI2_PSF_D,FWHM_GAUSS_D,ELONG_GAUSS_D,'
                   'CHI2_GAUSS_D,CLASS_REAL')

    parser.add_argument('--catcols2add', type=str, default=par_default,
                        help='additional columns from the full-source and/or '
                        'transient and/or reference catalogs to add to output '
                        'table for the entries with catalog matches; only '
                        'relevant if [use_catalog_mags] is set to True. For any '
                        'reference column add \'_REF\' to the name, to separate '
                        'them from the often-identical full-source column names; '
                        'default: {}'.format(par_default))

    par_default = ('float,float,float,float,float,float,float,uint8,float,float,'
                   'float,float,float,float,float,float,float,float,float,float,'
                   'float,float,float')

    parser.add_argument('--catcols2add_dtypes', type=str, default=par_default,
                        help='corresponding catalog columns dtypes; default: {}'
                        .format(par_default))

    par_default = 'MJD-OBS,OBJECT,FILTER,EXPTIME,S-SEEING,AIRMASS,PC-ZP,' \
        'PC-ZPSTD,QC-FLAG'
    parser.add_argument('--keys2add', type=str, default=par_default,
                        help='header keyword values to add to output '
                        'table; default: {}'.format(par_default))

    par_default = 'float,U5,U1,float,float,float,float,float,U6'
    parser.add_argument('--keys2add_dtypes', type=str, default=par_default,
                        help='corresponding header keyword dtypes; default: {}'
                        .format(par_default))

    parser.add_argument('--ncpus', type=int, default=None,
                        help='number of CPUs to use; if None, the number of '
                        'CPUs available as defined by environment variables '
                        'SLURM_CPUS_PER_TASK or OMP_NUM_THREADS will be used; '
                        'default=None')
    
    parser.add_argument('--logfile', type=str, default=None,
                        help='if name is provided, an output logfile is created; '
                        'default=None')
    
    args = parser.parse_args()


    # for timing
    t0 = time.time()


    # create logfile
    if args.logfile is not None:

        # since logfile is defined, change StreamHandler loglevel to
        # ERROR so that not too much info is sent to stdout
        for handler in log.handlers[:]:
            if 'Stream' in str(handler):
                handler.setLevel(logging.WARNING)

        fileHandler = logging.FileHandler(args.logfile, 'w')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel('INFO')
        log.addHandler(fileHandler)
        log.info ('logfile created: {}'.format(args.logfile))


    # define number of CPUs to use [ncpus]; if input parameter [npcus]
    # is defined, that value is used. If not and if the ilifu cluster
    # environment variable SLURM_CPUS_PER_TASK is set, either through
    # the --cpus-per-task option in the sbatch script, or when opening
    # an interactive node with multiple processors, that value is
    # adopted. If not, the environment variable OMP_NUM_THREADS is
    # looked for and used if defined.  If none of the above are
    # defined, npcus=1 is used.
    slurm_ncpus = os.environ.get('SLURM_CPUS_PER_TASK')
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    if args.ncpus is not None:
        ncpus = args.ncpus
        if slurm_ncpus is not None and ncpus > int(slurm_ncpus):
            log.warning ('number of CPUs defined ({}) is larger than the number '
                         'available ({})'.format(ncpus, slurm_ncpus))
        elif omp_num_threads is not None and ncpus > int(omp_num_threads):
            log.warning ('number of CPUs defined ({}) is larger than the number '
                         'available ({})'.format(ncpus, omp_num_threads))
    else:
        if slurm_ncpus is not None:
            ncpus = int(slurm_ncpus)
        elif omp_num_threads is not None:
            ncpus = int(omp_num_threads)
        else:
            ncpus = 1


    log.info ('number of CPUs used: {}'.format(ncpus))
        

    # infer RAs and DECs to go through from [args.radecs]
    # ---------------------------------------------------
    mjds_in = None
    if os.path.isfile(args.radecs):

        # if [args.radecs] is a file, read it into a table
        table_radecs = Table.read(args.radecs, format=args.radecs_file_format)
        colnames = table_radecs.colnames
        log.info ('input table column names: {}'.format(colnames))


        if len(table_radecs) == 0:
            log.critical('no input coordinates found; if the input is meant to '
                         'be a file, check whether its format provided through '
                         'the input parameter [radecs_file_format] is correct')
            raise SystemExit
        else:
            log.info ('{} lines in input file {}'
                      .format(len(table_radecs), args.radecs))


        # convert column [date_col] to list mjds_in
        if args.date_col is not None and args.date_col in colnames:
            dates_in = list(table_radecs[args.date_col])
            # convert to mjds
            mjds_in = Time(dates_in, format=args.date_format).mjd
            
            # filter table_radecs by comparing mjds_in with
            # args.date_start and args.date_end
            mask_dtime = get_mjd_mask (mjds_in, args.date_start, args.date_end,
                                       args.date_format)
            table_radecs = table_radecs[mask_dtime]

            log.info ('{} lines in input file {} after filtering on input '
                      '[date_start] and [date_end]'
                      .format(args.radecs, len(table_radecs)))

        else:
            # if not provided, set mjds_in to None
            mjds_in = None
            #log.warning ('no info on {} is found in the input, so the '
            #             'coordinates in [radecs] will be searched for in all '
            #             '[filenames] rather than a subset within a time window '
            #             'centered on {} with a total width of [dtime] hours'
            #             .format(args.date_col, args.date_col))


        # read column [ra_col] to list ras_in
        if args.ra_col in colnames:
            ras_in = list(table_radecs[args.ra_col])
            if not isinstance(ras_in[0], float) and ':' in ras_in[0]:
                ras_in = Angle(ras_in, unit=u.hour).degree
                table_radecs[args.ra_col] = ras_in
        else:
            log.error ('column {} not present in {}; exiting'
                       .format(args.ra_col, args.radecs))
            raise SystemExit
            
        # read column [dec_col] to list decs_in
        if args.dec_col in colnames:
            decs_in = list(table_radecs[args.dec_col])
            if not isinstance(decs_in[0], float) and ':' in decs_in[0]:
                decs_in = Angle(decs_in, unit=u.deg).degree
                table_radecs[args.dec_col] = decs_in
        else:
            log.error ('column {} not present in {}; exiting'
                       .format(args.dec_col, args.radecs))
            raise SystemExit


    else:

        # split input [radecs] into list of strings ['ra1', 'dec1', ...]
        radecs_list0 = re.sub('\(|\)|\[|\]|\{|\}', '', args.radecs)
        radecs_list = re.sub(';', ',', radecs_list0).split(',')

        # remove potential empty entries and check for an even number
        remove_empty (radecs_list)
        if len(radecs_list) % 2 != 0:
            log.exception ('number of coordinates in [radecs] is not even; '
                           'exiting')
            raise SystemExit

        ras_in = []
        decs_in = []
        mjds_in = None
        log.info('radecs_list: {}'.format(radecs_list))
        for i, s in enumerate(radecs_list):
            if i % 2 == 0:
                # RAs:
                if ':' in s:
                    ras_in.append(Angle(s, unit=u.hour).degree)
                else:
                    ras_in.append(float(s))

            else:
                # DECs
                if ':' in s:
                    decs_in.append(Angle(s, unit=u.deg).degree)
                else:
                    decs_in.append(float(s))



        # create table as if input file was provided with only RAs and DECs
        table_radecs = Table()
        table_radecs[args.ra_col] = ras_in
        table_radecs[args.dec_col] = decs_in



    # add column identifying (line) number of input coordinate, for
    # easy mapping between input and output tables
    table_radecs['NUMBER_IN'] = np.arange(1,len(table_radecs)+1)


    # create array of unique (ra_in, dec_in, [mjd_in]) tuples
    if mjds_in is not None:
        radecs_in = np.array(list(zip(ras_in, decs_in, mjds_in)))
    else:
        radecs_in = np.array(list(zip(ras_in, decs_in)))


    # unique elements 
    __, index_uniq = np.unique(radecs_in, axis=0, return_index=True)
    # sort the indexes to keep the original order; np.unique returns
    # the sorted values and corresponding indices
    index_uniq = np.sort(index_uniq)


    # update table_radecs and radecs_in
    table_radecs = table_radecs[index_uniq]
    radecs_in = radecs_in[index_uniq]

    log.info('{} unique RADECs (and MJDs) in input list/file [radecs]'
             .format(index_uniq.size))


    # check if all filters are needed
    filts_all = np.all([filt in args.filters for filt in 'ugqriz'])

        
    # infer list of filenames to consider from [args.filenames]
    # ---------------------------------------------------------
    radecs_cntr = None

    if os.path.isfile(args.filenames):

        # if the input is a single file, it could be a single image or
        # a fits table/file containing multiple files; to test between
        # the two, try to read the file as a table, which will cause
        # an exception if it is an image
        try:
            # read it into a table using the format
            # [args.filenames_file_format]
            table_filenames = Table.read(args.filenames,
                                         format=args.filenames_file_format)
            log.info ('{} line(s) in input file {}'
                      .format(len(table_filenames), args.filenames))


            # get central coordinates of filenames
            colnames = table_filenames.colnames
            if 'RA-CNTR' in colnames and 'DEC-CNTR' in colnames:

                # mask with files with a WCS solution
                mask_WCS = (np.isfinite(table_filenames['RA-CNTR']) &
                            np.isfinite(table_filenames['DEC-CNTR']) &
                            (table_filenames['RA-CNTR'] > 0) &
                            (table_filenames['RA-CNTR'] < 360) &
                            (np.abs(table_filenames['DEC-CNTR']) < 90))

                log.info ('{} filename(s) with valid CNTR coordinates'
                          .format(np.sum(mask_WCS)))
                table_filenames = table_filenames[mask_WCS]


                # define list of (ra_cntr,dec_cntr) tuples to be used in
                # function [radec_images]
                radecs_cntr = np.array(list(zip(table_filenames['RA-CNTR'],
                                                table_filenames['DEC-CNTR'])))


            # define list of filenames
            filenames = list(table_filenames[args.filenames_col])


            # define objects and mjds_obs arrays
            if 'OBJECT' in colnames and 'MJD-OBS' in colnames:
                objects = np.array(table_filenames['OBJECT']).astype(int)
                mjds_obs = np.array(table_filenames['MJD-OBS'])


            # and filter array if needed
            if not filts_all:
                if 'FILTER' in colnames:
                    filts = np.array(table_filenames['FILTER']).astype(str)


        except:
            # apparently the input is not a fits table, so assume it
            # is a single image and put it into a list
            filenames = args.filenames.split(',')
            remove_empty (filenames)
            
    else:

        # filenames were provided as a comma-separated list; put them
        # into a list
        filenames = args.filenames.split(',')
        remove_empty (filenames)



    # in case the dates of observation [mjds_obs] or field IDs
    # [objects] have not yet been defined, need to infer them from the
    # headers
    if ('mjds_obs' not in locals() or 'objects' not in locals() or
        (not filts_all and 'filts' not in locals())): 
        basenames = [fn.split('_red')[0] for fn in filenames]
        objects, mjds_obs, filts = get_headkeys (basenames)



    # filtering of filenames
    # ----------------------
    nfiles = len(filenames)
    mask_keep = np.ones(nfiles, dtype=bool)


    # filter by input args.date_start and args.date_end
    if args.date_start is not None or args.date_end is not None:
        mask_keep &= get_mjd_mask (mjds_obs, args.date_start, args.date_end,
                                   args.date_format)


    # filter by minimum and maximum [mjds_in] if it is not None
    if mjds_in is not None:
        mjd_start = min(mjds_in)-args.dtime_max
        mjd_end = max(mjds_in)+args.dtime_max
        mask_keep &= get_mjd_mask (mjds_obs, mjd_start, mjd_end, 'mjd')


    # filter by filters specified in args.filters
    if not filts_all:
        mask_filts = np.zeros(nfiles, dtype=bool)
        for filt in args.filters:
            mask_filts |= (filts == filt)

        # keep the relevant filters
        mask_keep &= mask_filts


    # update filenames, mjds_obs, objects and radecs_cntr
    if np.sum(~mask_keep) > 0:
        filenames = list(np.array(filenames)[mask_keep].astype(str))
        nfiles = len(filenames)
        mjds_obs = mjds_obs[mask_keep]
        objects = objects[mask_keep]
        if radecs_cntr is not None:
            radecs_cntr = radecs_cntr[mask_keep] 
            
        log.info ('{} filename(s) left to search through after filtering '
                  'on input [date_start] and [date_end] and/or on potential '
                  'input dates provided in input file [radecs] and/or on '
                  'filters provided in [filters]'.format(nfiles))



    # determine basenames    
    if True:
        
        basenames = [fn.split('_red')[0] for fn in filenames]

    else:
        # previously filenames were not required to include the full
        # paths and the latter were inferred from the filename itself
        # using the block below

        # define basename of filename, including full path, i.e.  /[path
        # to reduced folder]/[tel]_yyyymmdd_hhmmss, without any extensions;
        # for the moment, set tel=ML1, but this needs to be generalized
        # to BlackGEM in the future, e.g. by letting the filename in the
        # header tables to start with [tel]/red/yyyy/etc
        tel = 'ML1'
        red_dir = get_par(set_bb.red_dir,tel)

        # check if path ok for the last file
        path_ok = (len (glob.glob('{}/{}*'.format(red_dir, filenames[-1]))) > 0)

        t1 = time.time()
        if path_ok:
            # this assumes that the filename already starts with the
            # yyyy/mm/dd/ subfolders in the filename, like it does in the
            # default full-source and transient catalog header tables
            basenames = ['{}/{}'.format(red_dir, fn.split('_red')[0])
                         for fn in filenames]

        else:
            # use function [file2fullpath]
            basenames = file2fullpath (filenames, set_zogy=set_zogy,
                                       set_bb=set_bb)


        log.info ('time to determine basenames: {:.2f}'.format(time.time()-t1))



    # if [radecs_cntr] is not defined yet because the default fits
    # header was not used as args.filenames, infer it from the field
    # IDs' (objects') central coordinates of the ML/BG field grid
    # listed in table MLBG_fields, which is a global parameter set at
    # the top of the module
    if radecs_cntr is None:

        # dictionary with mapping of object ID to its index in
        # MLBG_fields table
        obj_indx = {int(MLBG_fields['field_id'][i]) : i
                    for i in range(len(MLBG_fields))}

        # define CNTR coordinates to be those of the ML/BG field grid
        radecs_cntr = np.array([(MLBG_fields['ra_c'][obj_indx[obj]],
                                 MLBG_fields['dec_c'][obj_indx[obj]])
                                for obj in objects])


    # limit the input coordinates to be within the range of observed
    # DECs plus 1 degree
    dec_cntr_min = np.min(radecs_cntr[:,1])
    dec_cntr_max = np.max(radecs_cntr[:,1])
    mask_coords = ((radecs_in[:,1] >= dec_cntr_min-1) &
                   (radecs_in[:,1] <= dec_cntr_max+1))

    if np.sum(~mask_coords) > 0:
        radecs_in = radecs_in[mask_coords]
        log.info ('{} entries in [radecs_in] after filtering on range of '
                  'observed declinations'.format(len(radecs_in)))


    # create {(RA1,DEC1): [images1], (RA2,DEC2): [images2] dictionary
    # using multiprocessing of function [radec_images]; the list of
    # tuples [radecs_in] will be unpacked correctly in the function
    # [radec_images], depending on whether it contains a list of
    # tuples with 2 or 3 elements.

    # to optimize the multiprocessing, split [radecs_in] into [ncpus]
    # lists so that each worker processes a number of radecs at a
    # time, rather than a single one
    radecs_in_lists = []
    index = np.linspace(0,len(radecs_in),num=ncpus+1).astype(int)
    for i in range(ncpus):
        radecs_in_lists.append(radecs_in[index[i]:index[i+1]])

    # run multiprocessing
    results = pool_func (radec_images, radecs_in_lists, mjds_obs, args.dtime_max,
                         basenames, radecs_cntr, nproc=ncpus)

    # get rid of lists in lists
    results = list(itertools.chain.from_iterable(results))


    # convert results to dictionary
    log.info ('converting resulting lists from [radec_images] to dictionary')
    radec_images_dict = {}
    for i in range(0,len(results),2):
        if len(results[i+1]) >= args.nepochs_min:
            radec_images_dict[results[i]] = results[i+1]


    # timing so far
    log.info ('time spent to select relevant images: {:.1f}s'
              .format(time.time()-t0))
    t1 = time.time()


    # could cut up dictionary in pieces if it turns out to be
    # too large, and loop over [force_phot] various times


    # convert input catcols2add and corresponding types to lists
    if args.catcols2add is not None:
        catcols2add = args.catcols2add.split(',')
    else:
        catcols2add = None

    if args.catcols2add_dtypes is not None:
        catcols2add_dtypes = args.catcols2add_dtypes.split(',')
    else:
        catcols2add_dtypes = None


    # check that they have the same length
    if catcols2add is not None and catcols2add_dtypes is not None:
        assert (len(catcols2add) == len(catcols2add_dtypes))
    

    # convert input keys2add and corresponding types to lists
    if args.keys2add is not None:
        keys2add = args.keys2add.upper().split(',')
    else:
        keys2add = None

    if args.keys2add_dtypes is not None:
        keys2add_dtypes = args.keys2add_dtypes.split(',')
    else:
        keys2add_dtypes = None

    # check that they have the same length
    if keys2add is not None and keys2add_dtypes is not None:
        assert (len(keys2add) == len(keys2add_dtypes))
        
    
    # call [force_phot]
    table_out = force_phot (
        radec_images_dict, trans=args.trans, ref=args.ref,
        fullsource=args.fullsource,
        nsigma=args.nsigma, use_catalog_mags=args.use_catalog_mags,
        sep_max=args.sep_max, catcols2add=catcols2add,
        catcols2add_dtypes=catcols2add_dtypes, keys2add=keys2add,
        keys2add_dtypes=keys2add_dtypes, bkg_global=args.bkg_global,
        ncpus=ncpus)


    # copy columns from the input to the output table; even if
    # [args.input_cols2copy] was not defined but [args.date_col] is
    # defined, let's copy over at least [args.date_col]
    if table_out is not None:

        if False:
            # rename columns RA and DEC to the input column names; if
            # the input columns were in sexagesimal notation, these
            # will be decimal degrees
            # decided to fix these names to RA_IN and DEC_IN, to
            # separate them from RA and DEC of full-source or
            # transient catalog match
            if 'RA' in table_out.colnames:
                table_out.rename_column('RA', args.ra_col)
                
            if 'DEC' in table_out.colnames:
                table_out.rename_column('DEC', args.dec_col)


        # add column with number of row to be able to order the output
        # table the same way
        # now done with 'NUMBER_IN' column which is written to the output
        # table
        #order_col = ''.join(choice(ascii_uppercase) for i in range(7))
        #table_radecs[order_col] = np.arange(len(table_radecs), dtype=int)


        # loop input [radecs] and create list of masks that determine
        # the index mapping from [table_radecs] to [table_out]; this
        # needs to be done only once and so is done before the loop
        # below
        mask_list = []
        for i in range(len(table_radecs)):
            mask = ((table_out['RA_IN']==table_radecs[args.ra_col][i]) &
                    (table_out['DEC_IN']==table_radecs[args.dec_col][i]))
            mask_list.append(mask)


        # first table_radecs column to add from the input table is NUMBER_IN
        cols2copy = ['NUMBER_IN']
        

        # add columns defined in [input_cols2copy]
        if args.input_cols2copy is not None:
            cols2copy += args.input_cols2copy.split(',')


        colnames = table_radecs.colnames
        if (args.date_col is not None and args.date_col in colnames and
            args.date_col not in cols2copy):
            cols2copy.append(args.date_col)


        for ncol, col2copy in enumerate(cols2copy):

            # check that the column is present in [table_radecs]
            if col2copy in colnames:

                # avoid the column names [args.ra_col] and [args.dec_col];
                # already present with the names RA_IN and DEC_IN
                if col2copy != args.ra_col and col2copy != args.dec_col:
                
                    # create empty array with the same length as [table_out]
                    # and with the same dtype as the input column
                    array_tmp = np.zeros(len(table_out),
                                         dtype=table_radecs[col2copy].dtype)

                    # loop [table_radecs] and fill above array using [mask_list]
                    for i in range(len(table_radecs)):
                        array_tmp[mask_list[i]] = table_radecs[col2copy][i]

                    # and add it to the output table at the beginning,
                    # but after the RA and DEC columns
                    table_out.add_column(array_tmp, name=col2copy, index=ncol+2)

            else:
                log.warning ('column {} not present in input file defined in '
                             'parameter [radecs]'.format(col2copy))


        # if [date_col] was provided, the delta time between it and
        # the image date of observations can be determined
        if args.date_col is not None and args.date_col in colnames:
            mjds_in = Time(table_out[args.date_col], format=args.date_format).mjd
            dtime_hr = 24*np.abs(mjds_in - table_out['MJD-OBS'])
            table_out.add_column(dtime_hr, name='DTIME',
                                 index=table_out.colnames.index('MJD-OBS')+1)


        # order the output table by original row number
        indices_sorted = np.argsort(table_out, order=(('NUMBER_IN','FILENAME')))
        table_out = table_out[indices_sorted]



    log.info ('time spent in [force_phot]: {:.1f}s'.format(time.time()-t1))
    log.info ('time spent in total:        {:.1f}s'.format(time.time()-t0))

    # list memory used
    mem_use('at end of [force_phot]')
    
    # write output table to fits
    if table_out is not None:
        table_out.write(args.fits_out, format='fits', overwrite=True)


