
import os

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

from astropy.coordinates import Angle

from zogy import *

import set_zogy
import set_blackbox as set_bb

try:
    from fitsio import FITS
except:
    log.info ('import of fitsio.FITS failed; using astropy.io.fits instead')
    

def force_phot (filenames, ra_deg, dec_deg, fits_out, nsigma=5,
                keys2add=None, keys2add_dtypes=None, tel='ML1'):

    
    # initialize rows to be converted to fits table
    rows = []

    # convert keys2add and keys2add_dtypes to lists
    keys2add_list = keys2add.upper().split(',')
    dtypes_list = keys2add_dtypes.split(',')
    

    # loop through [filenames]
    for nfile, filename in enumerate(filenames):

        # start row
        row = [filename]
        
        # read in header and data
        data, header = read_hdulist (filename, get_header=True)


        # add header keywords to output table
        for key in keys2add_list:
            if key in header:
                row.append (header[key])
            else:
                row.append (None)
                log.error ('key {} not in header of {}'
                           .format(key, filename))


        # convert input [ra_deg] and [dec_deg] to pixel coordinates
        xcoord, ycoord = WCS(header).all_world2pix(ra_deg, dec_deg, 1)

        # skip if coordinates not on the image
        ysize, xsize = data.shape
        dpix_edge = 10
        if (xcoord < dpix_edge or xcoord > xsize-dpix_edge or
            ycoord < dpix_edge or ycoord > ysize-dpix_edge):

            log.warning ('pixel coordinates (x,y)=({},{}) not on image for {}'
                         '; skipping it'.format(int(xcoord), int(ycoord),
                                                filename))
            continue


        # determine optimal flux
        # ----------------------

        # read PSFEx binary fits table
        base = filename.split('.fits')[0]
        psfex_bintable = '{}_psf.fits'.format(base)


        # background STD
        fits_bkg_std = '{}_bkg_std.fits'.format(base)
        if os.path.exists(fits_bkg_std):
            data_bkg_std = read_hdulist (fits_bkg_std, dtype='float32')
        else:
            # if it does not exist, create it from the background mesh
            fits_bkg_std_mini = '{}_bkg_std_mini.fits'.format(base)
            data_bkg_std_mini, header_mini = read_hdulist (
                fits_bkg_std_mini, get_header=True, dtype='float32')

            if 'BKG-SIZE' in header_mini:
                bkg_size = header_mini['BKG-SIZE']
            else:
                bkg_size = get_par(set_zogy.bkg_boxsize,tel)


            if True:
                # determine scalar bkg_std value from mini image at
                # xcoord, ycoord
                x_index_mini = int(int(xcoord-0.5)/bkg_size)
                y_index_mini = int(int(ycoord-0.5)/bkg_size)
                data_bkg_std = data_bkg_std_mini[y_index_mini, x_index_mini]

            else:
                # determine full bkg_std image from mini image

                # determine whether interpolation is allowed across different
                # channels in [mini2back] using function get_Xchan_bool
                chancorr = get_par(set_zogy.MLBG_chancorr,tel)
                interp_Xchan_std = get_Xchan_bool (tel, chancorr, 'new',
                                                   std=True)

                data_bkg_std = mini2back (data_bkg_std_mini, data.shape,
                                          order_interp=1, bkg_boxsize=bkg_size,
                                          interp_Xchan=interp_Xchan_std,
                                          timing=get_par(set_zogy.timing,tel))


        # data mask
        fits_mask = filename.replace('_red.fits', '_mask.fits')
        data_mask = read_hdulist (fits_mask, dtype='uint8')


        # object mask - needs to be created with source extractor; for
        # now, assume entire image is masked, leading to local
        # background not being used
        objmask = np.ones (data.shape, dtype=bool)


        # determine optimal fluxes at pixel coordinates
        flux_opt, fluxerr_opt = get_psfoptflux (
            psfex_bintable, data, data_bkg_std**2, data_mask,
            np.array([xcoord]), np.array([ycoord]),
            imtype='new', fwhm=header['PSF-FWHM'],
            D_objmask=objmask, set_zogy=set_zogy, tel=tel)

        # flux_opt and fluxerr_opt are 1-element arrays
        s2n = flux_opt[0] / fluxerr_opt[0]

        # convert fluxes to magnitudes by applying the zeropoint
        keys = ['EXPTIME', 'FILTER', 'DATE-OBS']
        exptime, filt, obsdate = [header[key] for key in keys]

        # get zeropoint from [header]
        if 'PC-ZP' in header:
            zp = header['PC-ZP']
        else:
            log.warning ('keyword PC-ZP not in header of {}; skipping it'
                         .format(filename))
            continue

        # determine object airmass, unless input image is a combined
        # image
        if 'R-V' in header or 'R-COMB-M' in header:
            airmass = 1.0
        else:
            lat = get_par(set_zogy.obs_lat,tel)
            lon = get_par(set_zogy.obs_lon,tel)
            height = get_par(set_zogy.obs_height,tel)
            airmass = get_airmass(ra_deg, dec_deg, obsdate, lat, lon, height)


        log.info ('airmass: {}'.format(airmass))
        ext_coeff = get_par(set_zogy.ext_coeff,tel)[filt]       
        mag_opt, magerr_opt = apply_zp (flux_opt, zp, airmass, exptime, filt,
                                        ext_coeff, fluxerr=fluxerr_opt)


        # read limiting magnitude at pixel coordinates
        filename_limmag = filename.replace('_red.fits', '_red_limmag.fits')

        if os.path.isfile(filename_limmag):

            x_index = int(xcoord-0.5)
            y_index = int(ycoord-0.5)

            t0 = time.time()
            try:
                data_limmag = FITS(filename_limmag)[-1]
                limmag = data_limmag[y_index, x_index][0][0]
            except:
                data_limmag = read_hdulist(filename_limmag)
                limmag = data_limmag[y_index, x_index]

            # convert limmag from 5-sigma to nsigma
            limmag += -2.5*np.log10(nsigma/5)



        row += [mag_opt[0], magerr_opt[0], s2n, limmag]

        # append row to rows
        rows.append(row)



    # initialize output table
    names = ['FILENAME']
    dtypes = ['U100']
    for ikey, key in enumerate(keys2add_list):
        names.append(key)
        dtype_str = dtypes_list[ikey]
        if dtype_str in ['float', 'int', 'bool']:
            dtype = eval(dtype_str)
        else:
            dtype = dtype_str

        dtypes.append(dtype)
            
    names += ['MAG_OPT', 'MAGERR_OPT', 'S2N', 'LIMMAG_{}SIGMA'.format(nsigma)]
    dtypes += [float, float, float, float]
    
    if len(rows) == 0:
        # rows without entries: create empty table
        table = Table(names=names, dtype=dtypes)
    else: 
        table = Table(rows=rows, names=names, dtype=dtypes)
        
        
    # save output fits table
    table.write(fits_out, overwrite=True)



################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Perform forced photometry')

    parser.add_argument('image_list', type=str, 
                        help='Comma-separated list or name of ASCII file '
                        'containing images to be processed')
    parser.add_argument('ra', type=str,
                        help='RA in decimal degrees or colon-separated '
                        'sexagesimal hours')
    parser.add_argument('dec', type=str,
                        help='DEC in decimal degrees or colon-separated '
                        'sexagesimal degrees')
    parser.add_argument('fits_out', type=str,
                        help='output fits table with resulting magnitudes')
    parser.add_argument('--nsigma', type=int, default=5,
                        help='Significance threshold for a detection; default=5')
    par_default = 'mjd-obs,object,filter,exptime,s-seeing,airmass,qc-flag'
    parser.add_argument('--keys2add', type=str, default=par_default,
                        help='Header keyword values to add to output '
                        'table; default: {}'.format(par_default))
    par_default = 'float,U5,U1,float,float,float,U6'
    parser.add_argument('--keys2add_dtypes', type=str, default=par_default,
                        help='Corresponding dtypes; default: {}'
                        .format(par_default))
    parser.add_argument('--telescope', type=str, default='ML1',
                        help='telescope; default=ML1')

    args = parser.parse_args()


    # create list of filenames to process from [image_list]
    if os.path.isfile(args.image_list) and '.fz' not in args.image_list:

        # if [image_list] is a file, read it into a table
        if 'fits' in args.image_list:
            table = Table.read(args.image_list, format='fits')
        else:
            table = Table.read(args.image_list, format='ascii', data_start=0)

        # assume 1st column contains the filenames
        filenames = list(table[table.colnames[0]])

    else:
        filenames = args.image_list.split(',')


    # make sure RA is in decimal degrees
    if ':' in args.ra:
        ra_deg = Angle(args.ra, unit=u.hour).degree
    else:
        ra_deg = float(args.ra)

    # same for DEC
    if ':' in args.dec:
        dec_deg = Angle(args.dec, unit=u.deg).degree
    else:
        dec_deg = float(args.dec)


    # call [force_phot]
    t0 = time.time()
    force_phot (filenames, ra_deg, dec_deg, args.fits_out, nsigma=args.nsigma,
                keys2add=args.keys2add, keys2add_dtypes=args.keys2add_dtypes,
                tel=args.telescope)
    log.info ('time spent in [force_phot]: {:.2f}s'.format(time.time()-t0))

