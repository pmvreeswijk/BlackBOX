
import os
import argparse
import datetime

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

from zogy import get_par, read_hdulist, get_index_around_xy, orient_data, isfile
import set_zogy
set_zogy.verbose=False

import numpy as np

import astropy.io.fits as fits
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval as zscale
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from astropy.visualization.wcsaxes import add_scalebar

import matplotlib
# matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('Agg')



################################################################################

# from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
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

def main():

    parser = argparse.ArgumentParser(description='Prepare finding chart')

    parser.add_argument('target_ra', type=str,
                        help='target RA (ICRS) in decimal degrees '
                        'or sexagesimal hours; no default')

    parser.add_argument('target_dec', type=str,
                        help='target DEC (ICRS) in decimal or sexagesimal '
                        'degrees; if negative, put a space before the string '
                        'value and add quotes, e.g. \' -10:45:32.3\'; no default')

    parser.add_argument('fits_red', type=str,
                        help='BG/ML reduced filename (e.g. '
                        '/idia/projects/meerlicht/ML1/red/2022/12/31/ML1_20221231_184851_red.fits.fz '
                        'or gs://blackgem-red/BG4/2024/01/13/BG4_20240114_060714_red.fits.fz); '
                        'full path required; no default')

    parser.add_argument('--target_name', type=str, default=None,
                        help='target name; default: None')

    parser.add_argument('--size_arcmin', type=float, default=3,
                        help='finding chart size in arcmin; default: 3')

    parser.add_argument('--cmap', type=str, default='gray_r',
                        help='colour map to use; default: gray_r')

    parser.add_argument('--run_id', type=str, default=None,
                        help='Observing Run ID (required by ESO); default: None')

    parser.add_argument('--pi_name', type=str, default=None,
                        help='PI name (required by ESO); default: None')

    parser.add_argument('--ob_name', type=str, default=None,
                        help='OB name (required by ESO); default: None')

    parser.add_argument('--output_format', type=str, choices=['pdf', 'jpg'],
                        default='pdf',
                        help='OB name (required by ESO); default: jpg')

    args = parser.parse_args()


    # convert input [ra] and [dec] to decimal degrees (float)
    if ':' in args.target_ra:
        # if RA in sexagesimal hours, convert to decimal degrees
        ra_deg = Angle(args.target_ra, unit=u.hour).degree
        ra_sexa = args.target_ra
    else:
        ra_deg = float(args.target_ra)
        ra_sexa = Angle(ra_deg/15., unit=u.hour).to_string(sep=':', precision=2)

    if ':' in args.target_dec:
        # if DEC in sexagesimal degrees, convert to decimal degrees
        dec_deg = Angle(args.target_dec, unit=u.deg).degree
        dec_sexa = args.target_dec
    else:
        dec_deg = float(args.target_dec)
        dec_sexa = Angle(dec_deg, unit=u.deg).to_string(sep=':', precision=2)


    # check if input parameter [fits_red] is provided
    if args.fits_red is None:
        # download image from SDSS, PanSTARRS, Blanco, DSS
        mlbg_image = False
        filt = None
        limmag = None

    else:

        # previously constructed the full name from short name, but
        # now requiring input [fits_red] to contain full path
        filename = args.fits_red

        # if image not found, exit with an error message
        if not isfile (filename):
            log.error ('filename {} not found; exiting'.format(filename))
            raise SystemExit


        # this is a MeerLICHT or BlackGEM image
        mlbg_image = True


        # full name of telescope used
        tel = filename.split('/')[-1][0:3]
        dict_tel = {'ML1': 'MeerLICHT', 'BG2': 'BlackGEM-2',
                    'BG3': 'BlackGEM-4', 'BG4': 'BlackGEM-4'}
        telescope = dict_tel[tel]


        # read image data and header
        data, header = read_hdulist(filename, get_header=True, memmap=True)
        ysize, xsize = data.shape


        # convert input RA,DEC to pixel coordinates
        xcoord, ycoord = WCS(header).all_world2pix(ra_deg, dec_deg, 1)


        # if coordinates not within image limits, exit with an error message
        if (xcoord < 0.5 or xcoord > xsize+0.5 or
            ycoord < 0.5 or ycoord > ysize+0.5):
            log.error ('pixel coordinates (x={}, y={}) not on image; exiting'
                       .format(int(xcoord), int(ycoord)))
            raise SystemExit


        # extract square image with the size of [args.size_arcmin]
        pixscale = 0.5642
        size_pix = int(args.size_arcmin * 60 / pixscale + 1)
        data_out = np.zeros((size_pix, size_pix), dtype='float32')


        # get index around x,y position using function [get_index_around_xy]
        index_full, index_out, __, __ = (
            get_index_around_xy (ysize, xsize, ycoord, xcoord, size_pix))


        # project original image data to output image
        data_out[index_out] = data[index_full]


        # orient to North up, East left
        data_out = orient_data (data_out, header, MLBG_rot90_flip=True,
                                pixscale=pixscale, tel=tel)

        # header keywords
        filt = header['FILTER']
        limmag = header['LIMMAG']



    # convert data_out to PDF or jpeg
    fig = plt.figure(figsize=(8.27,11.69))

    wcs = WCS(header)
    # tried wcs projection in different ways, but no luck
    #plt.subplot(projection=wcs)
    #ax = fig.add_subplot(1, 1, 1, projection=wcs)
    #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=wcs)
    #plt.axes([0.1, 0.1, 0.8, 0.8], projection=wcs)

    #ax = plt.subplot()
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])

    vmin, vmax = zscale().get_limits(data_out)
    ax.imshow(data_out, vmin=vmin, vmax=vmax, cmap=args.cmap, origin='lower')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel ('North', fontsize = 24)
    ax.set_ylabel ('East', fontsize = 24)

    # hide tick marks
    ax.set_xticks([])
    ax.set_yticks([])

    # labels dictionary
    labels = {}
    if args.target_name is not None:
        labels['Target'] = args.target_name

    labels['RA (ICRS)'] = '{} ({:.4f} deg)'.format(ra_sexa, ra_deg)
    labels['DEC (ICRS)'] = '{} ({:.4f} deg)'.format(dec_sexa, dec_deg)

    if args.fits_red is not None:
        fits_red_short = args.fits_red.split('/')[-1].split('.fits')[0]
        labels['Image'] = '{}'.format(fits_red_short)

    if filt is not None:
        labels['Filter'] = filt

    if limmag is not None:
        labels['Limiting magnitude'] = '{:.2f} (5$\sigma$)'.format(limmag)

    if args.run_id is not None:
        labels['Observing Run ID'] = args.run_id

    if args.pi_name is not None:
        labels['PI name'] = args.pi_name

    if args.ob_name is not None:
        labels['OB name'] = args.ob_name

    labels['Finding chart size'] = ('{:.1f}x{:.1f} arcmin'
                                    .format(args.size_arcmin, args.size_arcmin))

    yloc = 0.25
    for key in labels.keys():
        xloc = 0.49
        ax.annotate ('{}:'.format(key), (xloc, yloc), fontsize=12,
                     xycoords='figure fraction', ha='right')
        #xloc = 1-xloc
        xloc = 0.51
        ax.annotate ('{}'.format(labels[key]), (xloc, yloc), fontsize=12,
                     xycoords='figure fraction', ha='left')
        yloc -= 0.02


    # title
    yloc = 0.95
    title = '{} finding chart'.format(telescope)
    if args.target_name is not None:
        title += ' for {}'.format(args.target_name)

    ax.annotate (title, (xloc, yloc), fontsize=18,
                 xycoords='figure fraction', ha='center', weight='bold')


    # scale bar
    #add_scalebar (ax, 1*u.arcmin, label=None, corner='top left')
    scale = np.array([1,2,3,5,10,30])
    bar_size_init = args.size_arcmin/5
    unit = 'arcmin'
    if bar_size_init < 0.5:
        unit = 'arcsec'
        bar_size_init *= 60

    # pick closest possible scale
    index = np.argmin(abs(scale - bar_size_init))
    bar_size = scale[index]
    bar_size_pix = bar_size/pixscale
    unit_symbol = '\'\''
    if unit == 'arcmin':
        bar_size_pix *= 60
        unit_symbol = '\''

    size_pix = args.size_arcmin*60/pixscale
    bar_size_fig = 0.8*(bar_size_pix/size_pix)

    x1 = 0.11
    x2 = x1 + bar_size_fig
    y1 = y2 = 0.3
    ax.annotate('', xy=(x2,y2), xycoords='figure fraction',
                xytext=(x1,y1), textcoords='figure fraction',
                arrowprops=dict(arrowstyle= '|-|',
                                color='black', lw=2,
                                mutation_scale=5))
    ax.annotate ('{}{}'.format(bar_size, unit_symbol),
                 (x1+bar_size_fig/2, y1-0.02),
                 fontsize=15, xycoords='figure fraction', ha='center')

    x1 = y1 = size_pix/2
    offset = 10/pixscale
    length = size_pix/20
    width = max(length/15, 2)
    ax.add_patch(patches.Rectangle ((x1+offset, y1-width/2), length, width,
                                    fc='black', ec='white'))
    ax.add_patch(patches.Rectangle ((x1-width/2, y1+offset), width, length,
                                    fc='black', ec='white'))


    if args.target_name is not None:
        fig_name = '{}_findingchart.{}'.format(args.target_name,
                                         args.output_format)
    else:
        fig_name = ('RA{:.3f}_DEC{:.3f}_finder.{}'
                    .format(ra_deg, dec_deg, args.output_format))

    #{}_findingchart.pdf'.format(filename.split('/')[-1].split('.fits')[0])
    plt.savefig(fig_name)
    #plt.show()
    plt.close()


################################################################################

if __name__ == "__main__":
    main()
