
import os
import argparse
import datetime
import math

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

import numpy as np

import astropy.io.fits as fits
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval as zscale
from astropy.visualization.wcsaxes import WCSAxes
import astropy.units as u
from astropy.visualization.wcsaxes import add_scalebar

from scipy import ndimage

import matplotlib
# matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('Agg')

# in case google cloud is being used
from google.cloud import storage


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
                        help='OB name (required by ESO); default: pdf')

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
        # make sure it is even
        if size_pix % 2 != 0:
            size_pix += 1

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


    # target indicators

    # center
    x1 = y1 = size_pix/2
    # offset in fraction of image size
    offset = size_pix / 20
    # length in fraction of image size
    length = size_pix / 20
    # width in fraction of length, at least 1.5 pixels
    width = min(max(size_pix/100, 1), 3)
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

def read_hdulist (fits_file, get_data=True, get_header=False,
                  ext_name_indices=None, dtype=None, memmap=True):

    """Function to read the data (if [get_data] is True) and/or header
    (if [get_header] is True) of the input [fits_file].  The fits file
    can be an image or binary table, and can be compressed (with the
    compressions that astropy.io can handle, such as .gz and .fz). If
    [ext_name_indices] is defined, which can be an integer, a string
    matching the extension's keyword EXTNAME or a list or numpy array
    of integers, those extensions are retrieved.

    """

    if isfile(fits_file):
        fits_file_read = fits_file

    else:
        # if fits_file does not exist, look for compressed versions or
        # files without the .fz or .gz extension
        if isfile('{}.fz'.format(fits_file)):
            fits_file_read = '{}.fz'.format(fits_file)
        elif isfile(fits_file.replace('.fz','')):
            fits_file_read = fits_file.replace('.fz','')
        elif isfile('{}.gz'.format(fits_file)):
            fits_file_read = '{}.gz'.format(fits_file)
        elif isfile(fits_file.replace('.gz','')):
            fits_file_read = fits_file.replace('.gz','')
        else:
            raise FileNotFoundError ('file not found: {}'.format(fits_file))



    with fits.open(fits_file_read, memmap=memmap) as hdulist:

        n_exts = len(hdulist)

        # if [ext_name_indices] is a range, or list or numpy ndarray
        # of integers, loop over these extensions and concatenate the
        # data into one astropy Table; it is assumed the extension
        # formats are identical to one another - this is used to read
        # specific extensions from e.g. the calibration catalog.
        if type(ext_name_indices) in [list, range, np.ndarray]:

            for i_ext, ext in enumerate(ext_name_indices):

                # get header from first extension as they should be
                # all identical, except for NAXIS2 (nrows)
                if get_header and i_ext==0:
                    header = hdulist[ext].header

                if get_data:
                    # read extension
                    data_temp = hdulist[ext].data
                    # convert to table, as otherwise concatenation of
                    # extensions below using [stack_arrays] is slow
                    data_temp = Table(data_temp)
                    # could also read fits extension into Table directly,
                    # but this is about twice as slow as the 2 steps above
                    #data_temp = Table.read(fits_file_read, hdu=ext)
                    if i_ext==0:
                        data = data_temp
                    else:
                        #data = stack_arrays((data, data_temp),asrecarray=True,
                        #                    usemask=False)
                        # following does not work if data is a fitsrec
                        # array and the array contains boolean fields, as
                        # these are incorrectly converted; therefore the
                        # conversion to a Table above
                        data = np.concatenate([data, data_temp])
                        # could also use the following instead, but
                        # since the above is working ...
                        #data = vstack([data, data_temp])


                    log.info ('added {} rows from extension {} of {}'
                              .format(len(data_temp), ext, fits_file))


        else:
            # otherwise read the extension defined by [ext_name_indices]
            # or simply the last extension
            if type(ext_name_indices) in [int, str]:
                ext = ext_name_indices
            else:
                ext = n_exts-1

            if get_data:
                data = hdulist[ext].data
                # convert to [dtype] if it is defined
                if dtype is not None:
                    data = data.astype(dtype, copy=False)

            if get_header:
                header = hdulist[ext].header


    # return data and header depending on whether [get_data]
    # and [get_header] are defined or not
    if get_data:
        if get_header:
            return data, header
        else:
            return data
    else:
        if get_header:
            return header
        else:
            log.error ('parameters [get_data] and [get_header] are both False '
                       'in function [zogy.read_hdlist]; returning None'
                       )
            return None


################################################################################

def get_index_around_xy (ysize, xsize, ycoord, xcoord, size):

    """Function to retrieve indices around pixel coordinates [ycoord,
    xcoord] in original image of size (ysize, xsize) and a thumbnail
    image of size (size, size) onto which the original pixel values
    will be projected. Normally the shapes of the two returned indices
    will be (size, size), but not if the pixel coordinates are near
    the original image edge.

    The function also returns the pixel coordinates in the thumbnail
    image corresponding to ycoord, xcoord in the original image.

    N.B.: if (ycoord, xcoord) are pixel coordinates, then pixel
    coordinate (int(ycoord), int(xcoord)) will correspond to pixel
    coordinate (int(size/2), int(size/2)) in the thumbnail image. If
    instead they are pixel indices (i.e.  int(pixel coordinate) - 1),
    then index (ycoord, xcoord) will correspond to pixel index
    (int(size/2), int(size/2)) in the thumbnail index.

    """


    # size is assumed to be even!
    xpos = int(xcoord)
    ypos = int(ycoord)
    hsize = int(size/2)

    # if footprint is partially off the image, just go ahead
    # with the pixels on the image
    y1 = max(0, ypos-hsize)
    x1 = max(0, xpos-hsize)
    y2 = min(ysize, ypos+hsize)
    x2 = min(xsize, xpos+hsize)
    index = tuple([slice(y1,y2),slice(x1,x2)])

    # also determine corresponding indices of thumbnail image, which
    # will not be [0:size, 0:size] if an object is near the image edge
    y1_tn = max(0, hsize-ypos)
    x1_tn = max(0, hsize-xpos)
    y2_tn = min(size, size-(ypos+hsize-ysize))
    x2_tn = min(size, size-(xpos+hsize-xsize))
    index_tn = tuple([slice(y1_tn,y2_tn),slice(x1_tn,x2_tn)])

    # pixel coordinates in the thumbnail image corresponding to
    # ycoord, xcoord in the original image
    ycoord_tn = ycoord - y1
    xcoord_tn = xcoord - x1

    return index, index_tn, ycoord_tn, xcoord_tn


################################################################################

def orient_data (data, header, header_out=None, MLBG_rot90_flip=False,
                 pixscale=0.564, tel=None):

    """Function to remap [data] from the CD matrix defined in [header] to
    the CD matrix taken from [header_out].  If the latter is not
    provided the output orientation will be North up, East left.

    If [MLBG_rot90_flip] is switched on and the data is from MeerLICHT or
    BlackGEM, the data will be oriented within a few degrees from
    North up, East left while preserving the pixel values in the new,
    *remapped* reference, D and Scorr images.

    """

    # rotation matrix:
    # R = [[dx * cos(theta),  dy * -sin(theta)],
    #      [dx * sin(theta),  dy * cos(theta)]]
    # with theta=0: North aligned with positive y-axis
    # and East with the positive x-axis (RA increases to the East)
    #
    # N.B.: np.dot(R, [[x], [y]]) = np.dot([x,y], R.T)
    #
    # matrices below are defined using the (WCS) header keywords
    # CD?_?:
    #
    # [ CD1_1  CD2_1 ]
    # [ CD1_2  CD2_2 ]
    #
    # orient [data] with its orientation defined in [header] to the
    # orientation defined in [header_out]. If the latter is not
    # provided, the output orientation will be North up, East left.

    # check if input data is square; if it is not, the transformation
    # will not be done properly.
    assert data.shape[0] == data.shape[1]

    # define data CD matrix, assumed to be in [header]
    CD_data = read_CD_matrix (header)

    # determine output CD matrix, either from [header_out] or North
    # up, East left
    if header_out is not None:
        CD_out = read_CD_matrix (header_out)
    else:
        # define the CD matrix with North up and East left, using the
        # input pixel scale
        cdelt = pixscale/3600
        CD_out = np.array([[-cdelt, 0], [0, cdelt]])


    # check if values of CD_data and CD_out are similar
    CD_close = [math.isclose(CD_data[i,j], CD_out[i,j], rel_tol=1e-3)
                for i in range(2) for j in range(2)]


    if np.all(CD_close):

        #log.info ('data CD matrix already similar to CD_out matrix; '
        #          'no need to remap data')

        # if CD matrix values are all very similar, do not bother to
        # do the remapping
        data2return = data

    elif MLBG_rot90_flip and tel in ['ML1', 'BG2', 'BG3', 'BG4']:

        #log.info ('for ML/BG: rotating data by exactly 90 degrees and for '
        #          'ML also flip left/right')

        # rotate data by exactly 90 degrees counterclockwise (when
        # viewing data with y-axis increasing to the top!) and for ML1
        # also flip in the East-West direction; for ML/BG this will
        # result in an image within a few degrees of the North up,
        # East left orientation while preserving the original pixel
        # values of the new, *remapped* reference, D and Scorr images.

        data2return = np.rot90(data, k=-1)
        if tel=='ML1':
            data2return = np.fliplr(data2return)

        # equivalent operation: data2return = np.flipud(np.rot90(data))

    else:

        #log.info ('remapping data from input CD matrix: {} to output CD '
        #          'matrix: {}'.format(CD_data, CD_out))

        # transformation matrix, which is the dot product of the
        # output CD matrix and the inverse of the data CD matrix
        CD_data_inv = np.linalg.inv(CD_data)
        CD_trans = np.dot(CD_out, CD_data_inv)

        # transpose and flip because [affine_transform] performs
        # np.dot(matrix, [[y],[x]]) rather than np.dot([x,y], matrix)
        matrix = np.flip(CD_trans.T)

        # offset, calculated from
        #
        # [xi - dxi, yo - dyo] = np.dot( [xo - dxo, yo - dyo], CD_trans )
        #
        # where xi, yi are the input coordinates corresponding to the
        # output coordinates xo, yo in data and dxi/o, dyi/o are the
        # corresponding offsets from the point of
        # rotation/transformation, resulting in
        #
        # [xi, yi] = np.dot( [xo, yo], CD_trans ) + offset
        # with
        # offset = -np.dot( [dxo, dyo], CD_trans ) + [dxi, dyi]
        # setting [dx0, dy0] and [dxi, dyi] to the center
        center = (np.array(data.shape)-1)/2
        offset = -np.dot(center, np.flip(CD_trans)) + center

        # infer transformed data
        data2return = ndimage.affine_transform(data, matrix, offset=offset,
                                               mode='nearest')


    return data2return


################################################################################

def read_CD_matrix (header):

    if ('CD1_1' in header and 'CD1_2' in header and
        'CD2_1' in header and 'CD2_2' in header):

        data2return = np.array([[header['CD1_1'], header['CD2_1']],
                                [header['CD1_2'], header['CD2_2']]])
    else:
        msg = 'one of CD?_? keywords not in header'
        log.critical(msg)
        raise KeyError(msg)
        data2return = None


    return data2return


################################################################################

def get_bucket_name (path):

    """infer bucket- and filename from [path], which is expected
       to be gs://[bucket name]/some/path/file or [bucket
       name]/some/path/file; if [path] starts with a forward slash,
       empty strings will be returned"""

    bucket_name = path.split('gs://')[-1].split('/')[0]
    if len(bucket_name) > 0:
        # N.B.: returning filename without the starting '/'
        bucket_file = path.split(bucket_name)[-1][1:]
    else:
        bucket_file = ''

    return bucket_name, bucket_file


################################################################################

def isfile (filename):

    if filename[0:5] == 'gs://':

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (filename)
        # N.B.: bucket_file should not start with '/'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(bucket_file)
        return blob.exists()

    else:

        return os.path.isfile(filename)


################################################################################

if __name__ == "__main__":
    main()
