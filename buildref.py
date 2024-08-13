import os
import subprocess
import shutil
import argparse
import traceback
import collections
import itertools
import re
import fnmatch


#import multiprocessing as mp
#mp_ctx = mp.get_context('spawn')


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

# set_br is needed in definition of OMP_NUM_THREADS below
import set_buildref as set_br

# setting environment variable OMP_NUM_THREADS to number of threads;
# use value from environment variable SLURM_CPUS_PER_TASK if it is
# defined, otherwise set_br.nthreads; needs to be done before numpy is
# imported in [zogy]
os.environ['OMP_NUM_THREADS'] = str(os.environ.get('SLURM_CPUS_PER_TASK',
                                                   set_br.nthreads))

import numpy as np

import astropy.io.fits as fits
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.coordinates import Angle
from astropy import units as u

from scipy import ndimage

import zogy
import set_zogy
import blackbox as bb
import set_blackbox as set_bb
import qc


__version__ = '0.9.3'


################################################################################

def buildref (telescope=None, fits_hdrtable_list=None, date_start=None,
              date_end=None, field_IDs=None, filters=None, ascii_inputfiles=None,
              go_deep=None, qc_flag_max=None, seeing_max=None, skip_zogy=False,
              make_colfig=False, filters_colfig='iqu', mode_ref=False,
              results_dir=None, extension=None, keep_tmp=None):


    """Module to consider one specific or all available field IDs within a
    specified time range, and to combine the available images of that
    field ID in one or all filters, using those images that satisfy
    the quality flag and seeing constraint. The combining of the
    images is done using the function imcombine.

    The resulting reference image is put through zogy.py as the
    reference image and the corresponding reference directory is
    prepared.

    To do for BlackGEM:

    - needs to be able to combine images from all BGs, but also keep
      the option of combining images from specific telescope; so add
      boolean input parameter mix_BGs?

    + need to make compatible with google cloud

    + output name is different: [tel]_[field ID]_[filter]_[creation date]_..
      for both ML and BG. Since for BG, tel=BG, make tel=ML for ML1?

    - for BG: reference images have their separate bucket: gs://blackgem-ref
      what to do for ML? Keep the same as it is, except update the filenames?

    - for BG: if ref image already exists, the existing one needs to
      be moved to a separated bucket: gs://blackgem-ref-old. What to
      do for ML in that case?

    """

    global tel, max_qc_flag, max_seeing, start_date, end_date
    global time_refstart, ext, deep, dir_results, ref_mode
    tel = telescope
    max_qc_flag = qc_flag_max
    max_seeing = seeing_max
    start_date = date_start
    end_date = date_end
    ext = extension
    deep = go_deep
    dir_results = results_dir
    ref_mode = mode_ref


    # define number of processes or tasks [nproc]; when running on the
    # ilifu/google slurm cluster the environment variable SLURM_NTASKS
    # should be set through --ntasks-per-node in the sbatch script;
    # otherwise use the value from the set_br settings file
    nproc = int(os.environ.get('SLURM_NTASKS', get_par(set_br.nproc,tel)))

    # update nthreads in set_bb with value of environment variable
    # OMP_NUM_THREADS set at the top of this module
    set_br.nthreads = int(os.environ.get('OMP_NUM_THREADS', set_br.nthreads))


    if keep_tmp is not None:
        set_br.keep_tmp = str2bool(keep_tmp)


    # record starting time to add to header
    time_refstart = Time.now().isot


    log.info ('building reference images')
    log.info ('number of processes: {}'.format(nproc))
    log.info ('number of threads:   {}'.format(set_br.nthreads))
    log.info ('telescope:           {}'.format(telescope))
    log.info ('date_start:          {}'.format(date_start))
    log.info ('date_end:            {}'.format(date_end))
    log.info ('field_IDs:           {}'.format(field_IDs))
    log.info ('filters:             {}'.format(filters))
    log.info ('qc_flag_max:         {}'.format(qc_flag_max))
    log.info ('seeing_max:          {}'.format(seeing_max))
    #log.info ('make_colfig:         {}'.format(make_colfig))
    #if make_colfig:
    #    log.info ('filters_colfig:      {}'.format(filters_colfig))

    log.info ('mode_ref:            {}'.format(mode_ref))
    if not mode_ref:
        log.info ('results_dir          {}'.format(results_dir))
        log.info ('subfolder extension: {}'.format(extension))



    t0 = time.time()


    # read list of tables with filenames and relevant header keywords
    # ---------------------------------------------------------------
    # previously also included possiblity to prepare this table, but
    # not needed anymore as header tables are ready to be used
    if fits_hdrtable_list is None:

        # get base folder name of header tables from set_blackbox
        hdrtables_dir = get_par(set_bb.hdrtables_dir,tel)

        # refer to the existing header tables for both ML and BG
        if tel == 'ML1':

            table_name = '{}/{}_headers_cat.fits'.format(hdrtables_dir, tel)
            fits_hdrtable_list = [table_name]

        else:

            # for BG, loop telescopes and add header table if needed
            fits_hdrtable_list = []
            for tel_tmp in ['BG2', 'BG3', 'BG4']:
                if tel in tel_tmp:
                    table_name = '{}/{}_headers_cat.fits'.format(hdrtables_dir,
                                                                 tel_tmp)
                    fits_hdrtable_list.append(table_name)



    # read header fits files into table
    for it, fits_table in enumerate(fits_hdrtable_list):

        if zogy.isfile(fits_table):

            log.info ('reading header table: {}'.format(fits_table))
            table_tmp = Table.read(fits_table)
            if it==0:
                table = table_tmp
            else:
                # stack tables
                table = vstack([table, table_tmp])

        else:
            log.warning ('{} not found'.format(fits_table))



    # check if table contains any entries
    if len(table)==0:
        log.error ('no entries in tables in [fits_hdrtable_list]; exiting')
        logging.shutdown()
        return

    else:
        log.info ('{} files in input table(s)'.format(len(table)))



    # make sure FILENAME column contains the image name instead of
    # catalog
    table['FILENAME'] = ['{}_red.fits.fz'.format(fn.split('_red')[0])
                         for fn in table['FILENAME']]


    # if specific files to be used are listed in [ascii_inputfiles],
    # limit the table to those
    if ascii_inputfiles is not None:

        # read ascii file
        table_in = Table.read(ascii_inputfiles, format='ascii', data_start=0,
                              names=['FILENAME'])

        # select corresponding entries from table
        filenames_short = [fn.split('/')[-1].split('_red.fits.fz')[0]
                           for fn in table['FILENAME']]
        mask = np.array([any(fn in el for el in table_in['FILENAME'])
                         for fn in filenames_short])
        table = table[mask]
        log.info ('{} files left (ascii_inputfiles cut)'.format(len(table)))



    # filter table entries based on telescope
    # ---------------------------------------
    mask = np.array([tel in fn for fn in table['FILENAME']])
    table = table[mask]
    log.info ('{} files left ({} telescope cut)'.format(len(table), tel))


    # filter table entries based on date, field_ID, filter, qc-flag and seeing
    # ------------------------------------------------------------------------
    if date_start is not None or date_end is not None:

        # filter this list by converting the date and time contained in
        # the filenames to MJD and check if this is consistent with the
        # range specified by the input [date_start] and [date_end]
        mjd_start = set_date (date_start)
        mjd_end = set_date (date_end, start=False)
        # convert dates and times in filenames to MJDs (accurate to the second)
        mjd_files = np.array([bb.date2mjd(f.split('/')[-1].split('_')[1],
                                          time_str=f.split('/')[-1].split('_')[2]
                                          ) for f in table['FILENAME']])
        # mask of files to include
        if mjd_end >= mjd_start:
            if mjd_start == mjd_end:
                mjd_end += 1

            mask = ((mjd_files >= mjd_start) & (mjd_files <= mjd_end))

        else:
            # if mjd_start is greater than mjd_end, then select images
            # beyond mjd_start and before mjd_end
            mask = ((mjd_files >= mjd_start) | (mjd_files <= mjd_end))

        # select relevant table entries
        table = table[mask]
        log.info ('{} files left (date_start/end cut)'.format(len(table)))



    # if object (field ID) is specified, which can include the unix
    # wildcards * and ?, select only images with a matching object
    # string; field_IDs can also be an ascii file with the field ID(s)
    # listed in the 1st column
    filter_list = None
    if field_IDs is not None:

        # check if it's a file
        if zogy.isfile(field_IDs):

            # read ascii table
            table_ID = Table.read(field_IDs, format='ascii', data_start=0)
            # table can contain 1 or 2 columns and can therefore not
            # pre-define column names, while with data_start=0 the entries
            # on the first line are taken as the column names
            cols = table_ID.colnames

            # list with field IDs
            field_ID_list = list(table_ID[cols[0]].astype(str))

            # define list of filters if 2nd column is defined
            if len(cols)>1:
                filter_list = list(table_ID[cols[1]])

        else:

            # comma-split input string field_IDs into list; if no comma
            # is present, the list will contain a single entry
            field_ID_list = field_IDs.split(',')


        # check that the leading zeros are present for field IDs with
        # digits only
        for i_field, field_ID in enumerate(field_ID_list):
            if field_ID.isdigit() and len(field_ID)!=5:
                field_ID_list[i_field] = '{:0>5}'.format(field_ID)


        # prepare mask where presence of (header) table object entry
        # is checked against any of the field IDs in field_ID_list;
        # this mask will contain len(table) * len(field_ID_list)
        # entries
        mask = [fnmatch.fnmatch('{:0>5}'.format(obj), field_ID)
                for field_ID in field_ID_list
                for obj in table['OBJECT']]
        # reshape the mask to shape (len(field_ID_list, len(table))
        mask = np.array(mask).reshape(len(field_ID_list), len(table))
        # OR-combine the mask along axis=0 (if image object matches
        # any of the input field_IDs, use it)
        mask = np.any(mask, axis=0)
        table = table[mask]
        log.info ('{} files left (FIELD_ID cut)'.format(len(table)))



    # if filter(s) is specified, select only images with filter(s)
    # specified
    if filters is not None:
        #mask = [table['FILTER'][i] in filters for i in range(len(table))]
        mask = [filt in filters for filt in table['FILTER']]
        table = table[mask]
        log.info ('{} files left (FILTER cut)'.format(len(table)))


    # if qc_flag_max is specified, select only images with QC-FLAG of
    # qc_flag_max and better
    if len(table)>0 and qc_flag_max is not None:
        qc_col = ['green', 'yellow', 'orange', 'red']
        # redefine qc_col up to and including qc_flag_max
        qc_col = qc_col[0:qc_col.index(qc_flag_max)+1]

        mask_green = [table['QC-FLAG'][i].strip()=='green'
                      for i in range(len(table))]
        mask_yellow = [table['QC-FLAG'][i].strip()=='yellow'
                       for i in range(len(table))]
        mask_orange = [table['QC-FLAG'][i].strip()=='orange'
                       for i in range(len(table))]
        mask_red = [table['QC-FLAG'][i].strip()=='red'
                    for i in range(len(table))]
        log.info ('number of green: {}, yellow: {}, orange: {}, red: {}'
                  .format(np.sum(mask_green), np.sum(mask_yellow),
                          np.sum(mask_orange), np.sum(mask_red)))

        # strip table color from spaces
        mask = [table['QC-FLAG'][i].strip() in qc_col for i in range(len(table))]
        table = table[mask]
        log.info ('{} files left (QC-FLAG cut)'.format(len(table)))



    # if max_seeing is specified, select only images with the same or
    # better seeing
    if max_seeing is not None:
        mask = (table['S-SEEING'] <= max_seeing)
        table = table[mask]
        log.info ('{} files left (SEEING cut)'.format(len(table)))



    # ensure that telescope is tracking
    mask = (table['ISTRACKI'] == True)
    table = table[mask]
    log.info ('{} files left (telescope tracking cut)'.format(len(table)))


    # if centering is set to 'grid' in buildref settings file, read
    # the file that contains the ML/BG field grid definition, that
    # will be used to fill [radec_list] in the loop below
    center_type = get_par(set_br.center_type,tel)
    if center_type == 'grid':
        # read from grid definition file located in ${BBHOME}/CalFiles
        mlbg_fieldIDs = get_par(set_bb.mlbg_fieldIDs,tel)
        #table_grid = ascii.read(mlbg_fieldIDs, names=['ID', 'RA', 'DEC'],
        #                        data_start=0)
        table_grid = Table.read(mlbg_fieldIDs)



    # for table entries that have survived the cuts, prepare the list
    # of imagelists with the accompanying lists of field_IDs, filters,
    # image centers and sizes
    list_of_imagelists = []
    obj_list = []
    filt_list = []
    radec_list = []
    imagesize_list = []
    nfiles_list = []
    limmag_proj_list = []
    combine_type_list = []
    A_swarp_list = []
    nsigma_clip_list = []


    # unique objects in table
    objs_uniq = np.unique(table['OBJECT'])
    # if field_IDs were provided in input file, use those instead
    if field_IDs is not None and os.path.isfile(field_IDs):
        objs_uniq = field_ID_list


    # unique filters in table
    filts_uniq = np.unique(table['FILTER'])


    # various settings file parameters used in loop
    imagesize_type = get_par(set_br.imagesize_type,tel)
    # nominal image size
    ysize = get_par(set_bb.ny,tel) * get_par(set_bb.ysize_chan,tel)
    # pixel scale
    pixscale_out = get_par(set_br.pixscale_out,tel)
    # background box size
    bkg_size = get_par(set_zogy.bkg_boxsize,tel)


    # minimum number of images required to produce co-add
    nmin = get_par(set_br.nimages_min,tel)

    # maximum number of images to be used
    nmax = get_par(set_br.nimages_max,tel)


    # loop objects
    for n_obj, obj in enumerate(objs_uniq):

        # skip fields '00000' and those beyond 20,000
        #if int(obj) == 0 or int(obj) >= 20000:
        if int(obj) == 0 or int(obj) >= 20000:
            continue

        # table mask of this particular field_ID
        mask_obj = (table['OBJECT'] == obj)

        # if mask_obj is empty, continue
        if np.sum(mask_obj) == 0:
            continue

        # determine image center based on [center_type]
        if center_type == 'grid':
            # for 'grid' centering, let [radec] refer to a tuple pair
            # containing the RA and DEC coordinates
            mask_grid = (table_grid['field_id'].astype(int) == int(obj))
            if np.sum(mask_grid) > 0:
                radec = (table_grid['ra_c'][mask_grid][0],
                         table_grid['dec_c'][mask_grid][0])
            else:
                log.error ('field ID/OBJECT {} not present in ML/BG '
                           'grid definition file {}; skipping it'
                           .format(obj, mlbg_fieldIDs))
                continue


        elif center_type == 'median_field':
            # let [radec] refer to a tuple pair containing the median
            # RA-CNTR and DEC-CNTR for all images of a particular
            # field
            ra_cntr_med = np.median(table[mask_obj]['RA-CNTR'])
            dec_cntr_med = np.median(table[mask_obj]['DEC-CNTR'])
            radec = (ra_cntr_med, dec_cntr_med)


        elif center_type == 'median_filter':
            # set radec tuple to None values, so that median position
            # of the images combined is used as the center for the
            # resulting image - that is done inside [imcombine]
            radec = (None, None)



        # determine image size based on [imagesize_type]
        if imagesize_type == 'input':
            # same as input
            #image_size = '{},{}'.format(xsize, ysize)
            image_size = ysize


        elif imagesize_type == 'all_field':
            image_size = calc_imsize (table[mask_obj]['RA-CNTR'],
                                      table[mask_obj]['DEC-CNTR'],
                                      radec[0], radec[1],
                                      ysize, bkg_size, pixscale_out)


        elif imagesize_type == 'all_filter':
            # in this case, imagesize is determined inside [imcombine]
            image_size = None


        # filters to process for this object; if [filter_list]
        # provided through input file is defined, use that single
        # filter; otherwise use unique filters in table (which is
        # already filtered based on input [filters]
        if filter_list is not None:
            filts_toloop = filter_list[n_obj]
        else:
            filts_toloop = filts_uniq


        # loop filters
        for filt in filts_toloop:

            # table mask of entries with this object and filter
            # combination
            mask = (mask_obj & (table['FILTER'] == filt))
            nfiles = np.sum(mask)
            log.info ('{} files left for {} in filter {}'
                      .format(nfiles, obj, filt))

            # if too few files left, continue
            if nfiles < nmin:
                log.warning ('fewer images ({}) available than the minimum '
                             'number required ({}); not creating co-add'
                             'for {} in filter {}'
                             .format(nfiles, nmin, obj, filt))
                continue


            # default values for A_swarp and nsigma_clip; these are
            # not relevant in case combine_type is not set to clipped,
            # but their values are required to pass on to [imcombine]
            A_swarp = 0.3
            nsigma_clip = 2.5

            combine_type = get_par(set_br.combine_type,tel).lower()
            if combine_type == 'clipped':

                # pick images based on maximum spread in seeing values
                max_spread = get_par(set_br.max_spread_seeing,tel)
                seeing = np.array(table[mask]['S-SEEING'])
                mask_use = pick_images (seeing, max_spread=max_spread)

                # determine A to use to ensure bright stars are not
                # being clipped
                imagelist = list(table['FILENAME'][mask][mask_use])

                # A_range to consider
                tmplist = get_par(set_br.A_range,tel)
                if len(tmplist)==2: tmplist.append(1)
                A_range= np.arange(tmplist[0], tmplist[1], tmplist[2])

                # nsigma_range to consider
                tmplist = get_par(set_br.nsigma_range,tel)
                if len(tmplist)==2: tmplist.append(1)
                nsigma_range= np.arange(tmplist[0], tmplist[1], tmplist[2])

                # set size of PSF image to extract to the maximum allowed by
                # [size_vignet] in the zogy settings file
                psf_size = get_par(set_zogy.size_vignet,tel)

                Nlimit=0
                mask_imagelist, A_swarp, nsigma_clip = get_A_swarp (
                    imagelist, A_range=A_range,
                    nsigma_range=nsigma_range, psf_size=psf_size, Nlimit=Nlimit)


                # mask_imagelist is a mask the size of the True
                # elements of mask_use, so need to update mask_use;
                # use np.place as following does not update [mask_use]:
                # mask_use[mask_use][~mask_imagelist] = False
                np.place (mask_use, mask_use, mask_imagelist)


                # check again if there are still sufficient images
                # left to combine
                if np.sum(mask_use) < nmin:
                    log.warning ('fewer images ({}) available than the minimum '
                                 'number required ({}); not creating co-add '
                                 'for {} in filter {}'
                                 .format(np.sum(mask_use), nmin, obj, filt))
                    continue


                # if sufficient number of images within seeing range,
                # update [mask]
                nmin_4clipping = get_par(set_br.nmin_4clipping,tel)
                if np.sum(mask_use) >= nmin_4clipping:

                    np.place (mask, mask, mask_use)
                    nfiles = np.sum(mask)
                    log.info ('{} files left for {} in filter {} after '
                              'selecting images within seeing spread'
                              .format(nfiles, obj, filt))

                else:
                    # if not, use weighted combine_type rather than clipped
                    combine_type = 'weighted'
                    log.info ('number of images available ({}) within allowed '
                              'seeing spread ({}) is less than {}, using '
                              'weighted instead of clipped image combination'
                              .format(np.sum(mask_use), max_spread,
                                      nmin_4clipping))



            # sort files based on their LIMMAG, highest value first
            indices_sort = np.argsort(table[mask]['LIMMAG'])[::-1]
            limmags_sort = table[mask]['LIMMAG'][indices_sort]
            seeing_sort = table[mask]['S-SEEING'][indices_sort]
            bkgstd_sort = table[mask]['S-BKGSTD'][indices_sort]
            files_sort = table[mask]['FILENAME'][indices_sort]


            if False:

                # calculate projected cumulative LIMMAG if images
                # would be combined using simple average
                limmags_sort_cum = -2.5*np.log10(
                    np.sqrt(np.cumsum((10**(-0.4*limmags_sort))**2))
                    / (np.arange(len(limmags_sort))+1))


                # weighted version, calculating the error in the weighted
                # mean using the values from S-BKGSTD as the sigmas (the
                # images are weighted using the background STD images);
                # comparison with the value from the first image
                # determines how much more deeper the combined images are
                limmags_sort_cum = (limmags_sort[0]-2.5*np.log10(
                    1./np.sqrt(np.cumsum(1./bkgstd_sort**2))/bkgstd_sort[0]))



            # weighted version using limiting magnitudes (converted to
            # flux) instead of S-BKGSTD
            limflux_sort = 10**(-0.4*limmags_sort)
            limmags_sort_cum = (limmags_sort[0]-2.5*np.log10(
                1./np.sqrt(np.cumsum(1./limflux_sort**2))/limflux_sort[0]))


            # filter based on target limiting magnitude
            if not deep:
                limmag_target = get_par(set_br.limmag_target,tel)[filt]
            else:
                # set limiting magnitudes to mag=30 for all filters
                limmag_target = 30.0


            # add dmag to target magnitude to account for the fact
            # that the projected limiting magnitude will be somewhat
            # higher than the actual one. Even higher in clipped mode
            # due to some additional images possibly being discarded
            # in the A_swarp determination
            if combine_type == 'clipped':
                dmag = 0.3
            else:
                dmag = 0.2

            # mask of images within approximate target limiting
            # magnitude (which is 30 if deep is True)
            mask_sort_cum = (limmags_sort_cum <= limmag_target + dmag)

            # use a minimum number of files, adding 1 to images
            # selected above
            nuse = max (np.sum(mask_sort_cum)+1, nmin)

            # [nuse] should not be larger than number of images available
            nuse = min (nuse, nfiles)

            # if deep is not True, also limit number of images to
            # nimages_max defined in settings file
            if not deep and nuse > nmax:
                log.warning ('limiting number of images to {} defined in '
                             'set_br.nimages_max'.format(nmax))
                nuse = min(nuse, nmax)

            # update mask
            mask_sort_cum[0:nuse] = True
            mask_sort_cum[nuse:] = False


            # files that were excluded
            nfiles_excl = np.sum(~mask_sort_cum)
            if nfiles_excl > 0:
                files_2exclude = files_sort[~mask_sort_cum]
                limmags_2exclude = limmags_sort[~mask_sort_cum]
                seeing_2exclude = seeing_sort[~mask_sort_cum]
                log.warning ('{} files and their limmags excluded from {}-band '
                             'coadd'.format(nfiles_excl, filt))
                for i in range(len(files_2exclude)):
                    log.info ('{}, {:.3f}'.format(files_2exclude[i],
                                                  limmags_2exclude[i]))


            # files to combine
            files_2coadd = files_sort[mask_sort_cum]
            limmags_2coadd = limmags_sort[mask_sort_cum]
            seeing_2coadd = seeing_sort[mask_sort_cum]
            bkgstd_2coadd = bkgstd_sort[mask_sort_cum]

            nfiles_used = nfiles - nfiles_excl
            log.info ('{} files, limmags, projected cumulative limmags and '
                      'seeing used for {}-band coadd'.format(nfiles-nfiles_excl,
                                                             filt))
            for i in range(len(files_2coadd)):
                log.info ('{} {:.3f} {:.3f} {:.2f}'
                          .format(files_2coadd[i], limmags_2coadd[i],
                                  limmags_sort_cum[i], seeing_2coadd[i]))

            limmag_proj = limmags_sort_cum[mask_sort_cum][-1]

            log.info ('projected (target) {}-band limiting magnitude of '
                      'co-add: {:.2f} ({})'
                      .format(filt, limmag_proj, limmag_target))



            # add this set of images with their field_ID and
            # filter to the lists of images, field_IDs and filters
            # to be processed
            list_of_imagelists.append(list(files_2coadd))
            obj_list.append(obj)
            filt_list.append(filt)
            radec_list.append(radec)
            imagesize_list.append(image_size)
            nfiles_list.append(nfiles)
            limmag_proj_list.append(limmag_proj)
            combine_type_list.append(combine_type)
            A_swarp_list.append(A_swarp)
            nsigma_clip_list.append(nsigma_clip)


    if len(table)==0:
        log.warning ('no field IDs with sufficient number of good images to '
                     'process')
        logging.shutdown()
        return


    # (multi-)process remaining files
    # -------------------------------

    if nproc==1:

        # if only 1 process is requested, or [image] input, run it
        # witout multiprocessing; this will allow images to be shown
        # on the fly if [set_zogy.display] is set to True; something
        # that is not allowed (at least not on a macbook) when
        # multiprocessing.
        log.warning ('running with single processor')
        for i in range(len(obj_list)):
            prep_ref (list_of_imagelists[i], obj_list[i], filt_list[i],
                      radec_list[i], imagesize_list[i], nfiles_list[i],
                      limmag_proj_list[i], combine_type_list[i],
                      A_swarp_list[i], nsigma_clip_list[i], skip_zogy)

    else:

        # feed the lists that were created above to the
        # multiprocessing helper function [pool_func_lists] that will
        # arrange each process to call [prep_ref] to prepare the
        # reference image for a particular field and filter
        # combination, using the [imcombine] function
        result = zogy.pool_func_lists (prep_ref, list_of_imagelists, obj_list,
                                       filt_list, radec_list, imagesize_list,
                                       nfiles_list, limmag_proj_list,
                                       combine_type_list, A_swarp_list,
                                       nsigma_clip_list,
                                       [skip_zogy] * len(obj_list), nproc=nproc)



    # make color figures
    # ------------------
    if make_colfig:
        log.info ('preparing color figures')
        # also prepare color figures
        try:
            result = zogy.pool_func (prep_colfig, objs_uniq, filters_colfig,
                                     nproc=nproc)
        except Exception as e:
            #log.exception (traceback.format_exc())
            log.exception ('exception was raised during [pool_func]: {}'
                              .format(e))
            raise RuntimeError



    logging.shutdown()
    return


################################################################################

def calc_imsize (ra, dec, ra0, dec0, imsize, bkg_size, pixscale):

    # calculate maximum offset in RA and DEC to determine
    # output imagesize (square) that include all images for
    # this object, and expanding it for the size to contain a
    # multiple of the background boxsize
    offset_ra = np.amax(np.abs(zogy.haversine(ra, dec, ra0, dec0)))
    offset_dec = np.amax(np.abs(dec - dec0))
    offset_pix = int(max(offset_ra, offset_dec) * 3600 / pixscale)
    # grow until multiple of background boxsize
    while offset_pix % bkg_size != 0:
        offset_pix += 1

    return imsize + 2*offset_pix


################################################################################

def set_date (date, start=True):

    """function to convert start/end dates at noon to mjd"""

    mjd_today_noon = int(Time.now().mjd) + 0.5

    # if no date is specified, include all data from 20 years ago
    # until now
    if date is None:
        mjd = mjd_today_noon
        if start:
            mjd -= 365.25 * 20
    else:
        # if date string is less than 8 characters, assume it is
        # relative wrt now/today
        if len(date) < 8:
            mjd = mjd_today_noon + round(float(date))
        else:
            # otherwise convert date string to mjd
            date = re.sub(',|-|\.|\/', '', date)
            mjd = bb.date2mjd ('{}'.format(date), time_str='12:00')

    return mjd


################################################################################

def prep_colfig (field_ID, filters):

    # determine reference directory and file
    if ref_mode:
        # set according to definition in setttings file
        ref_path = '{}/{:0>5}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    else:
        # set to input parameter results_dir = global parameter dir_results
        ref_path = '{}/{:0>5}'.format(dir_results, field_ID)

        # add extension to field_ID subfolder
        if ext is not None:
            ref_path = '{}{}'.format(ref_path, ext)


    # header keyword to use for scaling (e.g. PC-ZP or LIMMAG)
    key = 'LIMMAG'

    # initialize rgb list of images
    images_rgb = []
    images_std = []
    images_zp = []

    for filt in filters:

        image = '{}/{}_{:0>5}_{}_red.fits'.format(ref_path, tel, field_ID, filt)
        exists, image = bb.already_exists(image, get_filename=True)

        if not exists:
            log.info ('{} does not exist; not able to prepare color '
                      'figure for field_ID {:0>5}'.format(image, field_ID))
            return
        else:

            # add to image_rgb list (unzip if needed)
            image = bb.unzip(image, put_lock=False)
            images_rgb.append(image)

            # read image data and header
            data, header = zogy.read_hdulist(image, get_header=True)

            # determine image standard deviation
            mean, median, std = sigma_clipped_stats(data)
            images_std.append(std)

            # read header zeropoint
            if key in header:
                images_zp.append(header[key])
            else:
                log.info ('missing header keyword {}; not able to '
                          'prepare color figure for field_ID {:0>5}'
                          .format(key, field_ID))
                return

    # scaling
    f_min = 0
    vmin_r = f_min * images_std[0]
    vmin_g = f_min * images_std[1]
    vmin_b = f_min * images_std[2]

    f_max = 10
    vmax_r = f_max * images_std[0] * 10**(-0.4*(images_zp[2]-images_zp[0]))
    vmax_g = f_max * images_std[1] * 10**(-0.4*(images_zp[2]-images_zp[1]))
    vmax_b = f_max * images_std[2]

    # make color figure
    colfig = '{}/{}_{:0>5}_{}.png'.format(ref_path, tel, field_ID, filters)
    aplpy.make_rgb_image(images_rgb, colfig,
                         vmin_r=vmin_r, vmax_r=vmax_r,
                         vmin_g=vmin_g, vmax_g=vmax_g,
                         vmin_b=vmin_b, vmax_b=vmax_b)


################################################################################

def ref_already_exists (ref_path, tel, field_ID, filt, get_filename=False):

    # list files in ref_path with search string [tel]_[fieldID]_[filt]
    # and ending with _red.fits.fz
    list_ref = bb.list_files(ref_path, search_str='{}_{:0>5}_{}_'
                             .format(tel, field_ID, filt),
                             end_str='_red.fits.fz')

    # initialize exists and filename
    exists, filename = False, None

    # check if list_ref has any entries
    if len(list_ref) > 0:
        exists = True
        filename = list_ref[-1]
        if len(list_ref) > 1:
            log.warning ('multiple reference images with the same field ID/'
                         'filter combination {:0>5}/{} present in {}:\n{}\n'
                         'returning the last one'
                         .format(field_ID, filt, ref_path, list_ref))

    if get_filename:
        return exists, filename
    else:
        return exists


################################################################################

def prep_ref (imagelist, field_ID, filt, radec, image_size, nfiles, limmag_proj,
              combine_type, A_swarp, nsigma_clip, skip_zogy, dlimmag_min=0.3):


    # determine and create reference directory
    if ref_mode:
        # set according to definition in blackbox setttings file
        ref_path = '{}/{:0>5}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    else:
        # set to input parameter results_dir = global parameter dir_results
        ref_path = '{}/{:0>5}'.format(dir_results, field_ID)

        # add extension to field_ID subfolder
        if ext is not None:
            ref_path = '{}{}'.format(ref_path, ext)


    log.info ('ref_path: {}'.format(ref_path))


    # create folder
    bb.make_dir (ref_path)


    # name of output file, including full path
    ref_fits_out = '{}/{}_{:0>5}_{}_red.fits'.format(ref_path, tel,
                                                     field_ID, filt)


    log.info ('ref_fits_out: {}'.format(ref_fits_out))

    # if reference image already exists, check if images used are the
    # same as the input [imagelist]
    exists, ref_fits_old = ref_already_exists (ref_path, tel, field_ID, filt,
                                               get_filename=True)

    if exists:

        if False and ref_mode:
            log.warning ('reference image {} already exists; not remaking it'
                         .format(ref_fits_out))
            return

        else:

            # only remake the reference image if new individual files
            # are available
            log.info ('reference image {} already exists; checking if it needs '
                      'updating'.format(ref_fits_old))
            # read header
            header_ref_old = zogy.read_hdulist (ref_fits_old, get_data=False,
                                                get_header=True)
            # check how many images were used
            if 'R-NUSED' in header_ref_old:
                n_used = header_ref_old['R-NUSED']
            else:
                n_used = 1

            # gather used images into list
            if 'R-IM1' in header_ref_old:
                imagelist_used = [header_ref_old['R-IM{}'.format(i+1)]
                                  for i in range(n_used)
                                  if 'R-IM{}'.format(i+1) in header_ref_old]

            # compare input [imagelist] with [imagelist_used]; if they are
            # the same, no need to build this particular reference image
            # again
            imagelist_new = [image.split('/')[-1].split('.fits')[0]
                             for image in imagelist]
            if set(imagelist_new) == set(imagelist_used):
                # same sets of images, return
                log.info ('imagelist_new: {}'.format(imagelist_new))
                log.info ('imagelist_used: {}'.format(imagelist_used))
                log.info ('reference image of {:0>5} in filter {} with same '
                          'set of images already present; not remaking it'
                          .format(field_ID, filt))
                return



    # prepare temporary folder
    tmp_path = ('{}/{:0>5}/{}'
                .format(get_par(set_bb.tmp_dir,tel), field_ID,
                        ref_fits_out.split('/')[-1].replace('.fits','')))
    bb.make_dir (tmp_path, empty=True)
    log.info ('tmp_path: {}'.format(tmp_path))


    # change to tmp folder to be able to track disk usage
    orig_path = os.getcwd()
    os.chdir(tmp_path)


    # names of tmp output fits and its mask
    ref_fits = '{}/{}'.format(tmp_path, ref_fits_out.split('/')[-1])
    ref_fits_mask = ref_fits.replace('red.fits','mask.fits')


    # RA and DEC center of output image
    ra_center, dec_center = radec


    # create logfile specific to this reference image in tmp folder
    # (to be copied to final output folder at the end)
    logfile = ref_fits.replace('.fits', '.log')
    fileHandler = logging.FileHandler(logfile, 'a')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel('INFO')
    log.addHandler(fileHandler)
    log.info ('logfile created: {}'.format(logfile))


    # check if sufficient images available to combine
    if len(imagelist) == 0:

        log.error ('no images available to combine for field ID {:0>5} '
                   'in filter {}'.format(field_ID, filt))
        bb.clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel))
        bb.close_log(log, logfile)
        return

    else:


        if len(imagelist) == 1:
            log.warning ('only a single image available for field ID {:0>5} in '
                         'filter {}; using it as the reference image'
                         .format(field_ID, filt))


        # run imcombine
        log.info('running imcombine; outputfile: {}'.format(ref_fits))

        try:
            imcombine (field_ID, imagelist, ref_fits, combine_type, filt,
                       masktype_discard = get_par(set_br.masktype_discard,tel),
                       tempdir = tmp_path,
                       ra_center = ra_center,
                       dec_center = dec_center,
                       image_size = image_size,
                       nfiles = nfiles,
                       limmag_proj = limmag_proj,
                       A_swarp = A_swarp,
                       nsigma_clip = nsigma_clip,
                       back_type = get_par(set_br.back_type,tel),
                       back_size = get_par(set_zogy.bkg_boxsize,tel),
                       back_filtersize = get_par(set_zogy.bkg_filtersize,tel),
                       remap_each = True,
                       swarp_cfg = get_par(set_zogy.swarp_cfg,tel),
                       nthreads = set_br.nthreads)

        except Exception as e:
            #log.exception (traceback.format_exc())
            log.exception ('exception was raised during [imcombine]: {}'
                           .format(e))
            bb.clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel))
            bb.close_log(log, logfile)
            raise RuntimeError


        if not skip_zogy:

            # run zogy on newly prepared reference image
            try:
                zogy_processed = False
                header_optsub = zogy.optimal_subtraction(
                    ref_fits=ref_fits, ref_fits_mask=ref_fits_mask,
                    set_file='set_zogy', verbose=None,
                    nthreads=set_br.nthreads, telescope=tel,
                    keep_tmp=get_par(set_br.keep_tmp,tel))

            except Exception as e:
                #log.exception (traceback.format_exc())
                log.exception ('exception was raised during reference '
                               '[optimal_subtraction]: {}'.format(e))

            else:
                zogy_processed = True

            finally:
                if not zogy_processed:
                    log.error ('due to exception: returning without copying '
                               'reference files')

                    bb.clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel))
                    bb.close_log(log, logfile)
                    return

            log.info('zogy_processed: {}'.format(zogy_processed))





        # check quality control
        # ---------------------

        qc_flag = qc.run_qc_check (header_optsub, tel, check_key_type='ref')

        if qc_flag == 'red':
            log.error ('encountered red flag; not saving reference image {}'
                       'to reference folder'.format(ref_fits))

        else:
            # update [ref_fits] header with qc-flags
            header_ref = zogy.read_hdulist(ref_fits, get_data=False,
                                           get_header=True)
            for key in header_optsub:
                if 'QC' in key or 'DUMCAT' in key:
                    log.info ('updating header keyword {} with: {} for image {}'
                              .format(key, header_optsub[key], ref_fits))
                    header_ref[key] = (header_optsub[key],
                                       header_optsub.comments[key])


            # update fits image and catalog headers and create
            # separate header files
            zogy.update_imcathead (ref_fits, header_ref, create_hdrfile=True)
            ref_fits_cat = '{}.fits'.format(ref_fits.split('.fits')[0])
            zogy.update_imcathead (ref_fits_cat, header_ref,
                                   create_hdrfile=True)



            # before replacing old reference file, first check if
            # delta LIMMAG is large enough; could already do so at
            # start of this function [prep_ref] to avoid making the
            # reference image, however, final LIMMAG is usually but
            # not always smaller than the projected LIMMAG

            limmag = header_ref['LIMMAG']
            limmag_old = header_ref_old['LIMMAG']

            if limmag - limmag_old > dlimmag_min:

                # copy/move files to the reference folder
                tmp_base = ref_fits.split('_red.fits')[0]
                ref_base = ref_fits_out.split('_red.fits')[0]
                # add date of creation to ref_base
                date_today = Time.now().isot.split('T')[0].replace('-','')
                ref_base = '{}_{}'.format(ref_base, date_today)


                # first (re)move old reference files
                oldfiles = bb.list_files (ref_base)
                if len(oldfiles)!=0:
                    if False:
                        # remove them
                        zogy.remove_files (oldfiles, verbose=True)
                    else:
                        # or move them to the ref-old folder instead
                        old_path = '{}-old/{:0>5}'.format(
                            get_par(set_bb.ref_dir,tel), field_ID)

                        bb.make_dir (old_path)
                        for f in oldfiles:
                            f_dest = '{}/{}'.format(old_path,f.split('/')[-1])
                            #shutil.move (f, f_dest)
                            bb.copy_file (f, f_dest, move=True)


                # now move [ref_2keep] to the reference directory
                result = bb.copy_files2keep(tmp_base, ref_base,
                                            get_par(set_bb.ref_2keep,tel),
                                            move=False)



    # also build a couple of alternative reference images for
    # comparison; name these ...._whatever_red.fits, so that they do
    # get copied over to the reference folder below (which uses the
    # file extensions defined in blackbox settings file)
    if False:
        masktype_discard = get_par(set_br.masktype_discard,tel)

        def help_imcombine (combine_type, back_type, back_default=0,
                            back_size=30, back_filtersize=5):

            if back_type == 'auto':
                ext_tmp = '_{}_{}_{}_{}.fits'.format(combine_type, back_type,
                                                     back_size, back_filtersize)
            elif back_type == 'manual':
                ext_tmp = '_{}_{}_{}.fits'.format(combine_type, back_type,
                                                  back_default)
            elif back_type == 'constant':
                ext_tmp = '_{}_{}_clipmed.fits'.format(combine_type, back_type)

            else:
                ext_tmp = '_{}_{}.fits'.format(combine_type, back_type)

            ref_fits_temp = ref_fits.replace('.fits', ext_tmp)

            imcombine (field_ID, imagelist, ref_fits_temp, combine_type, filt,
                       ra_center=ra_center, dec_center=dec_center, nfiles=nfiles,
                       limmag_proj=limmag_proj,
                       back_type=back_type, back_default=back_default,
                       back_size=back_size, back_filtersize=back_filtersize,
                       masktype_discard=masktype_discard, tempdir=tmp_path,
                       remap_each=False,
                       swarp_cfg=get_par(set_zogy.swarp_cfg,tel),
                       nthreads=set_br.nthreads)

            # copy combined image to reference folder
            shutil.move (ref_fits_temp, ref_path)


        if False:
            help_imcombine ('weighted', 'blackbox')
            help_imcombine ('clipped', 'auto', back_size=60, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=120, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=240, back_filtersize=5)
            help_imcombine ('clipped', 'auto', back_size=960, back_filtersize=5)
            help_imcombine ('average', 'none')
            help_imcombine ('clipped', 'constant')



    log.info('finished making reference image: {}'.format(ref_fits_out))


    # changing back to original working dir
    os.chdir(orig_path)


    bb.clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel))
    bb.close_log(log, logfile)
    return


################################################################################

def imcombine (field_ID, imagelist, fits_out, combine_type, filt, overwrite=True,
               masktype_discard=None, tempdir='.temp', ra_center=None,
               dec_center=None, image_size=None, nfiles=0, limmag_proj=None,
               A_swarp=None, nsigma_clip=None, use_wcs_center=True,
               back_type='auto', back_default=0, back_size=120,
               back_filtersize=3, remap_each=False, swarp_cfg=None, nthreads=0):


    """Module to combine MeerLICHT/BlackGEM images.  The headers of the
    input images (all assumed to be useable, i.e. no red flags) need
    to have a WCS solution that SWarp understands, as SWarp is used to
    project the images to a common WCS frame, before the combining of
    the separate images is done.

    If the input images have an accompanying mask, i.e. with the same
    base name and containing "mask", then that mask will be used to
    avoid using e.g. edge pixels or cosmic rays in the combination.

    [ra_center] and [dec_center] define the central coordinates of the
    resulting image. If they are not both defined, the median center
    of the input images is used, in which case [use_wcs_center]
    determines whether the WCS center or the header values from the
    'RA' and 'DEC' keywords is used.

    To do:

    - include 3 main parts:
      --> basic combination, including weights (using SWarp)
      --> optimal combination (ZOGY)
      --> astrodrizzle

    - filter out qc-flag==red images

    - include photometric scaling (FSCALE_DEFAULT) using the zeropoint
      ZP, airmass A and extinction coefficient k, to re-calculate
      the Fn/Fr flux ratio (header Z-FNR is often not available):

                 Fn/Fr = 10**(ZPn-ZPr-k*(An-Ar))

      (should Airmass be included??)

    """


    t0 = time.time()
    zogy.mem_use ('at start of imcombine')


    if os.path.isfile(fits_out) and not overwrite:
        raise RuntimeError ('output image {} already exist'
                            .format(fits_out))

    # if fits_mask_out already exists, raise error
    fits_mask_out = fits_out.replace('red.fits', 'mask.fits')
    if os.path.isfile(fits_out) and not overwrite:
        raise RuntimeError ('output image {} already exist'
                            .format(fits_out))

    # if output weights or background standard deviation mini image
    # (=sqrt(1/weights image)) already exists, raise error
    fits_weights_out = fits_out.replace('.fits', '_weights.fits')
    fits_bkg_std_mini = fits_out.replace('.fits', '_bkg_std_mini.fits')
    if (os.path.isfile(fits_weights_out) or os.path.isfile(fits_bkg_std_mini)
        and not overwrite):
        raise RuntimeError ('output weights {} or background STD mini image {} '
                            'already exist'.format(fits_weights_out,
                                                   fits_bkg_std_mini))


    # check if there are at least a single image selected
    if len(imagelist) < 1:
        raise RuntimeError ('zero images selected')


    # if SWarp configuration file does not exist, create default one in [tempdir]
    if swarp_cfg is None:
        swarp_cfg = tempdir+'/swarp.config'
        cmd = 'swarp -d > {}'.format(swarp_cfg)
        result = subprocess.run(cmd, shell=True)
    else:
        if not os.path.isfile(swarp_cfg):
            raise IOError ('file {} does not exist'.format(swarp_cfg))


    # COMBINE TYPE MEDIAN
    # Tells SWarp how to combine resampled images:
    # MEDIAN Take the median of pixel values
    # AVERAGE Take the average
    # MIN Take the minimum
    # MAX Take the maximum
    # WEIGHTED Take the weighted average
    # CHI2 Take the weighted, quadratic sum
    # SUM Take the sum
    # not in latest manual v2.21 (code is at v2.38)
    # CLIPPED, CHI-OLD, CHI-MODE, CHI-MEAN, WEIGHTED_WEIGHT, MEDIAN_WEIGHT,
    # AND, NAND, OR or NOR

    # make sure combine_type, back_type and center_type are lower case
    combine_type = combine_type.lower()
    back_type = back_type.lower()

    # check if value of [combine_type] is valid; if not, exit
    combine_type_list = ['median', 'average', 'min', 'max', 'weighted', 'chi2',
                         'sum', 'clipped', 'weighted_weight', 'median_weight']
    if combine_type not in combine_type_list:
        raise ValueError ('[combine_type] method "{}" should be one of {}'.
                          format(combine_type, combine_type_list))


    # keeping temporary files?
    keep_tmp = get_par(set_br.keep_tmp,tel)


    # initialize table with image values to keep
    names = ('ra_center', 'dec_center', 'xsize', 'ysize', 'zp', 'airmass', 'gain',
             'rdnoise', 'saturate', 'exptime', 'mjd_obs', 'fscale', 'seeing',
             'pixscale', 'image_name_red', 'image_name_tmp', 'mask_name_tmp')
    dtypes = ('f8', 'f8', 'i4', 'i4', 'f8', 'f8', 'f8',
              'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
              'f8', 'U100', 'U100', 'U100')
    data_tmp = np.zeros((len(imagelist), len(names)))
    imtable = Table(data=data_tmp, names=names, dtype=dtypes)


    # mask in case particular image is not used in loop below
    mask_keep = np.ones(len(imtable), dtype=bool)


    # loop input list of images
    for nimage, image in enumerate(imagelist):

        if not zogy.isfile(image):
            raise RuntimeError ('input image {} does not exist'.format(image))

        # read input image data and header
        data, header = zogy.read_hdulist(image, get_header=True, dtype='float32')

        # read corresponding mask image
        image_mask = image.replace('red.fits', 'mask.fits')
        data_mask, header_mask = zogy.read_hdulist(image_mask, get_header=True,
                                                   dtype='uint8')

        # read relevant header keywords
        keywords = ['naxis1', 'naxis2', 'ra', 'dec', 'pc-zp', 'pc-zpstd',
                    'airmass', 'pc-extco', 'gain', 'rdnoise', 'saturate',
                    'exptime', 'mjd-obs', 's-seeing', 'a-pscale']
        try:
            results = read_header_alt (header, keywords)
            xsize, ysize, ra_temp, dec_temp, zp, zp_std, airmass, extco, gain, \
                rdnoise, saturate, exptime, mjd_obs, seeing, pixscale = results

        except Exception as e:
            log.exception('exception was raised when reading header of image {}\n'
                          'not using it in image combination'.format(image, e))
            # do not use this row
            mask_keep[nimage] = False
            continue


        if 'BKG-SUB' in header and header['BKG-SUB']:
            bkg_sub = True
        else:
            bkg_sub = False


        if back_type != 'new' or bkg_sub:

            # background itself is not needed if the image was already
            # background subtracted; the background standard deviation
            # image is needed for the variance/weights calculation
            # below
            if not bkg_sub:
                # read corresponding mini background image
                image_bkg_mini = image.replace('red.fits', 'red_bkg_mini.fits')
                data_bkg_mini, header_bkg_mini = zogy.read_hdulist(
                    image_bkg_mini, get_header=True, dtype='float32')

                # convert mini to full background image
                bkg_boxsize = header_bkg_mini['BKG-SIZE']
                data_bkg = zogy.mini2back (
                    data_bkg_mini, data.shape, order_interp=3,
                    bkg_boxsize=bkg_boxsize, interp_Xchan=True, timing=False)


            # read mini background STD image
            image_bkg_std_mini = image.replace('red.fits',
                                               'red_bkg_std_mini.fits')
            data_bkg_std_mini, header_bkg_std_mini = zogy.read_hdulist(
                image_bkg_std_mini, get_header=True, dtype='float32')


            # convert mini STD to full background STD image
            bkg_boxsize = header_bkg_std_mini['BKG-SIZE']
            data_bkg_std = zogy.mini2back (
                data_bkg_std_mini, data.shape, order_interp=3,
                bkg_boxsize=bkg_boxsize, interp_Xchan=False, timing=False)


            image_temp = '{}/{}'.format(tempdir, image.split('/')[-1])
            image_temp_bkg_std = image_temp.replace('red.fits',
                                                    'red_bkg_std.fits')
            image_temp_bkg_std = image_temp_bkg_std.replace('.fz','')
            # save if keeping temporary files
            if keep_tmp:
                fits.writeto(image_temp_bkg_std, data_bkg_std, overwrite=True)


        else:

            # redo background determination with parameter settings as
            # defined in set_zogy and running source-extractor to get
            # object mask needed for proper background subtraction

            # copy image and mask to temp folder
            shutil.copy2 (image, tempdir)
            shutil.copy2 (image_mask, tempdir)
            # unzip if needed
            image_temp = '{}/{}'.format(tempdir, image.split('/')[-1])
            image_mask_temp = image_temp.replace('red.fits', 'mask.fits')
            image_temp = bb.unzip(image_temp, put_lock=False)
            image_mask_temp = bb.unzip(image_mask_temp, put_lock=False)

            # run source-extractor
            base = image_temp.split('.fits')[0]
            sexcat = '{}_ldac.fits'.format(base)
            fwhm = seeing / pixscale
            #fwhm = header['S-FWHM']
            imtype = 'new'
            sex_params = get_par(set_zogy.sex_par,tel)
            try:
                result = zogy.run_sextractor(
                    image_temp, sexcat, get_par(set_zogy.sex_cfg,tel),
                    sex_params, pixscale, header, fit_psf=False,
                    return_fwhm_elong=False, fraction=1.0, fwhm=fwhm,
                    update_vignet=False, imtype=imtype,
                    fits_mask=image_mask_temp, npasses=2, tel=tel,
                    set_zogy=set_zogy)

            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [run_sextractor]: {}'
                              .format(e))

            # read source-extractor output image data and header
            data, header = zogy.read_hdulist(image_temp, get_header=True,
                                             dtype='float32')

            # check if background was subtracted this time
            if 'BKG-SUB' in header and header['BKG-SUB']:
                bkg_sub = True
            else:
                bkg_sub = False

            if not bkg_sub:
                # read background image created in [run_sextractor]
                image_temp_bkg = '{}_bkg.fits'.format(base)
                data_bkg = zogy.read_hdulist (image_temp_bkg, dtype='float32')

            image_temp_bkg_std = '{}_bkg_std.fits'.format(base)
            data_bkg_std = zogy.read_hdulist (image_temp_bkg_std,
                                              dtype='float32')

            if not keep_tmp:
                files2remove = [image_temp_bkg, image_temp_bkg_std]
                zogy.remove_files (files2remove, verbose=True)




        # determine weights image (1/variance)
        # for Poisson noise component, use background image instead of
        # image itself:
        #data_var = data_bkg + rdnoise**2
        data_weights = data_bkg_std**2
        index_nonzero = np.nonzero(data_weights)
        data_weights[index_nonzero] = 1./data_weights[index_nonzero]


        if False:
            # alternatively, provide the absolute values of the
            # background RMS map and using WEIGHT_TYPE MAP_RMS below;
            # however: this results in the resampled weights maps
            # (except for the one of the very first image) and also
            # the output weights map to contain mainly zeros
            data_weights = np.abs(data_bkg_std)


        # set pixels in data_mask that are to be discarded (selected
        # with input parameter masktype_discard) to zero in weights image
        mask_weights = np.zeros(data_mask.shape, dtype=bool)
        mask_value = get_par(set_zogy.mask_value,tel)
        # if more than a single image is combined, iterate over all
        # mask values and set weights to zero of mask values to be
        # discarded
        if np.sum(mask_keep) > 1:
            for val in mask_value.values():
                # check if this one is to be discarded
                if masktype_discard & val == val:
                    mask_discard = (data_mask & val == val)
                    mask_weights[mask_discard] = True
                    log.info('discarding mask value {}; no. of pixels: {}'
                             .format(val, np.sum(mask_discard)))

            # set corresponding pixels to zero in data_weights
            data_weights[mask_weights] = 0


        # fix pixels if saturated pixels are not considered as bad
        # pixels, i.e. their weight is not set to zero; if they are
        # included in [masktype_discard], then the saturated and
        # connected pixels are replaced (interpolated over) in the
        # combined image near the end of [imcombine] instead of in the
        # individual images
        base = image_temp.split('.fits')[0]
        value_sat = mask_value['saturated']
        if not (masktype_discard & value_sat == value_sat):
            data = zogy.fixpix (data, satlevel=saturate, data_mask=data_mask,
                                base=base, imtype='new', mask_value=mask_value,
                                keep_tmp=keep_tmp, along_row=True,
                                interp_func='gauss')


        # fill arrays with header info
        list_tmp = ['xsize', 'ysize', 'zp', 'airmass', 'gain', 'rdnoise',
                    'saturate', 'exptime', 'mjd_obs', 'seeing', 'pixscale']
        for key in list_tmp:
            imtable[key][nimage] = eval(key)


        # calculate flux ratio (fscale in SWarp speak) using the
        # zeropoint difference between image and first image,
        # i.e. scale the fluxes in image to match those in the first
        # image:
        #
        # zp = mag_cal - mag_inst + airmass * k
        # mag_inst = -2.5*log10(flux/exptime)
        # zp[0] = mag_cal - mag_inst[0] + A[0]*k
        # zp[i] = mag_cal - mag_inst[i] + A[i]*k
        # zp[0] = zp[i] + mag_inst[i] - A[i]*k - mag_inst[0] + A[0]*k
        # mag_inst[0] - mag_inst[i] = zp[i] - zp[0] - k * (A[i] - A[0])
        # = -2.5 * log10( (flux[0]/exptime[0]) / (flux[i]/exptime[i]))
        #
        # (fscale * flux[i] = flux[0] --> fscale = flux[0] / flux[i])
        #
        # dmag = -2.5 * log10( fscale * exptime[i] / exptime[0] )
        #
        # so finally:
        # fscale = 10**(dmag/-2.5) * exptime[0] / exptime[i]
        #
        # And scale all images to an airmass of 1 by setting A[0]=1
        # (the 1st and every other image are then all scaled to A=1).
        #
        # N.B.: this will lead to the reference image having the same
        # zeropoint as the (arbitrary) first image, which may be
        # confusing; could also scale it to the image with highest zp,
        # but then need to do a separate loop inferring the zps of all
        # images first, but that is not that straightforward.

        dmag = zp - imtable['zp'][mask_keep][0] - extco * (airmass - 1)
        fscale = (10**(dmag/-2.5) * imtable['exptime'][mask_keep][0] / exptime)
        # record in table
        imtable['fscale'][nimage] = fscale

        # add fscale to image header
        header['FSCALE'] = (fscale, 'flux ratio wrt to first image and at '
                            'airmass=1')
        log.info ('FSCALE of image {}: {:.3f}'.format(image, fscale))

        # update these header arrays with fscale
        imtable['rdnoise'][nimage] *= fscale
        imtable['saturate'][nimage] *= fscale

        # update weights image with scale factor according to Eq. 26
        # or 27 in SWarp manual:
        # N.B.: this is done internally by SWarp!!!
        #data_weights /= fscale**2


        if False:
            # when converting output weights image provided by SWarp to a
            # bkg_std image, the values in it are ~10% higher than the
            # actual STD in the combined co-added image; make a combined
            # STD image from the individual STD images to compare with:
            if nimage==0:
                data_bkg_var_out = (fscale * data_bkg_std)**2
            else:
                data_bkg_var_out += (fscale * data_bkg_std)**2


        # determine and record image centers
        if use_wcs_center:
            # determine WCS center of field
            wcs = WCS(header)
            ra_temp, dec_temp = wcs.all_pix2world(int(xsize/2), int(ysize/2), 1)
        else:
            # alternatively, using header keywords read above
            if ':' in ra_temp:
                ra_temp = Angle(ra_temp, unit=u.hour).degree
            if ':' in dec_temp:
                dec_temp = Angle(dec_temp, unit=u.degree).degree


        imtable['ra_center'][nimage] = ra_temp
        imtable['dec_center'][nimage] = dec_temp


        # save image in temp folder; first subtract background if
        # background option 'blackbox' is selected
        if back_type == 'blackbox' or back_type == 'new':

            # first check again if image is not already
            # background-subtracted; if it was not in the first place,
            # it could have been background subtracted in the
            # source-extractor run with back_type == 'new'
            if not bkg_sub:
                data -= data_bkg


        elif back_type == 'constant':

            # subtract one single value from the image: clipped median
            data_mean, data_median, data_std = sigma_clipped_stats (data)
            data -= data_median


        # make sure edge pixels are (still) zero
        value_edge = mask_value['edge']
        mask_edge = (data_mask==value_edge)
        data[mask_edge] = 0


        image_temp = '{}/{}'.format(tempdir, image.split('/')[-1]
                                    .replace('.fz',''))
        fits.writeto(image_temp, data, header=header, overwrite=True)
        # add to table
        imtable['image_name_red'][nimage] = image
        imtable['image_name_tmp'][nimage] = image_temp


        # save weights image in the temp folder
        weights_temp = image_temp.replace('.fits','_weights.fits')
        fits.writeto(weights_temp, data_weights, overwrite=True)
        # add to array of weights names; not needed as WEIGHT_SUFFIX
        # can be used
        #weights_names = np.append(weights_names, weights_temp)


        # save mask image in temp folder
        mask_temp = image_temp.replace('red.fits', 'mask.fits')
        # add WCS of image to mask header
        header_mask += header
        # set flux scaling to unity
        header_mask['FSCALE'] = (1.0,
                                 'flux ratio wrt to first image and at airmass=1')
        fits.writeto(mask_temp, data_mask, header=header_mask, overwrite=True)
        # add to array of names
        imtable['mask_name_tmp'][nimage] = mask_temp




    zogy.mem_use ('in imcombine after looping input images')


    # clean imtable from images that were not used
    imtable = imtable[mask_keep]


    # remove images that are not being used
    if np.sum(~mask_keep) > 0 and not keep_tmp:
        list_tmp = list(imtable['image_name_tmp'][~mask_keep])
        for im in list_tmp:
            files2remove = image_associates(im)
            zogy.remove_files (files2remove, verbose=True)


    # if input [ra_center] or [dec_center] is not defined, use the
    # median RA/DEC of the input images as the center RA/DEC of the
    # output image
    if ra_center is None or dec_center is None:
        ra_center = np.median(imtable['ra_center'])
        dec_center = np.median(imtable['dec_center'])

    # convert coordinates to input string for SWarp
    radec_str = '{},{}'.format(ra_center, dec_center)


    # if input [image_size] is None, determine the size so that all
    # input images fit; this corresponds to the 'all_filter' option of
    # [imagesize_type], easiest done here to include only the images
    # that are used in the end
    if image_size is None:

        # pixel scale
        pixscale_out = get_par(set_br.pixscale_out,tel)
        # background box size
        bkg_size = get_par(set_zogy.bkg_boxsize,tel)

        image_size = calc_imsize (imtable['ra_center'],
                                  imtable['dec_center'],
                                  ra_center, dec_center,
                                  imtable['ysize'][0], bkg_size, pixscale_out)


    # set background settings in SWarp; if input background option was
    # 'blackbox', the background was already subtracted from the image
    if back_type == 'auto':
        subtract_back_SWarp = 'Y'
        back_type_SWarp = back_type
    else:
        subtract_back_SWarp = 'N'
        back_type_SWarp = 'manual'


    pixscale_type = get_par(set_br.pixscale_type,tel).upper()
    pixscale_out = get_par(set_br.pixscale_out,tel)


    # define a list of keywords, mostly those created in [set_header]
    # function in blackbox.py and a few additional ones defined in
    # blackbox.py, that do not change between images; the keywords
    # from zogy.py will be added automatically as the co-added image
    # is put through zogy.py; also add DATE-OBS and MJD-OBS even
    # though they are updated later on - this is to avoid an astropy
    # warning with datfix updating any missing DATE-OBS keyword
    keys2copy = ['XBINNING', 'YBINNING', 'RADESYS', 'EPOCH', 'FLIPSTAT',
                 'OBJECT', 'IMAGETYP', 'FILTER', 'DATE-OBS', 'MJD-OBS',
                 'TIMESYS', 'SITELAT', 'SITELONG', 'ELEVATIO', 'EQUINOX',
                 'CCD-ID', 'CONTROLL', 'DETSPEED', 'CCD-NW', 'CCD-NH', 'FOCUSPOS',
                 'ORIGIN', 'MPC-CODE', 'TELESCOP', 'INSTRUME',
                 'OBSERVER', 'ABOTVER', 'PROGNAME', 'PROGID',
                 'PYTHON-V', 'BB-V', 'KW-V']

    # create order dictionary with SWarp command to execute
    cmd_dict = collections.OrderedDict()

    cmd_dict['swarp'] = ','.join(imtable['image_name_tmp'])
    cmd_dict['-c'] = swarp_cfg
    cmd_dict['-COMBINE'] = 'Y'
    cmd_dict['-COMBINE_TYPE'] = combine_type.upper()
    # WEIGHT_IMAGE input is not needed as suffix is defined
    #cmd_dict['-WEIGHT_IMAGE'] = ','.join(weights_names)
    cmd_dict['-WEIGHT_SUFFIX'] = '_weights.fits'
    cmd_dict['-WEIGHTOUT_NAME'] = fits_weights_out
    cmd_dict['-WEIGHT_TYPE'] = 'MAP_WEIGHT'
    cmd_dict['-RESCALE_WEIGHTS'] = 'N'
    cmd_dict['-CENTER_TYPE'] = 'MANUAL'
    cmd_dict['-CENTER'] = radec_str
    cmd_dict['-IMAGE_SIZE'] = str(image_size)
    cmd_dict['-PIXEL_SCALE'] = str(pixscale_out)
    cmd_dict['-PIXELSCALE_TYPE'] = pixscale_type
    cmd_dict['-IMAGEOUT_NAME'] = fits_out
    cmd_dict['-RESAMPLE_DIR'] = tempdir
    cmd_dict['-RESAMPLE_SUFFIX'] = '_resamp.fits'
    cmd_dict['-RESAMPLING_TYPE'] = 'LANCZOS3'
    # GAIN_KEYWORD cannot be GAIN, as the value of GAIN1 would then be adopted
    cmd_dict['-GAIN_KEYWORD'] = 'anything_but_gain'
    cmd_dict['-GAIN_DEFAULT'] = '1.0'
    cmd_dict['-SATLEV_KEYWORD'] = get_par(set_zogy.key_satlevel,tel)
    cmd_dict['-SUBTRACT_BACK'] = subtract_back_SWarp
    cmd_dict['-BACK_TYPE'] = back_type_SWarp.upper()
    cmd_dict['-BACK_DEFAULT'] = str(back_default)
    cmd_dict['-BACK_SIZE'] = str(back_size)
    cmd_dict['-BACK_FILTERSIZE'] = str(back_filtersize)
    cmd_dict['-FSCALE_KEYWORD'] = 'FSCALE'
    cmd_dict['-FSCALE_DEFAULT'] = '1.0'
    cmd_dict['-FSCALASTRO_TYPE'] = 'FIXED'
    cmd_dict['-VERBOSE_TYPE'] = 'FULL'
    cmd_dict['-NTHREADS'] = str(nthreads)
    cmd_dict['-COPY_KEYWORDS'] = ','.join(keys2copy)
    cmd_dict['-WRITE_FILEINFO'] = 'Y'
    cmd_dict['-WRITE_XML'] = 'N'
    cmd_dict['-VMEM_DIR'] = '.'
    cmd_dict['-VMEM_MAX'] = str(4096)
    cmd_dict['-MEM_MAX'] = str(4096)
    cmd_dict['-DELETE_TMPFILES'] = 'N'
    cmd_dict['-NOPENFILES_MAX'] = '256'


    # execute SWarp, in CLIPPED mode, 2 passes are executed
    for npass in range(2):

        if combine_type == 'clipped':

            if npass==0:

                # set AMPFRAC=A_swarp
                cmd_dict['-CLIP_AMPFRAC'] = str(A_swarp)

                # save clipped pixels in ASCII file
                clip_logname = '{}/clipped.dat'.format(tempdir)
                cmd_dict['-CLIP_LOGNAME'] = clip_logname
                cmd_dict['-CLIP_WRITELOG'] = 'Y'

                # clipping threshold
                cmd_dict['-CLIP_SIGMA'] = str(nsigma_clip)


            else:

                log.info ('converting clipped pixels identified in 1st pass '
                          'of SWarp to masks in the frame of the input images, '
                          'filter them and add the result to the weights images '
                          'to be used in the 2nd pass of SWarp')

                imagelist_tmp = list(imtable['image_name_tmp'])


                # use function [clipped2mask] to convert clipped
                # pixels identified by SWarp, saved in [clip_logname],
                # to masks in the individual image frames, filter them
                # with a few sliding windows and update the weights
                # images in the tmp folder
                clipped2mask (clip_logname, imagelist_tmp, nsigma_clip, fits_out)


                # for the 2nd pass, use the WEIGHTED combination, where
                # the weights images have been updated by [clipped2mask]
                cmd_dict['-COMBINE_TYPE'] = 'WEIGHTED'

                # turn off logging of clipped pixels
                cmd_dict['-CLIP_WRITELOG'] = 'N'



        # convert cmd_dict to list and execute it
        cmd = list(itertools.chain.from_iterable(list(cmd_dict.items())))
        cmd_str = ' '.join(cmd)
        log.info ('creating combined image with SWarp:\n{}'.format(cmd_str))
        result = subprocess.call(cmd)


        zogy.mem_use ('after npass={} creation of combined image'.format(npass))


        # no need to do 2nd pass in case combine_type in 1st pass was
        # not set to 'clipped'
        if cmd_dict['-COMBINE_TYPE'].lower() != 'clipped':
            break





    # update header of fits_out
    data_out, header_out = zogy.read_hdulist(fits_out, get_header=True,
                                             dtype='float32')

    # with RA and DEC
    header_out['RA'] = (ra_center, '[deg] telescope right ascension')
    header_out['DEC'] = (dec_center, '[deg] telescope declination')

    # also add RA-CNTR and DEC-CNTR to header, as these are expected
    # by at least the force_phot module
    header_out['RA-CNTR'] = (ra_center, '[deg] RA (ICRS) at image center')
    header_out['DEC-CNTR'] = (dec_center, '[deg] DEC (ICRS) at image center')

    # with gain, readnoise, saturation level, exptime and mjd-obs
    gain_eff, rdnoise_eff, saturate_eff, exptime_eff, mjd_obs_eff = calc_headers(
        combine_type, imtable)

    header_out.set('GAIN', gain_eff, '[e-/ADU] effective gain', after='DEC')
    header_out.set('RDNOISE', rdnoise_eff, '[e-] effective read-out noise',
                   after='GAIN')
    header_out.set('SATURATE', saturate_eff, '[e-] effective saturation '
                   'threshold', after='RDNOISE')
    header_out.set('EXPTIME', exptime_eff, '[s] effective exposure time',
                   after='SATURATE')
    date_obs = Time(mjd_obs_eff, format='mjd').isot
    header_out.set('DATE-OBS', date_obs, 'average date of observation',
                   after='EXPTIME')
    header_out.set('MJD-OBS', mjd_obs_eff, '[days] average MJD', after='DATE-OBS')


    # this line below could be added to allow a combined image to
    # serve as input to the new image in zogy.py, but this is only
    # valid in case set_br.pixscale_type is set to "manual" - need to
    # infer the pixelscale in case it is not manual.
    #header_out.set('A-PSCALE', pixscale_out, '[arcsec/pix] pixel scale WCS '
    #               'solution')


    # buildref version
    header_out['R-V'] = (__version__, 'reference building module version used')

    # time when module was started
    header_out['R-TSTART'] = (time_refstart, 'UT time that module was started')

    val_str = '[{},{}]'.format(start_date, end_date)
    header_out['R-TRANGE'] = (val_str,
                              '[date/days wrt R-TSTART] image time range')

    header_out['R-QCMAX'] = (max_qc_flag, 'maximum image QC flag')

    header_out['R-SEEMAX'] = (max_seeing, '[arcsec] maximum image seeing')

    # number of images available and used
    header_out['R-NFILES'] = (nfiles, 'number of images within constraints '
                              'available')
    header_out['R-NUSED'] = (len(imtable), 'number of images used to combine')

    # names of images that were used
    for nimage, image in enumerate(imtable['image_name_tmp']):
        image = image.split('/')[-1].split('.fits')[0]
        header_out['R-IM{}'.format(nimage+1)] = (image, 'image {} used to combine'
                                                 .format(nimage+1))

    # FSCALE used
    for nimage in range(len(imtable)):
        # also record scaling applied
        header_out['R-FSC{}'.format(nimage+1)] = (imtable['fscale'][nimage],
                                                  'image {} FSCALE used in SWarp'
                                                  .format(nimage+1))

    # A-swarp and clipping
    if combine_type == 'clipped' and 'A_swarp' in locals():
        header_out['R-ASWARP'] = (A_swarp, 'fraction of flux variation used in SWarp')
        header_out['R-NSIGMA'] = (nsigma_clip, '[sigma] clipping threshold used '
                                  'in SWarp')
    else:
        header_out['R-ASWARP'] = ('None', 'fraction of flux variation used '
                                  'in SWarp')
        header_out['R-NSIGMA'] = ('None', '[sigma] clipping threshold used '
                                  'in SWarp')


    # projected and target limiting magnitudes
    header_out['R-LMPROJ'] = (limmag_proj, '[mag] projected limiting magnitude')

    if not deep:
        limmag_target = get_par(set_br.limmag_target,tel)[filt]
    else:
        # set limiting magnitudes to mag=30 for all filters
        limmag_target = 30.0

    header_out['R-LMTARG'] = (limmag_target, '[mag] target limiting magnitude')

    # combination method
    header_out['R-COMB-M'] = (combine_type,
                              'input images combination method')
    # background subtraction method
    header_out['R-BKG-M'] = (back_type,
                             'input images background subtraction method')

    # background subtracted? N.B.: back_type=='none' is only used for
    # the special case of average combination without background
    # subtraction
    if back_type == 'none':
        bkg_sub = False
    else:
        bkg_sub = True

    header_out['BKG-SUB'] = (bkg_sub, 'sky background was subtracted?')

    header_out['R-CNTR-M'] = (get_par(set_br.center_type,tel),
                              'reference image centering method')

    header_out['R-SIZE-M'] = (get_par(set_br.imagesize_type,tel),
                              'reference image size method')

    # discarded mask values
    header_out['R-MSKREJ'] = (masktype_discard,
                              'reject pixels with mask values part of this sum')


    # any nan value in the image?
    mask_infnan = ~np.isfinite(data_out)
    if np.any(mask_infnan):
        log.info ('combined image contains non-finite numbers; replace with 0')
        data_out[mask_infnan] = 0


    # fluxes of individual images were scaled to airmass=1, and set
    # header AIRMASS accordingly
    header_out['AIRMASS'] = (1.0, 'Airmass forced to 1 in refbuild module')


    # mask values used in if/else block below and at the end of this
    # function
    mask_value = get_par(set_zogy.mask_value,tel)


    # in case of only 2 images, also create a MIN combination; skip
    # for now - images have to be very close in seeing to result in a
    # useful ref image
    if False and len(imtable) == 2:

        # initial weights images were updated with any outliers found
        # during the 1st clipping pass of SWarp, but need to use the
        # initial weights images for determining the minimum combined
        # image
        for fits_tmp in list(imtable['image_name_tmp']):
            fits_w_tmp = fits_tmp.replace('.fits', '_weights.fits')
            fits_w_orig = fits_tmp.replace('.fits',
                                           '_weights_orig.fits')
            shutil.copy2 (fits_w_orig, fits_w_tmp)

        # name of output minimum image
        fits_out_min = fits_out.replace('.fits', '_min.fits')

        cmd_dict['-COMBINE_TYPE'] = 'MIN'
        cmd_dict['-WEIGHTOUT_NAME'] = '{}/weights_out_tmp.fits'.format(tempdir)
        cmd_dict['-IMAGEOUT_NAME'] = fits_out_min

        # convert cmd_dict to list and execute it
        cmd = list(itertools.chain.from_iterable(list(cmd_dict.items())))
        cmd_str = ' '.join(cmd)
        log.info ('creating MIN image with SWarp:\n{}'.format(cmd_str))
        result = subprocess.call(cmd)

        # read minimum combination
        data_min = zogy.read_hdulist(fits_out_min, get_header=False,
                                     dtype='float32')



    if not remap_each:

        # run SWarp twice on mask image with combine_type OR and MIN

        # OR mask
        # -------
        fits_mask_OR = fits_mask_out.replace('mask', 'mask_OR')

        # edit existing [cmd_dict]
        cmd_dict['swarp'] = ','.join(imtable['mask_name_tmp'])
        cmd_dict['-COMBINE_TYPE'] = 'OR'
        # name for output weights image in tmp folder; not relevant
        # for these mask combinations, but SWarp creates a
        # "coadd.weight.fits" image in the folder where SWarp is run
        # even if WEIGHT_TYPE set to NONE
        cmd_dict['-WEIGHTOUT_NAME'] = '{}/weights_out_tmp.fits'.format(tempdir)
        cmd_dict['-WEIGHT_TYPE'] = 'NONE'
        cmd_dict['-IMAGEOUT_NAME'] = fits_mask_OR
        cmd_dict['-RESAMPLING_TYPE'] = 'NEAREST'
        cmd_dict['-SUBTRACT_BACK'] = 'N'

        # convert cmd_dict to list and execute it
        cmd = list(itertools.chain.from_iterable(list(cmd_dict.items())))
        cmd_str = ' '.join(cmd)
        log.info ('creating OR mask with SWarp:\n{}'.format(cmd_str))
        result = subprocess.call(cmd)


        # MIN mask
        # --------
        fits_mask_MIN = fits_mask_out.replace('mask', 'mask_MIN')
        cmd_dict['-COMBINE_TYPE'] = 'MIN'
        cmd_dict['-IMAGEOUT_NAME'] = fits_mask_MIN

        # convert cmd_dict to list and execute it
        cmd = list(itertools.chain.from_iterable(list(cmd_dict.items())))
        cmd_str = ' '.join(cmd)
        log.info ('creating MIN mask with SWarp:\n{}'.format(cmd_str))
        result = subprocess.call(cmd)



        # read OR and MIN output masks
        data_mask_OR = (zogy.read_hdulist(fits_mask_OR, get_header=False)
                        +0.5).astype('uint8')
        data_mask_MIN = (zogy.read_hdulist(fits_mask_MIN, get_header=False)
                         +0.5).astype('uint8')


        # now, wherever mask_MIN is not zero, implying that none of the
        # pixel values in the cube were valid, replace it with the OR mask
        data_mask_out = np.copy(data_mask_MIN)
        index_nonzero = np.nonzero(data_mask_MIN)
        data_mask_out[index_nonzero] = data_mask_OR[index_nonzero]


    else:

        # remapping each individual image if needed
        log.info ('remapping individual images')

        # also SWarp individual images
        refimage = fits_out
        header_refimage = zogy.read_hdulist(refimage, get_data=False,
                                            get_header=True)

        # initialize combined mask
        mask_array_shape = (len(imtable), image_size, image_size)

        for nimage, image in enumerate(imtable['image_name_tmp']):

            # skip remapping of images themselves for the moment; only
            # needed if some combination of the images other than
            # those available in SWarp is needed
            if False:

                t_temp = time.time()
                image_remap = image.replace('.fits', '_remap.fits')

                log.info ('refimage: {}'.format(refimage))
                log.info ('image: {}'.format(image))
                log.info ('image_remap: {}'.format(image_remap))

                if not os.path.isfile(image_remap):
                    try:
                        result = zogy.run_remap (
                            refimage, image, image_remap,
                            (image_size,image_size), config=swarp_cfg,
                            resample='N', resample_dir=tempdir,
                            resample_suffix='_resamp.fits', nthreads=nthreads,
                            tel=tel, set_zogy=set_zogy)

                    except Exception as e:
                        #log.exception(traceback.format_exc())
                        log.exception('exception was raised during [run_remap]: '
                                      '{}'.format(e))
                        raise RuntimeError

                    else:
                        log.info ('time spent in run_remap: {}'
                                  .format(time.time()-t_temp))


            # remap image masks
            image_mask = imtable['mask_name_tmp'][nimage]

            log.info ('processing mask: {}'.format(image_mask))

            data_mask, header_mask = zogy.read_hdulist(
                image_mask, get_header=True, dtype='uint8')

            t_temp = time.time()
            image_mask_remap = image_mask.replace('.fits', '_remap.fits')
            if not os.path.isfile(image_mask_remap):

                try:
                    result = zogy.run_remap (
                        refimage, image_mask, image_mask_remap,
                        (image_size,image_size), config=swarp_cfg,
                        resampling_type='NEAREST', resample_dir=tempdir,
                        resample_suffix='_resamp.fits',
                        dtype=data_mask.dtype.name,
                        value_edge=mask_value['edge'], nthreads=nthreads,
                        oversampling=0, tel=tel, set_zogy=set_zogy)

                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('exception was raised during [run_remap]: '
                                  '{}'.format(e))
                    raise RuntimeError

                else:
                    log.info ('wall-time spent in remapping mask: {}'
                              .format(time.time()-t_temp))


            t_temp = time.time()


            # SWarp has converted input mask to float32, so need to
            # read fits image back into integer array to use in
            # combination of masks below
            data_mask_remap = (zogy.read_hdulist(image_mask_remap,
                                                 get_header=False)
                               +0.5).astype('uint8')

            # perform bitwise OR combination of mask_remap and
            # keep track of number of zeros
            if nimage==0:
                data_mask_OR = data_mask_remap
                data_sumzeros = (data_mask_remap==0).astype(int)
            else:
                data_mask_OR |= data_mask_remap
                data_sumzeros += (data_mask_remap==0).astype(int)


            if not keep_tmp:
                files2remove = image_associates(image)
                zogy.remove_files (files2remove, verbose=True)




        # combined mask, starting from the OR combination of all masks
        data_mask_out = np.copy(data_mask_OR)

        # but if pixels are not masked, i.e. equal to zero, in at
        # least 1/3 of the images with a bare minimum of 1, then set
        # combined mask also to zero
        mask_zeros = (data_sumzeros >= max(1,len(imtable)/3))
        data_mask_out[mask_zeros] = 0




    if False:
        # only in case of multiple images
        if len(imagelist) > 1:

            # try to clean up erratic mask edges by setting all
            # columns and rows that have less than some fraction of
            # the maximum width of non-edge pixels to edge pixels
            value_edge = mask_value['edge']
            mask_edge = (data_mask_out==value_edge)
            dy = np.sum(np.any(~mask_edge, axis=1))
            dx = np.sum(np.any(~mask_edge, axis=0))

            frac = 0.9
            mask_y = np.sum(~mask_edge, axis=1) < frac*dx
            mask_x = np.sum(~mask_edge, axis=0) < frac*dy

            data_mask_out[:,mask_x] = value_edge
            data_mask_out[mask_y,:] = value_edge



    # fill_holes and binary_close saturated pixels using function
    # blackbox.fill_sat_holes; mask is updated in place
    bb.fill_sat_holes (data_mask_out, mask_value)


    # set pixels that are in the actual combined edge to [value_edge],
    # while leaving pixels marked as edge in the valid image area
    # untouched

    # mask with pixels that contain a mix of [value_edge] and some
    # other mask value
    value_edge = mask_value['edge']
    mask_edge = (data_mask_out==value_edge)
    mask_edgemix = ((data_mask_out & value_edge == value_edge) & ~mask_edge)
    # perform binary_propagation of mask_edge into this mixed mask
    struct = np.ones((3,3), dtype=bool)
    mask_edge = ndimage.binary_propagation(mask_edge, structure=struct,
                                           mask=mask_edgemix)
    # set real edge pixels to single value in output mask
    data_mask_out[mask_edge] = value_edge
    # and to zero in output data
    data_out[mask_edge] = 0



    # convert combined weights image to standard deviation
    data_weights_out, header_weights_out = zogy.read_hdulist(fits_weights_out,
                                                             get_header=True,
                                                             dtype='float32')
    mask_zero_cw = (data_weights_out == 0)
    data_bkg_std_out = np.copy(data_weights_out)
    data_bkg_std_out[~mask_zero_cw] = 1./np.sqrt(data_weights_out[~mask_zero_cw])
    # replace zeros with maximum value
    data_bkg_std_out[mask_zero_cw] = np.amax(data_bkg_std_out[~mask_zero_cw])


    # if saturated pixels are considered to be bad pixels, i.e.  they
    # are given a weight of zero, then use zogy.fixpix to interpolate
    # over them
    value_sat = mask_value['saturated']
    if masktype_discard & value_sat == value_sat:
        base = fits_out.split('.fits')[0]
        data_out = zogy.fixpix (data_out, satlevel=saturate_eff,
                                data_bkg_std=data_bkg_std_out,
                                data_mask=data_mask_out, header=header_out,
                                base=base, mask_value=mask_value,
                                keep_tmp=get_par(set_br.keep_tmp,tel),
                                # along column as reference image has been
                                # re-oriented to North up, East left
                                along_row=False, interp_func='gauss')


    # add pixels that have zero weights in the combined weights
    # image as bad pixels, only if not already masked
    value_bad = mask_value['bad']
    mask_bad2add = (mask_zero_cw & (data_mask_out==0))
    data_mask_out[mask_bad2add] += 1

    log.info ('{} zero-weight pixels in combined image, of which {} are not '
              'coinciding with already flagged pixels'
              .format(np.sum(mask_zero_cw), np.sum(mask_bad2add)))

    # set values of these additional bad pixels to zero in output
    # image
    #data_out[mask_bad2add] = 0


    if False:
        if len(imtable) != 2:
            data_mask_out[mask_bad2add] += 1
        else:
            # for combination of 2 images, replace these
            # pixels in the data with the minimum
            data_out[mask_bad2add] = data_min[mask_bad2add]


    # time stamp of writing output files
    ut_now = Time.now().isot
    header_out['DATEFILE'] = (ut_now, 'UTC date of writing file')
    header_out['R-DATE'] = (ut_now, 'time stamp reference image creation')

    # write combined mask and data to fits images, with header of
    # combined output image
    fits.writeto(fits_mask_out, data_mask_out, header_out, overwrite=True)
    fits.writeto(fits_out, data_out, header_out, overwrite=True)

    # also write separate header fits file
    hdulist = fits.HDUList(fits.PrimaryHDU(header=header_out))
    hdulist.writeto(fits_out.replace('.fits', '_hdr.fits'), overwrite=True)


    # convert data_bkg_std_out to a bkg_std_mini image to save disk space;
    # see also function [get_back] in zogy.py
    bkg_boxsize = get_par(set_zogy.bkg_boxsize,tel)
    # reshape
    nxsubs = int(image_size / bkg_boxsize)
    nysubs = int(image_size / bkg_boxsize)
    data_bkg_std_out_reshaped = data_bkg_std_out.reshape(
        nysubs,bkg_boxsize,-1,bkg_boxsize).swapaxes(1,2).reshape(nysubs,nxsubs,-1)
    # take the non-clipped nanmedian along 2nd axis
    mini_std = np.nanmedian (data_bkg_std_out_reshaped, axis=2)
    # update header with [set_zogy.bkg_boxsize]
    header_weights_out['BKG-SIZE'] = (bkg_boxsize, '[pix] background boxsize '
                                      'used')


    # write mini bkg_std file
    header_weights_out['COMMENT'] = ('combined weights image was converted to '
                                     'STD image: std=1/sqrt(w)')
    ut_now = Time.now().isot
    header_weights_out['DATEFILE'] = (ut_now, 'UTC date of writing file')
    fits.writeto(fits_bkg_std_mini, mini_std.astype('float32'),
                 header_weights_out, overwrite=True)



    if not keep_tmp:
        list_tmp = list(imtable['image_name_tmp'])
        for im in list_tmp:
            files2remove = image_associates(im)
            zogy.remove_files (files2remove, verbose=True)

        # also remove file with clipped pixels
        zogy.remove_files([clip_logname], verbose=True)



    zogy.mem_use ('at end of imcombine')
    log.info ('wall-time spent in imcombine: {}s'.format(time.time()-t0))


    return


################################################################################

def image_associates (image):

    """return list of files associated with input image"""

    base = image.split('_red.fits')[0]
    image_remap = '{}_red_remap.fits'.format(base)
    image_resamp = '{}_red_resamp.fits'.format(base)
    image_resamp_weights = '{}_red_resamp.weight.fits'.format(base)
    image_weights = '{}_red_weights.fits'.format(base)
    mask = '{}_mask.fits'.format(base)
    mask_remap = '{}_mask_remap.fits'.format(base)

    return [image, image_remap, image_resamp, image_resamp_weights,
            image_weights, mask, mask_remap]


################################################################################

def mask_objects (imtable):

    imagelist = list(imtable['image_name_red'])
    imagelist_tmp = list(imtable['image_name_tmp'])

    for nimage, image in enumerate(imagelist):

        # read image data
        data = zogy.read_hdulist(image)

        # get background STD image
        image_temp_bkg_std = imagelist_tmp[nimage].replace('red.fits',
                                                           'red_bkg_std.fits')
        # if present, read it from the temp folder
        if os.path.exists(image_temp_bkg_std):

            # read it from the full background image
            data_bkg_std = zogy.read_hdulist (image_temp_bkg_std,
                                              dtype='float32')

        # otherwise extract it from the mini background STD image
        else:

            # read mini background STD image
            image_bkg_std_mini = image.replace('red.fits',
                                               'red_bkg_std_mini.fits')
            data_bkg_std_mini, header_bkg_std_mini = zogy.read_hdulist(
                image_bkg_std_mini, get_header=True, dtype='float32')

            # convert mini STD to full background STD image
            bkg_boxsize = header_bkg_std_mini['BKG-SIZE']
            data_bkg_std = zogy.mini2back (
                data_bkg_std_mini, data.shape, order_interp=3,
                bkg_boxsize=bkg_boxsize, interp_Xchan=False, timing=False)


        # define box sizes, nsigmas and maximum number of outliers to
        # use in [pass_filters] below; the minimum value of [fsigma] is
        # used as nsigma to create the initial mask
        fsize = [  7, 1 ]
        fsigma = [ 2, 4 ]
        fmax = [   7, 1 ]
        #fsize = [    7,   3,   1 ]
        #fsigma = [ 1.5, 2.5, 3.5 ]
        #fmax = [     7,   3,   1 ]


        # nsigma image
        nsigma = min(fsigma)
        data_nsigma = data/data_bkg_std
        mask_obj = (np.abs(data_nsigma) > nsigma)

        # construct table
        ysize, xsize = data.shape
        xy = range(1, ysize+1)
        xx, yy = np.meshgrid(xy, xy)
        table_im = Table([xx[mask_obj], yy[mask_obj], data_nsigma[mask_obj]],
                         names=('x', 'y', 'nsigma'))

        # use function [pass_filters] to improve [mask_obj] with
        # sliding boxes
        t0 = time.time()
        mask_obj = pass_filters (table_im, fsize, fsigma, fmax, data.shape)
        log.info ('created mask for {} in {:.2f}s'
                  .format(image, time.time()-t0))


        # set these pixels to zero in the weights image
        weights_tmp = imagelist_tmp[nimage].replace('.fits','_weights.fits')
        with fits.open(weights_tmp, 'update', memmap=True) as hdulist:
            hdulist[-1].data[mask_obj] = 0


        if False:
            sigmas_masked = np.copy(data_nsigma)
            sigmas_masked[mask_obj] = 0
            zogy.ds9_arrays(data_mo=data, data_nsigma_mo=data_nsigma,
                            mask_obj_mo=mask_obj.astype(int),
                            sigmas_masked_mo=sigmas_masked)


    return


################################################################################

def pick_images (seeing, max_spread=0.3):

    # number of seeing values
    nvalues = len(seeing)

    # initial mask
    mask_use = np.zeros(nvalues, dtype=bool)

    # seeing values sorted
    seeing_sort = np.sort(seeing)

    for i, val in enumerate(seeing_sort):

        # create mask, starting from the current seeing value,
        # including all values that are within [max_spread]
        mask_tmp = ((np.abs(seeing/val-1) <= max_spread) &
                    (seeing >= val))

        # if new mask contains more entries than before, adopt it as
        # the mask to use
        if np.sum(mask_tmp) > np.sum(mask_use):
            mask_use = mask_tmp


    if False:

        # use combination of 2 images only if seeing values are very
        # close; if they are not, revert to using single best-seeing
        # image.  Even with seeing 2.6 and 2.9, i.e. spread of 0.11,
        # resulting Scorr image doesn't appear bad but still shows
        # some fake transients - better just take weighted average
        # with 2 images
        if np.sum(mask_use)==2:
            spread = np.amax(seeing[mask_use]) / np.amin(seeing[mask_use]) - 1
            if spread > 0.05:
                # use best seeing image
                mask_use = np.zeros(nvalues, dtype=bool)
                mask_use[np.argsort(seeing)[0]] = True


    log.info ('{}/{} images picked in [pick_images] with seeing values:\n{}'
              .format(np.sum(mask_use), mask_use.size, seeing[mask_use]))

    return mask_use


################################################################################

def calc_headers (combine_type, imtable):

    nimages = len(imtable)
    gain = np.mean(imtable['gain'])
    mjd_obs = np.mean(imtable['mjd_obs'])

    if combine_type == 'sum':

        rdnoise = np.sqrt(np.sum(imtable['rdnoise']**2))
        saturate = np.sum(imtable['saturate'])
        exptime = imtable['exptime'][0] * nimages

    else:

        rdnoise = np.sqrt(np.sum(imtable['rdnoise']**2)) / nimages
        saturate = np.amin(imtable['saturate'])
        # all images have been scaled in flux to the 1st image, so
        # effective exposure time is that of the 1st image
        exptime = imtable['exptime'][0]

    return gain, rdnoise, saturate, exptime, mjd_obs


################################################################################

def get_A_swarp (imagelist, nsigma_range=np.arange(2.5, 10, 0.1),
                 A_range=np.arange(0.3, 10, 0.1), psf_size=99, Nlimit=0):

    """Given a list of images to combine, calculate the number of expected
    outliers around bright stars, using the PSFEx-determined PSFs of
    the input images, as a function of A and nsigma. For increasing
    values of A and nsigma, the set of images is determined that
    produces [Nlimit] outliers. The mask of images and the values of A
    and nsigma are returned.

    The A value is the same as the parameter CLIP_AMPFRAC (fraction of
    flux variation allowed with clipping) to use in the CLIPPED
    combination in SWarp. Based on Gruen et al. 2014
    (https://ui.adsabs.harvard.edu/abs/2014PASP..126..158G/abstract)
    and their PSFHomTest program.

    """

    # A_range or nsigma could also be floats or integers; convert to
    # list
    if isinstance(nsigma_range, (int, float)):
        nsigma_range = [nsigma_range]
    if isinstance(A_range, (int, float)):
        A_range = [A_range]


    # if imagelist contains less than 3 images, no need to continue
    # any further
    nimages = len(imagelist)
    if nimages < 3:
        return np.ones(nimages, dtype=bool), A_range[0], nsigma_range[0]


    # pixel coordinates in the 1st image at which to extract the PSFs
    header = zogy.read_hdulist(imagelist[0], get_data=False, get_header=True)
    xsize, ysize = header['NAXIS1'], header['NAXIS2']
    low, high, cntr = int(xsize/8), int(xsize*7/8), int(xsize/2)
    # use 4 corners and the center
    #pixcoords = [(low, low), (high, low), (high, high), (low, high), (cntr, cntr)]
    # use the center only
    pixcoords = [(cntr, cntr)]
    ncoords = len(pixcoords)

    # array to record background STD read from header and the peak
    # value of the PSF images
    bkg_std = np.zeros(nimages)

    # initialize [data_psf] with shape (nimages, ncoords, psf_size**2)
    data_psf = np.zeros ((nimages, ncoords, psf_size**2), dtype='float32')

    # coordinate mask to be able to avoid including a PSF that is off
    # a particular image after the coordinate transformation
    mask_coord_off = np.zeros((nimages, ncoords, psf_size**2), dtype=bool)


    # make 2D x,y grid of pixel coordinates for Gauss PSF, which is to
    # test if the same results as shown in Fig. 6 from Gruen et
    # al. can be reached
    gauss_test = False
    if gauss_test:
        xy = range(1, psf_size+1)
        xx, yy = np.meshgrid(xy, xy, indexing='ij')
        # lognormal distribution of Gaussion with FWHM of 4 pixels and
        # spread in log10(FWHM) of 0.05, as in Gruen et al. paper
        rng = np.random.default_rng()
        fwhm_gauss = 4 * rng.lognormal(0, 0.05, nimages)
        #print ('fwhm_gauss: {}'.format(fwhm_gauss))
        sigma_gauss = fwhm_gauss / 2.355


    # loop images
    for nimage, image in enumerate(imagelist):

        # read header
        header = zogy.read_hdulist(image, get_data=False, get_header=True)

        # background STD
        bkg_std[nimage] = header['S-BKGSTD']

        # infer name of psfex binary table from [image]
        psfex_bintable = '{}_psf.fits'.format(image.split('.fits')[0])

        # need to remember WCS solution of first image
        if nimage == 0:
            wcs_first = WCS(header)
        else:
            wcs = WCS(header)

        # loop different coordinates on the image
        for ncoord, coord in enumerate(pixcoords):

            if nimage == 0:
                xcoord, ycoord = coord
            else:
                # transform pixel coordinates of 1st image to current
                # image using the header WCS
                ra, dec = wcs_first.all_pix2world(coord[0], coord[1], 1)
                xcoord, ycoord = wcs.all_world2pix(ra, dec, 1)


            # check if coordinates are off the image
            if xcoord < 1 or xcoord > xsize or ycoord < 1 or ycoord > ysize:
                mask_coord_off[nimage, ncoord, :] = True
                continue


            # read in PSF output binary table from psfex, containing the
            # polynomial coefficient images, and various PSF parameters using
            # the function [extract_psf_datapars] in zogy.py
            results = zogy.extract_psf_datapars (psfex_bintable, verbose=False)
            (data, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
             psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg) = results


            if not gauss_test:
                # extract PSF
                psf_ima, __ = zogy.get_psf_ima (data, xcoord, ycoord, psf_size,
                                                psf_samp, polzero1, polscal1,
                                                polzero2, polscal2, poldeg)
                #psf_ima[psf_ima<0] = 0
                #psf_ima /= np.sum(psf_ima)

            else:
                sigma = sigma_gauss[nimage]
                x0 = y0 = int(psf_size/2)
                psf_ima = zogy.EllipticalGauss2D (xx, yy, x0=x0, y0=y0,
                                                  sigma1=sigma, sigma2=sigma,
                                                  theta=0, amplitude=1,
                                                  background=0)
                # normalize
                psf_ima /= np.sum(psf_ima)


            # record as 1D array in [data_psf]
            data_psf[nimage, ncoord] = psf_ima.ravel()




    # create masked array from data_psf and mask_coord_off
    data_psf_masked = np.ma.masked_array(data_psf, mask=mask_coord_off)

    # calculate median images at the different coordinates
    data_psf_median = np.median(data_psf_masked, axis=0)

    # total flux STD; background STD plus Poisson noise from profile
    #print ('bkg_std: {}'.format(bkg_std))
    # scale flux_tot such that peak of object is around 1e5 e-
    #print ('psf_peak: {}'.format(psf_peak))
    psf_peak = np.median(np.amax(data_psf_masked, axis=2))
    flux_tot = np.sum(5e4 / psf_peak)
    #print ('flux_tot: {}'.format(flux_tot))
    flux_std = np.sqrt(flux_tot * data_psf_median
                       + (bkg_std**2).reshape(nimages,1,1))
    # it would be the same to boost bkg_std with a factor
    # flux_std = bkg_std * np.sqrt(1 + data_psf_median * flux_tot / bkg_std)
    # as done in PSFHomTest.cpp

    # arrays used within the loop below
    data_diff = data_psf - data_psf_median
    abs_data_psf_median = np.abs(data_psf_median)


    # loop nsigma and A:
    for nsigma, A in itertools.product(nsigma_range, A_range):

        # calculate number of outlier pixels in images; [outlier] has
        # same shape as [data_psf]: (nimages, ncoords, psf_size**2)
        # see Eq. 14 from Gruen et al. paper
        outlier = (np.maximum(np.abs(data_diff) - A*abs_data_psf_median, 0)
                   * (flux_tot / flux_std))

        # set outliers related to coordinates that are off the image
        # to zero
        outlier[mask_coord_off] = 0

        # negative whenever data_diff is negative
        mask_neg = (data_diff < 0)
        outlier[mask_neg] *= -1

        # initialize Nmax
        Nmax = np.zeros((2, nimages, ncoords), dtype=int)

        # above threshold
        Nmax[0] = np.sum(outlier > nsigma, axis=2)
        # below threshold
        Nmax[1] = np.sum(outlier < -nsigma, axis=2)

        # maxima per image; shape of N_outliers is (nimages)
        N_outliers = np.amax(np.amax(Nmax, axis=0), axis=1)

        # log number of outliers for each image
        for nimage, image in enumerate(imagelist):
            log.info ('A:{:.1f}, nsigma:{:.1f}, {} outliers for image {}'
                      .format(A, nsigma, N_outliers[nimage], image))

        # any outlying images?
        mean, median, std = sigma_clipped_stats(N_outliers)
        mask_imagelist = (np.abs(N_outliers - median) <= 3*std)
        log.info('mean: {:.2f}, median: {}, std: {:.2f}'
                 .format(mean, median, std))

        imfrac_use = np.sum(mask_imagelist)/mask_imagelist.size
        N_outliers_max = np.amax(N_outliers[mask_imagelist])

        if (imfrac_use >= 2/3 and N_outliers_max <= Nlimit and
            np.sum(mask_imagelist) >= 2):

            log.info ('images selected in [get_A_swarp]: {}'
                      .format(np.array(imagelist)[mask_imagelist]))
            log.info ('A_swarp: {:.2f}, nsigma: {:.2f}'
                      .format(A, nsigma))
            break

        else:
            # revert to full list in case this is last iteration
            mask_imagelist = np.ones_like(N_outliers, dtype=bool)


    # issue error if Nlimit was not reached
    if N_outliers_max > Nlimit:
        log.error ('desired Nlimit was not reached in [get_A_swarp]; '
                   'N_outliers_max={} for A={} and nsigma={}'
                   .format(N_outliers_max, A, nsigma))


    return mask_imagelist, A, nsigma


################################################################################

def clipped2mask (clip_logname, imagelist, nsigma_clip, fits_ref):

    """Given the file with the clipped pixels (in the frame of the
    combined image) created by SWarp [clip_logname] and a list of
    reduced images [imagelist], this function will create a mask from
    the clipped pixels in the frame of the input images, where the
    clipped pixels are filtered using boxes of shapes (3,3), (10,10)
    and (50,50). Based on Gruen et al. 2014
    (https://ui.adsabs.harvard.edu/abs/2014PASP..126..158G/abstract)
    and their MaskMap program.

    """

    header = zogy.read_hdulist(imagelist[0], get_data=False, get_header=True)
    xsize, ysize = header['NAXIS1'], header['NAXIS2']

    # filter definitions
    fsize = [            5, 1 ]
    fsigma = [ nsigma_clip, 4 ]
    fmax = [             4, 1 ]


    # read clip_logname file created by SWarp
    table = ascii.read(clip_logname, format='fast_no_header', data_start=0,
                       names=['nfile', 'x', 'y', 'nsigma'])

    # keep only the entries above minimum sigma
    mask_keep = np.abs(table['nsigma']) > min(fsigma)
    table = table[mask_keep]

    # read ref image header
    hdr_ref = zogy.read_hdulist(fits_ref, get_data=False, get_header=True)


    # convert pixel coordinates to RA,DEC
    wcs_ref = WCS(hdr_ref)
    ra_ref, dec_ref = wcs_ref.all_pix2world(table['x'], table['y'], 1)


    # loop imagelist; that list is assumed to correspond to the
    # integers in the first column of [clip_logname]
    nimages = len(set(table['nfile']))
    if nimages != len(imagelist):
        # log error if these numbers are not consistent
        log.error ('#images in imagelist in [clipped2mask]: {} is not '
                   'consistent with #images in first column of {}: {}'
                   .format(len(imagelist), clip_logname, nimages))


    for nimage, image in enumerate(imagelist):

        # part of table relevant for this image
        mask_im = (table['nfile']==nimage)
        table_im = table[mask_im]

        # if empty, continue with next image
        if len(table_im)==0:
            continue

        # convert ra, dec coordinates from the reference frame (the
        # subset relevant for the current image) to pixel coordinates
        # in the frame of the current image
        hdr_im = zogy.read_hdulist(image, get_data=False, get_header=True)


        wcs_im = WCS(hdr_im)
        x_im, y_im = wcs_im.all_world2pix(ra_ref[mask_im], dec_ref[mask_im], 1)
        # update table_im coordinates with integers of x_im and y_im
        table_im['x'] = (x_im+0.5).astype(int)
        table_im['y'] = (y_im+0.5).astype(int)

        # discard objects beyond edges, if any
        mask_keep = ((table_im['x'] >= 1) & (table_im['x'] <= xsize) &
                     (table_im['y'] >= 1) & (table_im['y'] <= ysize))
        table_im = table_im[mask_keep]


        # use [pass_filters] to convert [table_im] with x, y, nsigma
        # into a boolean mask in which pixels to be masked are True
        t0 = time.time()
        mask_im = pass_filters (table_im, fsize, fsigma, fmax, (ysize, xsize))


        # make sure not to include pixels near saturated stars
        data_mask = (zogy.read_hdulist(image.replace('_red.fits', '_mask.fits'))
                     .astype('uint8'))
        mask_dict = get_par(set_zogy.mask_value,tel)
        # iterate over all mask values
        mask_sat = np.zeros(data_mask.shape, dtype=bool)
        for key in mask_dict.keys():
            # add saturated and saturated-connected pixels to [mask_sat]
            if 'saturated' in key:
                val = mask_dict[key]
                mask_sat |= (data_mask & val == val)

        # indices of [mask_sat]=True pixels
        (y_sat, x_sat) = np.nonzero(mask_sat)
        # indices of [mask_im]=True pixels
        (y_im, x_im) = np.nonzero(mask_im)
        dist2_limit = (5*header['PSF-FWHM'])**2
        for i in range(np.sum(mask_sat)):
            dist2 = (x_im - x_sat[i])**2 + (y_im - y_sat[i])**2
            mask_dist = (dist2 <= dist2_limit)
            if np.sum(mask_dist) > 0:
                mask_im[y_im[mask_dist], x_im[mask_dist]] = False


        log.info ('created mask for {} in {:.2f}s'.format(image, time.time()-t0))


        if False:
            # create image with nsigmas
            im_sigmas = np.zeros((ysize,xsize), dtype='float32')
            im_sigmas[table_im['y']-1,table_im['x']-1] = table_im['nsigma']

            data = zogy.read_hdulist(image)
            zogy.ds9_arrays (data_c2m=data, im_sigmas_c2m=im_sigmas,
                             mask_im_c2m=mask_im.astype(int))


        # update corresponding weights image, i.e. set weights value
        # at [mask_im] to zero
        weights_tmp = image.replace('.fits','_weights.fits')
        with fits.open(weights_tmp, 'update', memmap=True) as hdulist:
            hdulist[-1].data[mask_im] = 0


    return


################################################################################

def pass_filters (table_im, fsize, fsigma, fmax, mask_shape):

    # make sure input fsize, fsigma and fmax are lists
    if not isinstance(fsize, list):
        fsize = list(fsize)

    if not isinstance(fsigma, list):
        fsigma = list(fsigma)

    if not isinstance(fmax, list):
        fmax = list(fmax)


    # initialize mask image to return
    mask_im = np.zeros(mask_shape, dtype=bool)

    # loop filters
    for nf in range(len(fsize)):

        # select table entries with nsigma above fsigma[nf]
        mask_nsigma = (np.abs(table_im['nsigma']) > fsigma[nf])
        table_im_filt = table_im[mask_nsigma]

        # discard entries that were already masked previously
        mask_masked = mask_im[table_im_filt['y']-1, table_im_filt['x']-1]
        table_im_filt = table_im_filt[~mask_masked]

        # number of entries left
        ntable = len(table_im_filt)

        # coordinate indices
        x_index = table_im_filt['x'] - 1
        y_index = table_im_filt['y'] - 1

        if fsize[nf] == 1:

            # if filter size is 1, straightforward to add pixels
            # affected to [mask_im]
            mask_tmp = np.zeros(mask_shape, dtype=bool)
            mask_tmp[y_index, x_index] = True
            mask_im |= mask_tmp

        else:

            # initialize counting image, one for negative and one for
            # positive sigmas
            ysize, xsize = mask_shape
            count_im = np.zeros((2, ysize, xsize), dtype='uint16')
            count_index = np.zeros(ntable, dtype='uint16')
            mask_pos = (table_im_filt['nsigma'] > 0)
            count_index[mask_pos] = 1


            # loop table_im_filt entries
            for it in range(ntable):

                # define window with size fsize[nf] with current pixel
                # at lower left
                i0 = x_index[it]
                j0 = y_index[it]
                i1 = min(i0+fsize[nf], xsize)
                j1 = min(j0+fsize[nf], ysize)

                # increase pixel values in this region with 1
                count_im[count_index[it], j0:j1,i0:i1] += 1


            # mask where count_im is above threshold
            mask_count = ((count_im[0] >= fmax[nf]) | (count_im[1] >= fmax[nf]))

            # loop entries in [mask_count] and mask windows to the
            # lower left of them
            (y_index1, x_index1) = np.nonzero(mask_count)
            ntable1 = np.sum(mask_count)
            for it in range(ntable1):

                # define window with size fsize[nf] with current pixel
                # at upper right
                i1 = x_index1[it]+1
                j1 = y_index1[it]+1
                i0 = max(i1-fsize[nf], 0)
                j0 = max(j1-fsize[nf], 0)

                mask_im[j0:j1,i0:i1] = True


    return mask_im


################################################################################

def read_header_alt (header, keywords):

    values = []
    for i in range(len(keywords)):
        if keywords[i] in header:
            values.append(header[keywords[i]])
        else:
            raise RuntimeError ('keyword {} not present in header - change keyword '
                                'name or add manually'.format(keywords[i]))
    return values


################################################################################

def get_par (par, tel):

    """Function to check if [par] is a dictionary with one of the keys
       being [tel] or the alphabetic part of [tel] (e.g. 'BG'), and if
       so, return the corresponding value. Otherwise just return the
       parameter value."""

    par_val = par
    if type(par) is dict:
        if tel in par:
            par_val = par[tel]
        else:
            # cut off digits from [tel]
            tel_base = ''.join([char for char in tel if char.isalpha()])
            if tel_base in par:
                par_val = par[tel_base]

    return par_val


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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='build MeerLICHT/BlackGEM '
                                     'reference images')
    parser.add_argument('--telescope', type=str, default='BG',
                        choices=['ML1', 'BG2', 'BG3', 'BG4', 'BG'],
                        help='telescope name (ML1, BG2, BG3, BG4 or BG); if set '
                        'to BG, files from any BG present in the tables of '
                        '[fits_hdrtable_list] will be mixed into single '
                        'reference image with prefix BG_, even if all those '
                        'files happen to be from the same BG; default=\'BG\'')

    parser.add_argument('--fits_hdrtable_list', type=str, default=None,
                        help='list of one or more (comma-separated) binary fits '
                        'tables, containing header keywords MJD-OBS, OBJECT, '
                        'FILTER, QC-FLAG, RA-CNTR, DEC-CNTR, S-SEEING, LIMMAG '
                        'and S-BKGSTD of the possible images to be included; '
                        'if left to default of None, the catalog header tables '
                        'available for ML and BG will be used')

    parser.add_argument('--date_start', type=str, default=None,
                        help='start date (noon) to include images, date string '
                        '(e.g. yyyymmdd) or days relative to noon today '
                        '(negative number); default=None')

    parser.add_argument('--date_end', type=str, default=None,
                        help='end date (noon) to include images, date string '
                        '(e.g. yyyymmdd) or days relative to noon today '
                        '(negative number); default=None')

    parser.add_argument('--field_IDs', type=str, default=None,
                        help='only consider images with this(these) field ID(s) '
                        '(can be multiple field IDs separated by a comma, '
                        'and with the optional use of unix wildcards, '
                        'e.g. 1600[0-5],16037,161??); can also be an ascii file '
                        'with the field ID(s) in the 1st column; default=None')

    parser.add_argument('--filters', type=str, default=None,
                        help='only consider this(these) filter(s), e.g. uqi')

    parser.add_argument('--ascii_inputfiles', type=str, default=None,
                        help='name of ASCII file with the specific reduced '
                        'image filenames to be used in the co-addition; the '
                        'filenames are assumed to be in the first column '
                        'without any header/column name info and need to '
                        'include the part [telescope]_yyyymmdd_hhmmss; if '
                        'their full names are used they need to end in '
                        '_red.fits.fz; default=None')

    parser.add_argument('--go_deep', type=str2bool, default=False,
                        help='use all images available (i.e. neglect values of '
                        '[set_buildref.limmag_target])')

    parser.add_argument('--qc_flag_max', type=str, default='orange',
                        choices=['green', 'yellow', 'orange', 'red'],
                        help='worst QC flag to consider; default=\'orange\'')

    parser.add_argument('--seeing_max', type=float, default=None,
                        help='[arcsec] maximum seeing to consider; default=None')

    parser.add_argument('--skip_zogy', type=str2bool, default=False,
                        help='skip execution of zogy on resulting image?; '
                        'default=False')

    #parser.add_argument('--make_colfig', type=str2bool, default=False,
    #                    help='make color figures from uqi filters?; '
    #                    'default=False')

    parser.add_argument('--filters_colfig', type=str, default='iqu',
                        help='set of 3 filters to use for RGB color figures; '
                        'default=\'uqi\'')

    parser.add_argument('--mode_ref', type=str2bool, default=False,
                        help='original reference image mode, where results are '
                        'saved in reference folder defined in BlackBOX settings '
                        'file; if set to False, [results_dir] is used for the '
                        'output folder; default=False')

    parser.add_argument('--results_dir', type=str, default='.',
                        help='output directory with resulting images, separated '
                        'in subfolders equal to the field IDs; only relevant if '
                        '[mode_ref] is False; default: \'.\'')

    parser.add_argument('--extension', type=str, default=None,
                        help='extension to add to default field ID subfolder; '
                        'only relevant if [mode_ref] is False; default=None')

    parser.add_argument('--keep_tmp', default=None,
                        help='keep temporary directories')

    args = parser.parse_args()

    # make sure fits_hdrtable_list is a list
    if args.fits_hdrtable_list is not None:
        fits_hdrtable_list = args.fits_hdrtable_list.split(',')
    else:
        fits_hdrtable_list = args.fits_hdrtable_list


    buildref (telescope = args.telescope,
              fits_hdrtable_list = fits_hdrtable_list,
              date_start = args.date_start,
              date_end = args.date_end,
              field_IDs = args.field_IDs,
              filters = args.filters,
              ascii_inputfiles = args.ascii_inputfiles,
              go_deep = args.go_deep,
              qc_flag_max = args.qc_flag_max,
              seeing_max = args.seeing_max,
              skip_zogy = args.skip_zogy,
              #make_colfig = args.make_colfig,
              filters_colfig = args.filters_colfig,
              mode_ref = args.mode_ref,
              results_dir = args.results_dir,
              extension = args.extension,
              keep_tmp = args.keep_tmp)


################################################################################
