
import os
import set_buildref as set_br

# setting environment variable OMP_NUM_THREADS to number of threads,
# (used by e.g. astroscrappy); needs to be done before numpy is
# imported in [zogy]. However, do not set it when running a job on the
# ilifu cluster as it is set in the job script and that value would
# get overwritten here
cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
if cpus_per_task is None:
    os.environ['OMP_NUM_THREADS'] = str(set_br.nthreads)
else:
    # not really necessary - already done in cluster batch script
    os.environ['OMP_NUM_THREADS'] = str(cpus_per_task)

from zogy import *

import re, fnmatch
import itertools
import math

from astropy.coordinates import Angle
from astropy.table import vstack

# trouble installing fitsio in singularity container so create
# fallback on astropy.io.fits (imported in zogy import above)
#try:
#    import fitsio
#    use_fitsio = True
#except:
#    use_fitsio = False
use_fitsio = False
    
from multiprocessing import Pool, Manager, Lock, Queue, Array

import aplpy

from blackbox import date2mjd, unzip
from blackbox import get_par, already_exists, copy_files2keep
from blackbox import create_log, close_log, define_sections, make_dir
from blackbox import pool_func, fpack, create_jpg, clean_tmp
from qc import qc_check, run_qc_check

import set_zogy
import set_blackbox as set_bb


__version__ = '0.7.0'


################################################################################

def buildref (telescope=None, fits_table=None, table_only=None, date_start=None,
              date_end=None, field_IDs=None, filters=None, qc_flag_max=None,
              seeing_max=None, make_colfig=False, filters_colfig='iqu',
              extension=None):


    """Module to consider one specific or all available field IDs within a
    specified time range, and to combine the available images of that
    field ID in one or all filters, using those images that satisfy
    the quality flag and seeing constraint. The combining of the
    images is done using the function imcombine.
    
    The resulting reference image is put through zogy.py as the
    reference image and the corresponding reference directory is
    prepared.

    """
    

    global tel, genlog, lock, max_qc_flag, max_seeing, start_date, end_date
    global time_refstart, ext
    tel = telescope
    lock = Lock()
    max_qc_flag = qc_flag_max
    max_seeing = seeing_max
    start_date = date_start
    end_date = date_end
    ext = extension


    # define number of processes or tasks [nproc]; when running on the
    # ilifu cluster the environment variable SLURM_NTASKS should be
    # set through --ntasks-per-node in the sbatch script; otherwise
    # use the value from the set_br settings file
    slurm_ntasks = os.environ.get('SLURM_NTASKS')
    if slurm_ntasks is not None:
        nproc = int(slurm_ntasks)
    else:
        nproc = int(get_par(set_br.nproc,tel))

    # update nthreads in set_br with value of environment variable
    # 'OMP_NUM_THREADS' set at the top
    if int(os.environ['OMP_NUM_THREADS']) != get_par(set_br.nthreads,tel):
        set_br.nthreads = int(os.environ['OMP_NUM_THREADS'])


    # record starting time to add to header
    time_refstart = Time.now().isot
    
    # initialize logging
    if not os.path.isdir(get_par(set_bb.log_dir,tel)):
        os.makedirs(get_par(set_bb.log_dir,tel))

    genlogfile = '{}/{}_{}_buildref.log'.format(get_par(set_bb.log_dir,tel), tel,
                                                Time.now().strftime('%Y%m%d_%H%M%S'))
    genlog = create_log (genlogfile)


    genlog.info ('building reference images')
    genlog.info ('log file: {}'.format(genlogfile))
    genlog.info ('number of processes: {}'.format(nproc))
    genlog.info ('number of threads:   {}'.format(get_par(set_br.nthreads,tel)))
    genlog.info ('telescope:           {}'.format(telescope))
    genlog.info ('date_start:          {}'.format(date_start))
    genlog.info ('date_end:            {}'.format(date_end))
    genlog.info ('field_IDs:           {}'.format(field_IDs))
    genlog.info ('filters:             {}'.format(filters))
    genlog.info ('qc_flag_max:         {}'.format(qc_flag_max))
    genlog.info ('seeing_max:          {}'.format(seeing_max))
    genlog.info ('make_colfig:         {}'.format(make_colfig))
    if make_colfig:
        genlog.info ('filters_colfig:      {}'.format(filters_colfig))
    genlog.info ('folder extension:    {}'.format(extension))

    
    t0 = time.time()


    # read or prepare a table with filenames and relevant header keywords
    # -------------------------------------------------------------------

    # if provided and it exists, read input table with header info
    if fits_table is not None and os.path.isfile(fits_table):
        
        genlog.info ('reading existing header table: {}'.format(fits_table))
        table = Table.read(fits_table)
    
    else:

        red_path = get_par(set_bb.red_dir,tel)
        filenames = glob.glob('{}/*/*/*/*_red.fits*'.format(red_path))
        nfiles = len(filenames)
        genlog.info ('total number of files: {}'.format(nfiles))


        # split into [nproc] lists
        list_of_filelists = []
        index = np.linspace(0,nfiles,num=nproc+1).astype(int)
        for i in range(nproc):
            list_of_filelists.append(filenames[index[i]:index[i+1]])


        # use function pool_func_lists to read headers from files
        # using multiple processes and write them to a table
        results = pool_func_lists (header2table, list_of_filelists,
                                   log=genlog, nproc=nproc)
        # stack separate tables in results
        table = vstack(results)

        genlog.info ('file headers read in {:.2f}s'.format(time.time()-t0))

        # write to file
        if fits_table is not None:
            genlog.info ('writing table to file {}'.format(fits_table))
            table.write(fits_table, format='fits', overwrite=True)
        else:
            genlog.warning ('not writing table to file')



    if table_only:
        genlog.info ('[table_only] parameter is set to True; nothing left to do')
        logging.shutdown()
        return


    # filter table entries based on date, field_ID, filter, qc-flag and seeing
    # ------------------------------------------------------------------------

    genlog.info ('number of files with all required keywords: {}'
                 .format(len(table)))

    
    if date_start is not None or date_end is not None:

        # filter this list by converting the date and time contained in
        # the filenames to MJD and check if this is consistent with the
        # range specified by the input [date_start] and [date_end]
        mjd_start = set_date (date_start)
        mjd_end = set_date (date_end, start=False)
        # convert dates and times in filenames to MJDs (accurate to the second)
        mjd_files = np.array([date2mjd (f.split('/')[-1].split('_')[1],
                                        time_str=f.split('/')[-1].split('_')[2])
                              for f in table['FILE']])
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
        genlog.info ('number of files left (date_start/end cut): {}'
                     .format(len(table)))


    
    # if object (field ID) is specified, which can include the unix
    # wildcards * and ?, select only images with a matching object
    # string; field_IDs can also be an ascii file with the field ID(s)
    # listed in the 1st column
    filter_list = None
    if field_IDs is not None:
        
        # check if it's a file
        if os.path.isfile(field_IDs):

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
        genlog.info ('number of files left (FIELD_ID cut): {}'.format(len(table)))


    # if filter(s) is specified, select only images with filter(s)
    # specified
    if filters is not None:
        #mask = [table['FILTER'][i] in filters for i in range(len(table))]
        mask = [filt in filters for filt in table['FILTER']]
        table = table[mask]
        genlog.info ('number of files left (FILTER cut): {}'.format(len(table)))
        
        
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
        genlog.info ('number of green: {}, yellow: {}, orange: {}, red: {}'
                     .format(np.sum(mask_green), np.sum(mask_yellow),
                             np.sum(mask_orange), np.sum(mask_red)))
        
        # strip table color from spaces
        mask = [table['QC-FLAG'][i].strip() in qc_col for i in range(len(table))]
        table = table[mask]
        genlog.info ('number of files left (QC-FLAG cut): {}'.format(len(table)))


    # if max_seeing is specified, select only images with the same or
    # better seeing
    if max_seeing is not None:
        mask = (table['PSF-SEE'] <= max_seeing)
        table = table[mask]
        genlog.info ('number of files left (SEEING cut): {}'.format(len(table)))


    # if centering is set to 'grid' in buildref settings file, read
    # the file that contains the ML/BG field grid definition, that
    # will be used to fill [radec_list] in the loop below
    center_type = get_par(set_br.center_type,tel)
    if center_type == 'grid':
        # read from grid definition file located in ${ZOGYHOME}/CalFiles
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

    # loop objects
    for n_obj, obj in enumerate(objs_uniq):
        
        # skip fields '00000' and those beyond 20,000
        #if int(obj) == 0 or int(obj) >= 20000:
        if int(obj) == 0 or int(obj) >= 20000:
            continue

        # table mask of this particular field_ID
        mask_obj = (table['OBJECT'] == obj)
        

        # determine image center based on [center_type]        
        if center_type == 'grid':
            # for 'grid' centering, let [radec] refer to a tuple pair
            # containing the RA and DEC coordinates
            mask_grid = (table_grid['field_id'].astype(int) == int(obj))
            if np.sum(mask_grid) > 0:
                radec = (table_grid['ra_c'][mask_grid][0],
                         table_grid['dec_c'][mask_grid][0])
            else:
                genlog.error ('field ID/OBJECT {} not present in ML/BG '
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
            genlog.info ('number of files left for {} in filter {}: {}'
                         .format(obj, filt, nfiles))
            # if no files left, continue
            if nfiles == 0:
                continue

            # sort files based on their LIMMAG, highest value first
            indices_sort = np.argsort(table[mask]['LIMMAG'])[::-1]
            limmags_sort = table[mask]['LIMMAG'][indices_sort]
            seeing_sort = table[mask]['PSF-SEE'][indices_sort]
            bkgstd_sort = table[mask]['S-BKGSTD'][indices_sort]
            files_sort = table[mask]['FILE'][indices_sort]


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
            limmag_target = get_par(set_br.limmag_target,tel)[filt]
            # add dmag to target magnitude to account for the fact
            # that the projected limiting magnitude will be
            # somewhat higher than the actual one
            dmag = 0.15
            mask_sort_cum = (limmags_sort_cum <= limmag_target + dmag)
            # use a minimum number of files, adding 1 to images
            # selected above
            nmin = get_par(set_br.nimages_min,tel)
            nuse = max (np.sum(mask_sort_cum)+1, nmin)
            # [nuse] should not be larger than number of images available
            nuse = min (nuse, nfiles)
            # update mask
            mask_sort_cum[0:nuse] = True
            
            
            # files that were excluded
            nfiles_excl = np.sum(~mask_sort_cum)
            if nfiles_excl > 0:
                files_2exclude = files_sort[~mask_sort_cum]
                limmags_2exclude = limmags_sort[~mask_sort_cum]
                seeing_2exclude = seeing_sort[~mask_sort_cum]
                genlog.warning ('files ({}) and their limmags excluded from '
                                '{}-band coadd:'.format(nfiles_excl, filt))
                for i in range(len(files_2exclude)):
                    genlog.info ('{}, {:.3f}'
                                 .format(files_2exclude[i],
                                         limmags_2exclude[i]))

                    
            # files to combine
            files_2coadd = files_sort[mask_sort_cum]
            limmags_2coadd = limmags_sort[mask_sort_cum]
            seeing_2coadd = seeing_sort[mask_sort_cum]
            bkgstd_2coadd = bkgstd_sort[mask_sort_cum]

            nfiles_used = nfiles - nfiles_excl
            genlog.info ('files ({}), limmags, projected cumulative limmags '
                         'and seeing used for {}-band coadd:'
                         .format(nfiles-nfiles_excl, filt))
            for i in range(len(files_2coadd)):
                genlog.info ('{} {:.3f} {:.3f} {:.2f}'
                             .format(files_2coadd[i], limmags_2coadd[i],
                                     limmags_sort_cum[i], seeing_2coadd[i]))

            limmag_proj = limmags_sort_cum[mask_sort_cum][-1]
            
            genlog.info ('projected (target) {}-band limiting magnitude of '
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


    if len(table)==0:
        genlog.warning ('no field IDs with sufficient number of good images to '
                        'process')
        logging.shutdown()
        return


    # multiprocess remaining files
    # ----------------------------

    # feed the lists that were created above to the multiprocessing
    # helper function [pool_func_lists] that will arrange each
    # process to call [prep_ref] to prepare the reference image
    # for a particular field and filter combination, using
    # the [imcombine] function
    result = pool_func_lists (prep_ref, list_of_imagelists, obj_list,
                              filt_list, radec_list, imagesize_list, nfiles_list,
                              limmag_proj_list, log=genlog, nproc=nproc)


    # make color figures
    # ------------------
    if make_colfig:
        genlog.info ('preparing color figures')
        # also prepare color figures
        try:
            result = pool_func (prep_colfig, objs_uniq, filters_colfig, genlog,
                                log=genlog, nproc=nproc)
        except Exception as e:
            #genlog.exception (traceback.format_exc())
            genlog.exception ('exception was raised during [pool_func]: {}'
                              .format(e))
            raise RuntimeError

    # not needed anymore; fpacking and jpegging is now done inside
    # BB's [copy_files2keep]
    if False:
        
        # fpack reference fits files
        # --------------------------
        genlog.info ('fpacking reference fits images')
        ref_dir = get_par(set_bb.ref_dir,tel)
        list_2pack = []
        for field_ID in objs_uniq:
            ref_path = '{}/{:0>5}'.format(ref_dir, field_ID)
            if ext is not None:
                ref_path = '{}{}'.format(ref_path, ext)
                
            list_2pack.append(glob.glob('{}/*.fits'.format(ref_path)))
            list_2pack.append(glob.glob('{}/Old/*.fits'.format(ref_path)))

        # unnest nested list_2pack
        list_2pack = list(itertools.chain.from_iterable(list_2pack))
        
        # use [pool_func] to process the list
        result = pool_func (fpack, list_2pack, genlog, log=genlog, nproc=nproc)


        # create jpg images for reference frames
        # --------------------------------------
        genlog.info ('creating jpg images')
        # create list of files to jpg
        list_2jpg = []
        for field_ID in objs_uniq:
            ref_path = '{}/{:0>5}'.format(ref_dir, field_ID)
            if ext is not None:
                ref_path = '{}{}'.format(ref_path, ext)
                
            list_2jpg.append(glob.glob('{}/*_red.fits.fz'.format(ref_path)))

        # unnest nested list_2pack
        list_2jpg = list(itertools.chain.from_iterable(list_2jpg))

        # use [pool_func] to process the list
        result = pool_func (create_jpg, list_2jpg, genlog, log=genlog, nproc=nproc)


    logging.shutdown()
    return
    
    
################################################################################

def calc_imsize (ra, dec, ra0, dec0, imsize, bkg_size, pixscale):
    
    # calculate maximum offset in RA and DEC to determine
    # output imagesize (square) that include all images for
    # this object, and expanding it for the size to contain a
    # multiple of the background boxsize
    offset_ra = np.amax(np.abs(haversine(ra, dec, ra0, dec0)))
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
            mjd = date2mjd ('{}'.format(date), time_str='12:00')

    return mjd


################################################################################

def pool_func_lists (func, list_of_imagelists, *args, log=None, nproc=1):
    
    try:
        results = []
        pool = Pool(nproc)
        for nlist, filelist in enumerate(list_of_imagelists):
            args_temp = [filelist]
            for arg in args:
                args_temp.append(arg[nlist])
                
            results.append(pool.apply_async(func, args_temp))

        pool.close()
        pool.join()
        results = [r.get() for r in results]
        #if log is not None:
        #    log.info ('result from pool.apply_async: {}'.format(results))
        return results
        
    except Exception as e:
        if log is not None:
            #log.exception (traceback.format_exc())
            log.exception ('exception was raised during [pool.apply_async({})]: '
                           '{}'.format(func, e))

        raise RuntimeError
    

################################################################################

def header2table (filenames):

    # initialize rows
    rows = []

    # keywords to add to table
    keys = ['MJD-OBS', 'OBJECT', 'FILTER', 'QC-FLAG', 'RA-CNTR', 'DEC-CNTR',
            'PSF-SEE', 'LIMMAG', 'S-BKGSTD']
    keys_dtype = [float, 'U5', 'U1', 'U6', float, float, float, float, float]


    # loop input list of filenames
    for filename in filenames:

        # check if filename exists, fpacked or not
        exists, filename = already_exists (filename, get_filename=True)
        if exists:
            try:
                if use_fitsio:
                    # read header; use fitsio as it is faster than
                    # astropy.io.fits on when not using solid state disk
                    with fitsio.FITS(filename) as hdulist:
                        h = hdulist[-1].read_header()
                else:
                    with fits.open(filename) as hdulist:
                        h = hdulist[-1].header
            except Exception as e:
                genlog.exception ('trouble reading header; skipping image {}'
                                  .format(filename))
                continue
                        
        else:
            genlog.warning ('file does not exist; skipping image {}'
                            .format(filename))
            continue


        # check if all keywords present before appending to table
        mask_key = [key not in h for key in keys]
        if np.any(mask_key):
            genlog.warning ('keyword(s) {} not in header; skipping image {}'
                            .format(np.array(keys)[mask_key], filename))
            continue


        if False:
            # up to v0.9.2, it was possible for the reduced image
            # header not to contain a red flag, while the catalog file
            # did contain it - this happened when zogy was not
            # processed properly (e.g. astrometry.net failed), then
            # dummy catalogs were produced but the reduced image
            # header was not updated - this bug was fixed in v0.9.2.
            if '.fz' in filename:
                catname = filename.replace('.fits.fz', '_cat.fits')
            else:
                catname = filename.replace('.fits', '_cat.fits')
            # if it doesn't exist, continue with the next
            if not os.path.isfile(catname):
                genlog.warning ('catalog file {} does not exist; skipping '
                                'image {}'.format(catname, filename))
                continue

            # read header
            if use_fitsio:
                with fitsio.FITS(catname) as hdulist:
                    h_cat = hdulist[-1].read_header()
            else:
                with fits.open(catname) as hdulist:
                    h_cat = hdulist[-1].header

            # if dummycats were created, copy the qc-flag to the
            # reduced image header
            if 'DUMMYCAT' in h_cat and h_cat['DUMMYCAT']:
                key = 'QC-FLAG'
                if key in h_cat:
                    h[key] = h_cat[key]
                    #genlog.info ('h[{}]: {}, h_cat[{}]: {}'
                    #             .format(key, h[key], key, h_cat[key]))
                # need to copy the qc-red??, qc-ora??, qc-yel?? as well
                qc_col = ['red', 'orange', 'yellow']
                for col in qc_col:
                    key_base = 'QC{}'.format(col[:3]).upper()
                    for i in range(1,100):
                        key = '{}{}'.format(key_base, i)
                        if key in h_cat:
                            h[key] = h_cat[key]
                            #genlog.info ('h[{}]: {}, h_cat[{}]: {}'
                            #             .format(key, h[key], key, h_cat[key]))
                            

        # skip for now
        if False:
                            
            # check if flag of particular colour was set in the
            # image-subtraction stage; if yes, then promote the flag
            # colour as that flag is not relevant to the image itself and
            # the image should be used in building the reference image
            qc_col = ['red', 'orange', 'yellow', 'green']
            for col in qc_col:
                # check if current colour is the same as the QC flag
                if h['QC-FLAG'] == col and col != 'green':
                    # loop keywords with this flag; potentially 100
                    key_base = 'QC{}'.format(col[:3]).upper()
                    for i in range(1,99):
                        key_temp = '{}{}'.format(key_base, i)
                        if key_temp in h:
                            # if keyword value does not start with 'Z-',
                            # 'T-' or 'V-', then break; the QC-FLAG will
                            # not get updated
                            if not h[key_temp].startswith(('Z-', 'T-', 'V-')):
                                break

                    else: # associated to the for loop!
                        # all of the flags' keywords were due to image subtraction
                        # stage, so promote the QC-FLAG
                        h['QC-FLAG'] = qc_col[qc_col.index(col)+1]
                        genlog.info ('updating QC-FLAG from {} to {} for image {}'
                                     .format(col, h['QC-FLAG'], filename))



        # for surviving files, prepare row of filename and header values
        row = [filename]
        for key in keys:
            row.append(h[key])
        # append to rows
        rows.append(row)


    # create table from rows
    names = ['FILE']
    dtypes = ['U100']
    for i_key, key in enumerate(keys):
        names.append(key)
        dtypes.append(keys_dtype[i_key])

    if len(rows) == 0:
        # rows without entries: create empty table
        table = Table(names=names, dtype=dtypes)
    else: 
        table = Table(rows=rows, names=names, dtype=dtypes)

        
    return table


################################################################################

def prep_colfig (field_ID, filters, log=None):
    
    # determine reference directory and file
    ref_path = '{}/{:0>5}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    # for the moment, add _alt to this path to separate it from
    # existing reference images
    if ext is not None:
        ref_path = '{}{}'.format(ref_path, ext)

    # header keyword to use for scaling (e.g. PC-ZP or LIMMAG)
    key = 'LIMMAG'
    
    # initialize rgb list of images
    images_rgb = []
    images_std = []
    images_zp = []
    
    for filt in filters:

        image = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)
        exists, image = already_exists(image, get_filename=True)
        
        if not exists:
            if log is not None:
                log.info ('{} does not exist; not able to prepare color '
                          'figure for field_ID {}'.format(image, field_ID))
                return
        else:
            
            # add to image_rgb list (unzip if needed)
            image = unzip(image, put_lock=False)
            images_rgb.append(image)
            
            # read image data and header
            data, header = read_hdulist(image, get_header=True)
            
            # determine image standard deviation
            mean, median, std = sigma_clipped_stats(data)
            images_std.append(std)
            
            # read header zeropoint
            if key in header:
                images_zp.append(header[key])
            else:
                if log is not None:
                    log.info ('missing header keyword {}; not able to '
                              'prepare color figure for field_ID {}'
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
    colfig = '{}/{}_{}_{}.png'.format(ref_path, tel, field_ID, filters)
    aplpy.make_rgb_image(images_rgb, colfig,
                         vmin_r=vmin_r, vmax_r=vmax_r,
                         vmin_g=vmin_g, vmax_g=vmax_g,
                         vmin_b=vmin_b, vmax_b=vmax_b)
    

    
################################################################################

def prep_ref (imagelist, field_ID, filt, radec, image_size, nfiles, limmag_proj):
    
    # determine and create reference directory
    ref_path = '{}/{:0>5}'.format(get_par(set_bb.ref_dir,tel), field_ID)
    make_dir (ref_path, lock=lock)
    
    # name of output file, including full path
    ref_fits_out = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)

    # if reference image already exists, check if images used are the
    # same as the input [imagelist]
    exists, ref_fits_temp = already_exists (ref_fits_out, get_filename=True)
    if exists:

        genlog.warning ('reference image {} already exists; not remaking it'
                        .format(ref_fits_out))
        return
        

        # this block below was used previously to only remake the
        # reference image if new individual files were available
        if False:

            genlog.info ('reference image {} already exists; checking if it '
                         'needs updating'.format(ref_fits_out))
            # read header
            header_ref = read_hdulist (ref_fits_temp, get_data=False,
                                       get_header=True)
            # check how many images were used
            if 'R-NUSED' in header_ref:
                n_used = header_ref['R-NUSED']
            else:
                n_used = 1
            
            # gather used images into list
            if 'R-IM1' in header_ref:
                imagelist_used = [header_ref['R-IM{}'.format(i+1)]
                                  for i in range(n_used)
                                  if 'R-IM{}'.format(i+1) in header_ref]
            
            # compare input [imagelist] with [imagelist_used]; if they are
            # the same, no need to build this particular reference image
            # again
            imagelist_new = [image.split('/')[-1].split('.fits')[0]
                             for image in imagelist]
            if set(imagelist_new) == set(imagelist_used):
                # same sets of images, return
                genlog.info ('imagelist_new: {}'.format(imagelist_new))
                genlog.info ('imagelist_used: {}'.format(imagelist_used))
                genlog.info ('reference image of {} in filter {} with same '
                             'set of images already present; skipping'
                             .format(field_ID, filt))
                return

        
    # prepare temporary folder
    tmp_path = ('{}/{:0>5}/{}'
                .format(get_par(set_bb.tmp_dir,tel), field_ID,
                        ref_fits_out.split('/')[-1].replace('.fits','')))
    make_dir (tmp_path, empty=True, lock=lock)
    
    # names of tmp output fits and its mask
    ref_fits = '{}/{}'.format(tmp_path, ref_fits_out.split('/')[-1])
    ref_fits_mask = ref_fits.replace('red.fits','mask.fits')


    # RA and DEC center of output image
    ra_center, dec_center = radec
        

    # create logfile specific to this reference image in tmp folder
    # (to be copied to final output folder at the end)
    logfile = ref_fits.replace('.fits', '.log')
    log = create_log (logfile, name='log')
    log.info ('logfile created: {}'.format(logfile))
    

    # check if sufficient images available to combine
    if len(imagelist) == 0:

        log.error ('no images available to combine for field ID {} in filter {}'
                   .format(field_ID, filt))
        clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel), log=log)
        close_log(log, logfile)
        return

    else:
    

        if len(imagelist) == 1:
            log.warning ('only a single image available for field ID {} in filter {}'
                         '; using it as the reference image'.format(field_ID, filt))


        # run imcombine
        log.info('running imcombine; outputfile: {}'.format(ref_fits))

        try:
            imcombine (field_ID, imagelist, ref_fits,
                       get_par(set_br.combine_type,tel), filt,
                       masktype_discard = get_par(set_br.masktype_discard,tel),
                       tempdir = tmp_path,
                       ra_center = ra_center,
                       dec_center = dec_center,
                       image_size = image_size,
                       nfiles = nfiles,
                       limmag_proj = limmag_proj,
                       back_type = get_par(set_br.back_type,tel),
                       back_size = get_par(set_zogy.bkg_boxsize,tel),
                       back_filtersize = get_par(set_zogy.bkg_filtersize,tel),
                       remap_each = False,
                       swarp_cfg = get_par(set_zogy.swarp_cfg,tel),
                       nthreads = get_par(set_br.nthreads,tel),
                       log = log)

        except Exception as e:
            #log.exception (traceback.format_exc())
            log.exception ('exception was raised during [imcombine]: {}'
                           .format(e))
            clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel), log=log)
            close_log(log, logfile)
            raise RuntimeError


        # run zogy on newly prepared reference image
        try:
            zogy_processed = False
            header_optsub = optimal_subtraction(
                ref_fits=ref_fits, ref_fits_mask=ref_fits_mask,
                set_file='set_zogy', log=log, verbose=None,
                nthreads=get_par(set_br.nthreads,tel), telescope=tel)
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

                clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel), log=log)
                close_log(log, logfile)
                return
            

        log.info('zogy_processed: {}'.format(zogy_processed))


        # copy/move files to the reference folder
        tmp_base = ref_fits.split('_red.fits')[0]
        ref_base = ref_fits_out.split('_red.fits')[0]

        # (re)move old reference files
        oldfiles = glob.glob('{}*'.format(ref_base))
        if len(oldfiles)!=0:
            if False:
                # remove them
                for f in oldfiles:
                    os.remove(f)
            else:
                # or move them to an Old folder instead
                old_path = '{}/Old/'.format(ref_path)
                make_dir (old_path, lock=lock)
                for f in oldfiles:
                    f_dst = '{}/{}'.format(old_path,f.split('/')[-1])
                    shutil.move (f, f_dst)

                
        # now move [ref_2keep] to the reference directory
        result = copy_files2keep(tmp_base, ref_base,
                                 get_par(set_bb.ref_2keep,tel), move=False,
                                 log=log)


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
                       nthreads=get_par(set_br.nthreads,tel), log=log)

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

    if False:
        
        # feed [header_optsub] to [run_qc_check] and check for a red flag
        qc_flag = run_qc_check (header_optsub, tel, log=log)
        qc_flag = 'green'
    
        if qc_flag != 'red':
            tmp_base = ref_fits.split('.fits')[0]
            # now move [ref_2keep] to the reference directory
            ref_base = ref_fits_out.split('.fits')[0]
            result = copy_files2keep(tmp_base, ref_base,
                                     get_par(set_bb.ref_2keep,tel),
                                     move=False, log=log)
            log.info('finished making reference image: {}'.format(ref_fits_out))

        else:
            log.info('encountered red flag; not using image: {} as reference'
                     .format(ref_fits))


    clean_tmp(tmp_path, get_par(set_br.keep_tmp,tel), log=log)
    close_log(log, logfile)
    return
    
        
################################################################################

def imcombine (field_ID, imagelist, fits_out, combine_type, filt, overwrite=True,
               masktype_discard=None, tempdir='.temp', ra_center=None,
               dec_center=None, image_size=None, nfiles=0, limmag_proj=None,
               use_wcs_center=True, back_type='auto', back_default=0,
               back_size=120, back_filtersize=3, resample_suffix='_resamp.fits',
               remap_each=False, remap_suffix='_remap.fits', swarp_cfg=None,
               nthreads=0, log=None):


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

    else:
        # if combine_type is clipped and the background pixels of
        # non-selected images based on the seeing range are not used,
        # then already define the final subset to be used here,
        # avoiding unnecessary work in the image loop futher below
        use_bkg_discarded = get_par(set_br.use_bkg_discarded,tel)
        if combine_type.lower() == 'clipped' and not use_bkg_discarded:
            seeing_list = []
            for nimage, image in enumerate(imagelist):
                header = read_hdulist(image, get_data=False, get_header=True)
                seeing_list.append(header['PSF-SEE'])
                
            max_spread = get_par(set_br.max_spread_seeing,tel)
            mask_use = pick_images (seeing_list, max_spread=max_spread)

            # temporary mask to select specific images
            if False:
                mask_use = np.array([True,     #2.630
                                     False,     #2.701
                                     False,    #2.859
                                     False,    #2.933
                                     False,    #4.108
                                     False,    #3.617
                                     False,    #6.860
                                     False])   #6.967

            imagelist = list(np.array(imagelist)[mask_use])


        log.info ('{} images selected to combine: {}'.format(len(imagelist),
                                                             imagelist))


    # clean up or make temporary directory if it is not the current directory '.'
    if tempdir[-1]=='/':
        tempdir = tempdir[0:-1]
    if tempdir != '.':
        if os.path.isdir(tempdir):
            if os.listdir(tempdir):
                log.info ('cleaning temporary directory {}'.format(tempdir))
                cmd = 'rm {}/*'.format(tempdir)
                result = subprocess.call(cmd, shell=True)
        else:
            cmd = ['mkdir','{}'.format(tempdir)]
            result = subprocess.call(cmd)


    # if SWarp configuration file does not exist, create default one in [tempdir]
    if swarp_cfg is None:
        swarp_cfg = tempdir+'/swarp.config'
        cmd = 'swarp -d > {}'.format(swarp_cfg)
        result = subprocess.call(cmd, shell=True)
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


    # initialize table with image values to keep
    nimages = len(imagelist)
    names = ('ra_center', 'dec_center', 'xsize', 'ysize', 'zp', 'airmass', 'gain',
             'rdnoise', 'saturate', 'exptime', 'mjd_obs', 'fscale', 'seeing',
             'pixscale', 'image_name_red', 'image_name_tmp', 'mask_name_tmp')
    dtypes = ('f8', 'f8', 'i4', 'i4', 'f8', 'f8', 'f8',
              'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
              'f8', 'U100', 'U100', 'U100')
    data_tmp = np.zeros((nimages, len(names)))
    imtable = Table(data=data_tmp, names=names, dtype=dtypes)

    # mask in case particular image is not used in loop below
    mask_keep = np.ones(len(imtable), dtype=bool)

    # loop input list of images
    for nimage, image in enumerate(imagelist):
        
        if not os.path.isfile(image):
            raise RuntimeError ('input image {} does not exist'.format(image))

        # read input image data and header
        data, header = read_hdulist(image, get_header=True)
        
        # read corresponding mask image
        image_mask = image.replace('red.fits', 'mask.fits')
        data_mask, header_mask = read_hdulist(image_mask, get_header=True,
                                              dtype='uint8')

        # read relevant header keywords
        keywords = ['naxis1', 'naxis2', 'ra', 'dec', 'pc-zp', 'pc-zpstd',
                    'airmass', 'pc-extco', 'gain', 'rdnoise', 'saturate',
                    'exptime', 'mjd-obs', 'psf-see', 'a-pscale']
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
                data_bkg_mini, header_bkg_mini = read_hdulist(image_bkg_mini,
                                                              get_header=True,
                                                              dtype='float32')
                # convert mini to full background image
                bkg_boxsize = header_bkg_mini['BKG-SIZE']
                data_bkg = mini2back (data_bkg_mini, data.shape, 
                                      order_interp=3, bkg_boxsize=bkg_boxsize,
                                      interp_Xchan=True, timing=False, log=log)


            # read mini background STD image
            image_bkg_std_mini = image.replace('red.fits', 'red_bkg_std_mini.fits')
            data_bkg_std_mini, header_bkg_std_mini = read_hdulist(image_bkg_std_mini,
                                                                  get_header=True,
                                                                  dtype='float32')

            # convert mini STD to full background STD image
            bkg_boxsize = header_bkg_std_mini['BKG-SIZE']
            data_bkg_std = mini2back (data_bkg_std_mini, data.shape,
                                      order_interp=3, bkg_boxsize=bkg_boxsize,
                                      interp_Xchan=False, timing=False, log=log)

            # save as probably used later on
            image_temp = '{}/{}'.format(tempdir, image.split('/')[-1])
            image_temp_bkg_std = image_temp.replace('red.fits', 'red_bkg_std.fits')
            image_temp_bkg_std = image_temp_bkg_std.replace('.fz','')
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
            image_temp = unzip(image_temp, put_lock=False)
            image_mask_temp = unzip(image_mask_temp, put_lock=False)

            # run source-extractor
            base = image_temp.split('.fits')[0]
            sexcat = '{}_ldac.fits'.format(base)
            fwhm = seeing / pixscale
            #fwhm = header['S-FWHM']
            imtype = 'new'
            sex_params = get_par(set_zogy.sex_par,tel)
            try:
                result = run_sextractor(
                    image_temp, sexcat, get_par(set_zogy.sex_cfg,tel),
                    sex_params, pixscale, log, header, fit_psf=False,
                    return_fwhm_elong=False, fraction=1.0, fwhm=fwhm,
                    update_vignet=False, imtype=imtype,
                    fits_mask=image_mask_temp, npasses=2, tel=tel,
                    set_zogy=set_zogy)
                
            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [run_sextractor]: {}'
                              .format(e))

            # read source-extractor output image data and header
            data, header = read_hdulist(image_temp, get_header=True)

            # check if background was subtracted this time
            if 'BKG-SUB' in header and header['BKG-SUB']:
                bkg_sub = True
            else:
                bkg_sub = False

            if not bkg_sub:
                # read background image created in [run_sextractor]
                image_temp_bkg = '{}_bkg.fits'.format(base)
                data_bkg = read_hdulist (image_temp_bkg, dtype='float32')

            image_temp_bkg_std = '{}_bkg_std.fits'.format(base)
            data_bkg_std = read_hdulist (image_temp_bkg_std, dtype='float32')



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
        # iterate over all mask values
        for val in mask_value.values():
            # check if this one is to be discarded
            if masktype_discard & val == val:
                mask_discard = (data_mask & val == val)
                mask_weights[mask_discard] = True
                log.info('discarding mask value {}; no. of pixels: {}'
                         .format(val, np.sum(mask_discard)))


        # set corresponding pixels to zero in data_weights
        data_weights[mask_weights] = 0

        # fix pixels
        data = fixpix (data, log, satlevel=saturate, data_mask=data_mask,
                       imtype='new', mask_value=mask_value)
        
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
                data_bkg_var_comb = (fscale * data_bkg_std)**2
            else:
                data_bkg_var_comb += (fscale * data_bkg_std)**2


        # determine and record image centers
        if use_wcs_center:
            # determine WCS center of field
            wcs = WCS(header)
            ra_temp, dec_temp = wcs.all_pix2world(int(xsize/2), int(ysize/2), 1)
            #print ('center RA, DEC', ra_temp, dec_temp)
            #print ('center RA, DEC', Angle(ra_temp, unit=u.deg).to_string(u.hour),
            #       Angle(dec_temp, unit=u.deg).to_string(u.deg))

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

            # set edge pixel values of image to zero, otherwise those
            # pixels will be negative and will/may end up in the edge
            # of the combined image
            data[data_mask==mask_value['edge']] = 0

        elif back_type == 'constant':
            # subtract one single value from the image: clipped median
            data_mean, data_median, data_std = sigma_clipped_stats (data)
            data -= data_median
            # make sure edge pixels are zero
            data[data_mask==mask_value['edge']] = 0



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



    # clean imtable from images that were not used
    imtable = imtable[mask_keep]


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
    # is put through zogy.py
    keys2copy = ['XBINNING', 'YBINNING', 'RADESYS', 'EPOCH', 'FLIPSTAT',
                 'OBJECT', 'IMAGETYP', 'FILTER',
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
    cmd_dict['-RESAMPLE_SUFFIX'] = resample_suffix
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

                # use [pick_images] to select subset of images with
                # seeing values within a maximum spread
                max_spread = get_par(set_br.max_spread_seeing,tel)
                mask_use = pick_images (imtable['seeing'], max_spread=max_spread,
                                        log=log)

                # determine A to use to ensure bright stars are not
                # being clipped
                imagelist = list(imtable['image_name_red'][mask_use])
                Alist = get_par(set_br.A_range,tel)
                if len(Alist)==2: Alist.append(1)
                A_range= np.arange(Alist[0], Alist[1], Alist[2])
                # set size of PSF image to extract to the maximum allowed by
                # [size_vignet] in the zogy settings file
                psf_size = get_par(set_zogy.size_vignet,tel)
                nsigma_clip = get_par(set_br.nsigma_clip,tel)
                N_outliers = get_A_swarp (imagelist, A_range=A_range,
                                          nsigma_range=nsigma_clip,
                                          psf_size=psf_size, log=log)

                # mask where N_outliers sufficiently small
                mask_Nzero = (N_outliers[:,0]<=0)

                # only do clipping pass if at least 2 images available
                # and there is a value for A within A_range for which
                # N_outliers is zero
                if np.sum(mask_use) >= 2 and np.sum(mask_Nzero) > 0:
                    
                    # SWarp subset of images
                    cmd_dict['swarp'] = ','.join(
                        imtable['image_name_tmp'][mask_use])

                    # determine A to use
                    A_swarp = A_range[mask_Nzero][0]
                    #A_swarp = 0.3
                    log.info ('A_swarp: {:.2f}'.format(A_swarp))
                    cmd_dict['-CLIP_AMPFRAC'] = str(A_swarp)

                    # save clipped pixels in ASCII file
                    clip_logname = '{}/clipped.dat'.format(tempdir)
                    cmd_dict['-CLIP_LOGNAME'] = clip_logname
                    cmd_dict['-CLIP_WRITELOG'] = 'Y'

                    # clipping threshold
                    cmd_dict['-CLIP_SIGMA'] = str(nsigma_clip)

                else:
                    
                    # use all images
                    cmd_dict['swarp'] = ','.join(imtable['image_name_tmp'])

                    # no good subset of images could be found, so skip
                    # the CLIPPED pass
                    cmd_dict['-COMBINE_TYPE'] = 'WEIGHTED'   
                    # turn off logging of clipped pixels
                    cmd_dict['-CLIP_WRITELOG'] = 'N'


            else:
                
                # use function [clipped2mask] to convert clipped
                # pixels identified by SWarp, saved in [clip_logname],
                # to masks in the individual image frames that were
                # used [mask_use], filter them with a few sliding
                # windows and update the weights images in the tmp
                # folder
                imagelist_tmp = list(imtable['image_name_tmp'][mask_use])
                clipped2mask (clip_logname, imagelist_tmp, fits_out, log=log)

                if get_par(set_br.use_bkg_discarded,tel):
                    # for the images that were not used in the clipping
                    # pass, set all the object pixels to zero in the
                    # corresponding weights images
                    mask_objects (imtable[~mask_use], log=log)
                    imagelist_tmp = imtable['image_name_tmp']

                else:
                    # selection of imtable for use below in making of
                    # masks and header info
                    imtable = imtable[mask_use]

                    
                # update images to use
                cmd_dict['swarp'] = ','.join(imagelist_tmp)

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


        # no need to do 2nd pass in case combine_type in 1st pass was
        # not set to 'clipped'
        if cmd_dict['-COMBINE_TYPE'].lower() != 'clipped':
            break



    # update header of fits_out
    data_out, header_out = read_hdulist(fits_out, get_header=True)

    # with RA and DEC
    header_out['RA'] = (ra_center, '[deg] telescope right ascension')
    header_out['DEC'] = (dec_center, '[deg] telescope declination')

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
    limmag_target = get_par(set_br.limmag_target,tel)[filt]
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


    # convert combined weights image to standard deviation and save as
    # mini image
    data_weights, header_weights = read_hdulist(fits_weights_out, get_header=True)
    mask_zero_cw = (data_weights == 0)
    data_bkg_std = np.copy(data_weights)
    data_bkg_std[~mask_zero_cw] = 1./np.sqrt(data_weights[~mask_zero_cw])
    # replace zeros with maximum value
    data_bkg_std[mask_zero_cw] = np.amax(data_bkg_std[~mask_zero_cw])


    # set pixels with zeros in combined weights to zero in output image
    data_out[mask_zero_cw] = 0
    

    # time stamp of writing file
    ut_now = Time.now().isot
    header_out['DATEFILE'] = (ut_now, 'UTC date of writing file')
    header_out['R-DATE'] = (ut_now, 'time stamp reference image creation')
    # write file
    fits.writeto(fits_out, data_out.astype('float32'), header_out, overwrite=True)


    if False:
        # save self-built data_bkg_var_comb as STD image to compare
        fits_bkg_std_test = fits_out.replace('.fits', '_bkg_std_test.fits')
        data_bkg_std_test = np.sqrt(data_bkg_var_comb) / nimages
        fits.writeto(fits_bkg_std_test, data_bkg_std_test.astype('float32'),
                     header_weights, overwrite=True)


    # convert this to a bkg_std_mini image to save disk space; see
    # also function [get_back] in zogy.py
    bkg_boxsize = get_par(set_zogy.bkg_boxsize,tel)
    # reshape
    nxsubs = int(image_size / bkg_boxsize)
    nysubs = int(image_size / bkg_boxsize)
    data_bkg_std_reshaped = data_bkg_std.reshape(
        nysubs,bkg_boxsize,-1,bkg_boxsize).swapaxes(1,2).reshape(nysubs,nxsubs,-1)
    # take the non-clipped nanmedian along 2nd axis
    mini_std = np.nanmedian (data_bkg_std_reshaped, axis=2)
    # update header with [set_zogy.bkg_boxsize]
    header_weights['BKG-SIZE'] = (bkg_boxsize, '[pix] background boxsize used')


    # write mini bkg_std file
    header_weights['COMMENT'] = ('combined weights image was converted to STD '
                                 'image: std=1/sqrt(w)')
    header_weights['DATEFILE'] = (ut_now, 'UTC date of writing file')
    fits.writeto(fits_bkg_std_mini, mini_std.astype('float32'),
                 header_weights, overwrite=True)



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
        data_mask_OR = (read_hdulist(fits_mask_OR, get_header=False)
                        +0.5).astype('uint8')
        data_mask_MIN = (read_hdulist(fits_mask_MIN, get_header=False)
                         +0.5).astype('uint8')
    
        # now, wherever mask_MIN is not zero, implying that none of the
        # pixel values in the cube were valid, replace it with the OR mask
        data_mask_comb = data_mask_MIN
        index_nonzero = np.nonzero(data_mask_MIN)
        data_mask_comb[index_nonzero] = data_mask_OR[index_nonzero]

        # add pixels that have zero weights in the combined weights
        # image as bad pixels, only if not already masked
        value_bad = get_par(set_zogy.mask_value,tel)['bad']
        mask_bad2add = (mask_zero_cw & (data_mask_comb==0))
        data_mask_comb[mask_bad2add] += 1
        
        # write combined mask to fits image 
        fits.writeto(fits_mask_out, data_mask_comb, overwrite=overwrite)
    

    else:
        # remapping each individual image if needed
        log.info ('remapping individual images')
        
        # also SWarp individual images, e.g. for colour combination
        refimage = fits_out
        header_refimage = read_hdulist(refimage, get_data=False, get_header=True)

        # initialize combined mask
        mask_array_shape = (len(imtable), image_size, image_size)
        data_mask_array = np.zeros(mask_array_shape, dtype='uint8')
        
        for nimage, image in enumerate(imtable['image_name_tmp']):

            # skip remapping of images themselves for the moment; only
            # needed if some combination of the images other than
            # those available in SWarp is needed
            if False:

                t_temp = time.time()
                image_remap = image.replace('.fits', remap_suffix)
                
                log.info ('refimage: {}'.format(refimage))
                log.info ('image: {}'.format(image))
                log.info ('image_remap: {}'.format(image_remap))
                
                if not os.path.isfile(image_remap):
                    try:
                        result = run_remap (refimage, image, image_remap,
                                            (image_size,image_size),
                                            log=log, config=swarp_cfg,
                                            resample='N', resample_dir=tempdir,
                                            resample_suffix=resample_suffix,
                                            nthreads=nthreads)
                    except Exception as e:
                        #log.exception(traceback.format_exc())
                        log.exception('exception was raised during [run_remap]: '
                                      '{}'.format(e))
                        raise RuntimeError
                    else:
                        log.info ('time spent in run_remap: {}'
                                  .format(time.time()-t_temp))
                        

            # same for image masks if there are any
            if len(imtable) >= nimage:

                image_mask = imtable['mask_name_tmp'][nimage]
                
                log.info ('processing mask: {}'.format(image_mask))
                
                data_mask, header_mask = read_hdulist(image_mask, get_header=True)

                t_temp = time.time()
                image_mask_remap = image_mask.replace('.fits', remap_suffix) 
                if not os.path.isfile(image_mask_remap):

                    try:
                        result = run_remap (refimage, image_mask, image_mask_remap,
                                            (image_size,image_size),
                                            log=log, config=swarp_cfg,
                                            resampling_type='NEAREST',
                                            resample_dir=tempdir,
                                            resample_suffix=resample_suffix,
                                            dtype=data_mask.dtype.name,
                                            value_edge=32,
                                            nthreads=nthreads,
                                            oversampling=0)
                                                  
                    except Exception as e:
                        #log.exception(traceback.format_exc())
                        log.exception('exception was raised during [run_remap]: '
                                      '{}'.format(e))
                        raise RuntimeError
                    else:
                        log.info ('wall-time spent in remapping mask: {}'
                                  .format(time.time()-t_temp))


                t_temp = time.time()

                # alternative way for masks, or images where
                # interpolation is not really needed by simply mapping
                # values from mask image to the refimage using header WCS
                # info
                if False:
                    def trans_func (output_coords):
                        ra_temp, dec_temp = wcs_mask.all_pix2world(
                            output_coords[0], output_coords[1], 0)
                        x, y = wcs_ref.all_world2pix(ra_temp, dec_temp, 0)
                        return (x,y) 
                    
                    global wcs_mask, wcs_ref
                    wcs_mask = WCS(header_mask)
                    wcs_ref = WCS(header_refimage)
                    data_mask_remap_alt = ndimage.geometric_transform(data_mask,
                                                                      trans_func)

                    # write to image
                    fits.writeto(image_mask.replace('.fits',remap_suffix),
                                 data_mask_remap_alt)
                    
                    log.info ('wall-time spent in remapping alt mask: {}'
                              .format(time.time()-t_temp))
                    
                
                # SWarp has converted input mask to float32, so need to
                # read fits image back into integer array to use in
                # combination of masks below
                data_mask_remap = (read_hdulist(image_mask_remap,get_header=False)
                                   +0.5).astype('uint8')

                # perform bitwise OR combination of mask_remap
                if nimage==0:
                    data_mask_OR = data_mask_remap
                else:
                    data_mask_OR = data_mask_OR | data_mask_remap

                # save mask in 3D array
                data_mask_array[nimage] = data_mask_remap

                
        # combine mask, starting from the OR combination of all masks
        data_mask_comb = data_mask_OR

        # but if at least 3 pixels are not masked, i.e. equal to zero,
        # then set combined mask also to zero
        mask_array_zeros = (data_mask_array == 0)
        mask_array_zerosum = np.sum(mask_array_zeros, axis=0)
        log.info ('np.shape(mask_array_zerosum): {}'
                  .format(np.shape(mask_array_zerosum)))
        mask_zero = (mask_array_zerosum >= 3)
        data_mask_comb[mask_zero] = 0

        # write combined mask to fits image 
        fits.writeto(fits_mask_out, data_mask_comb, overwrite=overwrite)
        
        # feed resampled images to function [buildref_optimal]
        #result = buildref_optimal(imagelist)

        
    log.info ('wall-time spent in imcombine: {}s'.format(time.time()-t0))

    return
    

################################################################################

def mask_objects (imtable, log=None):
    
    imagelist = list(imtable['image_name_red'])
    imagelist_tmp = list(imtable['image_name_tmp'])
    
    for nimage, image in enumerate(imagelist):

        # read image data
        data = read_hdulist(image)


        # get background STD image
        image_temp_bkg_std = imagelist_tmp[nimage].replace('red.fits',
                                                           'red_bkg_std.fits')
        # if present, read it from the temp folder
        if os.path.exists(image_temp_bkg_std):

            # read it from the full background image
            data_bkg_std = read_hdulist (image_temp_bkg_std, dtype='float32')

        # otherwise extract it from the mini background STD image
        else:
            
            # read mini background STD image
            image_bkg_std_mini = image.replace('red.fits', 'red_bkg_std_mini.fits')
            data_bkg_std_mini, header_bkg_std_mini = read_hdulist(image_bkg_std_mini,
                                                                  get_header=True,
                                                                  dtype='float32')
            # convert mini STD to full background STD image
            bkg_boxsize = header_bkg_std_mini['BKG-SIZE']
            data_bkg_std = mini2back (data_bkg_std_mini, data.shape,
                                      order_interp=3, bkg_boxsize=bkg_boxsize,
                                      interp_Xchan=False, timing=False, log=log)


        # define box sizes, nsigmas and maximum number of outliers to
        # use in [pass_filters] below; the minimum value of [fsigma] is
        # used as nsigma to create the initial mask
        fsize = [  7, 1 ]
        fsigma = [ 2, 4 ]
        fmax = [   7, 1 ]


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
        mask_obj = pass_filters (table_im, fsize, fsigma, fmax, data.shape,
                                 log=log)
        if log is not None:
            log.info ('created mask for {} in {:.2f}s'
                      .format(image, time.time()-t0))

        
        
        # set these pixels to zero in the weights image
        weights_tmp = imagelist_tmp[nimage].replace('.fits','_weights.fits')
        with fits.open(weights_tmp, 'update') as hdulist:
            hdulist[-1].data[mask_obj] = 0


        if False:
            sigmas_masked = np.copy(data_nsigma)
            sigmas_masked[mask_obj] = 0 
            ds9_arrays(data_mo=data, data_nsigma_mo=data_nsigma,
                       mask_obj_mo=mask_obj.astype(int),
                       sigmas_masked_mo=sigmas_masked)


    return


################################################################################

def pick_images (seeing, max_spread=0.4, log=None):
    
    nvalues = len(seeing)
    mask_use = np.zeros(nvalues, dtype=bool)
    seeing_sort = np.sort(seeing)
    
    nmask = 0
    for i, val in enumerate(seeing_sort):
        mask_tmp = ((np.abs(seeing/val-1) <= max_spread) &
                    (seeing >= val))
        if np.sum(mask_tmp) > np.sum(mask_use):
            mask_use = mask_tmp

    # if less than 3 images selected, pick the two lowest-seeing
    # images
    if np.sum(mask_use) < 3:
        mask_use = np.zeros(nvalues, dtype=bool)
        mask_use[np.argsort(seeing)[0:2]] = True

    if log is not None:
        log.info ('{}/{} images used with seeing values:\n{}'
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
                 A_range=np.arange(0.3, 10, 0.1), psf_size=99, log=None):

    """Given a list of images to combine, calculate the number of expected
    outliers around bright stars, using the PSFEx-determined PSFs of
    the input images, as a function of A and nsigma. A 2D array is
    returned: N_outliers[A, nsigma]. The A value is the same as the
    parameter CLIP_AMPFRAC (fraction of flux variation allowed with
    clipping) to use in the CLIPPED combination in SWarp. Based on
    Gruen et al. 2014
    (https://ui.adsabs.harvard.edu/abs/2014PASP..126..158G/abstract)
    and their PSFHomTest program.

    """

    # A_range or nsigma could also be floats or integers; convert to
    # list
    if isinstance(nsigma_range, (int, float)):
        nsigma_range = [nsigma_range]
    if isinstance(A_range, (int, float)):
        A_range = [A_range]


    # pixel coordinates in the 1st image at which to extract the PSFs;
    # use 4 corners and the center
    header = read_hdulist(imagelist[0], get_data=False, get_header=True)
    xsize, ysize = header['NAXIS1'], header['NAXIS2']
    low, high, cntr = int(xsize/8), int(xsize*7/8), int(xsize/2)
    #pixcoords = [(low, low), (high, low), (high, high), (low, high), (cntr, cntr)]
    pixcoords = [(cntr, cntr)]
    ncoords = len(pixcoords)
    nimages = len(imagelist)

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
        sigma_gauss = fwhm_gauss / (2.355)

    
    # loop images
    for nimage, image in enumerate(imagelist):
        
        # read header
        header = read_hdulist(image, get_data=False, get_header=True)

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
            results = extract_psf_datapars (psfex_bintable)
            (data, header_psf, psf_fwhm, psf_samp, psf_size_config, psf_chi2,
             psf_nstars, polzero1, polscal1, polzero2, polscal2, poldeg) = results


            if not gauss_test:
                # extract PSF
                psf_ima, __ = get_psf_ima (data, xcoord, ycoord, psf_size, psf_samp,
                                           polzero1, polscal1, polzero2, polscal2,
                                           poldeg)
                #psf_ima[psf_ima<0] = 0
                #psf_ima /= np.sum(psf_ima)
                
            else:
                sigma = sigma_gauss[nimage]
                x0 = y0 = int(psf_size/2)
                psf_ima = EllipticalGauss2D (xx, yy, x0=x0, y0=y0,
                                             sigma1=sigma, sigma2=sigma, 
                                             theta=0, amplitude=1, background=0)
                # normalize
                psf_ima /= np.sum(psf_ima)


            # record as 1D array in [data_psf]
            data_psf[nimage, ncoord] = psf_ima.ravel()




    # create masked array from data_psf and mask_coord_off
    data_psf_masked = np.ma.masked_array(data_psf, mask=mask_coord_off)

    # calculate median images at the different coordinates
    data_psf_median = np.median(data_psf_masked, axis=0)

    # total STD; background STD plus Poisson noise from profile
    #print ('bkg_std: {}'.format(bkg_std))
    # scale flux_tot such that peak of object is around 1e5 e-
    #print ('psf_peak: {}'.format(psf_peak))
    psf_peak = np.median(np.amax(data_psf_masked, axis=2))
    flux_tot = np.sum(1e5 / psf_peak)
    #print ('flux_tot: {}'.format(flux_tot))
    std = np.sqrt( flux_tot * data_psf_median + (bkg_std**2).reshape(nimages,1,1))
    # it would be the same to boost bkg_std with a factor
    # std = bkg_std * np.sqrt(1 + data_psf_median * flux_tot / bkg_std)
    # as done in PSFHomTest.cpp

    # initialize N_outliers
    N_outliers = np.zeros((len(A_range), len(nsigma_range)), dtype=int)
    
    data_diff = data_psf - data_psf_median
    abs_data_psf_median = np.abs(data_psf_median)
    
    # loop A
    for i, A in enumerate(A_range):
        
        # calculate number of outlier pixels in images; [outlier] has
        # same shape as [data_psf]: (nimages, ncoords, psf_size**2)
        # see Eq. 14 from Gruen et al. paper
        outlier = (np.maximum(np.abs(data_diff) - A*abs_data_psf_median, 0)
                   * (flux_tot / std))

        # set outliers related to coordinates that are off the image
        # to zero
        outlier[mask_coord_off] = 0
        
        # negative whenever data_diff is negative
        mask_neg = (data_diff < 0)
        outlier[mask_neg] *= -1


        # loop nsigma:
        for j, nsigma in enumerate(nsigma_range):

            # initialize Nmax
            Nmax = np.zeros((2, nimages, ncoords), dtype=int)

            # above threshold
            Nmax[0] = np.sum(outlier > nsigma, axis=2)
            # below threshold
            Nmax[1] = np.sum(outlier < -nsigma, axis=2)

            # maxima per image
            #Nmax_im = np.amax(np.amax(Nmax, axis=0), axis=1)
            # maximum overall
            N_outliers[i,j] = np.amax(Nmax)


        # show outlier images
        if False:
            for nimage in range(nimages):
                for ncoord in range(ncoords):
                    img_tmp = outlier[nimage, ncoord].reshape(psf_size, psf_size)
                    ds9_arrays(img_tmp=img_tmp)


    return N_outliers


################################################################################

def clipped2mask (clip_logname, imagelist, fits_ref, log=None):

    """Given the file with the clipped pixels (in the frame of the
    combined image) created by SWarp [clip_logname] and a list of
    reduced images [imagelist], this function will create a mask from
    the clipped pixels in the frame of the input images, where the
    clipped pixels are filtered using boxes of shapes (3,3), (10,10)
    and (50,50). Based on Gruen et al. 2014
    (https://ui.adsabs.harvard.edu/abs/2014PASP..126..158G/abstract)
    and their MaskMap program.

    """

    header = read_hdulist(imagelist[0], get_data=False, get_header=True)
    xsize, ysize = header['NAXIS1'], header['NAXIS2']
    
    # filter definitions
    nsigma_clip = get_par(set_br.nsigma_clip,tel)
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
    hdr_ref = read_hdulist(fits_ref, get_data=False, get_header=True)
    wcs_ref = WCS(hdr_ref)
    
    # convert pixel coordinates to RA,DEC
    ra_ref, dec_ref = wcs_ref.all_pix2world(table['x'], table['y'], 1)

    
    # loop imagelist; that list is assumed to correspond to the
    # integers in the first column of [clip_logname]
    nimages = len(set(table['nfile']))
    if nimages != len(imagelist):
        # log error if these numbers are not consistent
        if log is not None:
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
        hdr_im = read_hdulist(image, get_data=False, get_header=True)
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
        mask_im = pass_filters (table_im, fsize, fsigma, fmax, (ysize, xsize),
                                log=log)


        # make sure not to include pixels near saturated stars
        data_mask = (read_hdulist(image.replace('_red.fits', '_mask.fits'))
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


        if log is not None:
            log.info ('created mask for {} in {:.2f}s'
                      .format(image, time.time()-t0))

        if False:
            # create image with nsigmas
            im_sigmas = np.zeros((ysize,xsize), dtype='float32')
            im_sigmas[table_im['y']-1,table_im['x']-1] = table_im['nsigma']

            data = read_hdulist(image)
            ds9_arrays (data_c2m=data, im_sigmas_c2m=im_sigmas,
                        mask_im_c2m=mask_im.astype(int))

            
        # update corresponding weights image, i.e. set weights value
        # at [mask_im] to zero
        weights_tmp = image.replace('.fits','_weights.fits')
        with fits.open(weights_tmp, 'update') as hdulist:
            hdulist[-1].data[mask_im] = 0


    return
            

################################################################################

def pass_filters (table_im, fsize, fsigma, fmax, mask_shape, log=None):

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

def buildref_optimal(imagelist):

    start_time1 = os.times()

    # size of image
    ysize = refimage_ysize
    xsize = refimage_xsize

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size,
                                                                       ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    nsubs = centers.shape[0]
    if verbose:
        print ('nsubs', nsubs)

    # number of images
    nimages = len(imagelist)
    
    # prepare arrays;
    # sigma is determined inside nsub loop, so only need 1 dimension
    # of size nimages
    sigma_array = np.ndarray((nimages), dtype='float32')
    # the same goes for fratio in case it is determined across the
    # entire image, but it requires nimages x nsubs shape in case
    # fratio is determined for each subimage.
    fratio_array = np.ndarray((nimages, nsubs), dtype='float32')
    # P and data need 4 dimensions: nimages x nsub x ysize x xsize as
    # P needs to be constructed from the full image initially, then
    # inside the subimage loop the 1st dimension will be summed
    P_array = np.ndarray((nimages, nsubs, ysize_fft, xsize_fft), dtype='float32')
    data_array = np.ndarray((nimages, nsubs, ysize_fft, xsize_fft), dtype='float32')
    
    # these output arrays will be filled with the output summed R and
    # P_R for each subimage
    data_R_full = np.zeros((ysize, xsize), dtype='float32')
    data_PR_full = np.zeros((ysize, xsize), dtype='float32')
    
    # list of keywords to be read from header
    keywords = [key_gain, key_ron, key_satlevel]

    # loop images in imagelist
    for nima, image in enumerate(imagelist):

        if verbose:
            print ('\nimage:', image)
            print ('----------------------------------------------------------------------')

        # base name
        base = image[0:-5]
        # name of resampled image produced by swarp
        image_resampled = image.replace('.fits', resample_suffix)
        
        # read in header of image
        t = time.time()    
        with fits.open(image) as hdulist:
            header = hdulist[0].header
            gain, readnoise, satlevel = read_header_alt (header, keywords)
        if verbose:
            print (keywords)
            print (read_header_alt (header, keywords))
            
        # prepare cubes with shape (nsubs, ysize_fft, xsize_fft) with new,
        # ref, psf and background images
        data_array[nima], P_array[nima] = prep_optimal_subtraction_adapted(
            image, header, image_resampled, nsubs)
        
        # get x, y and fratios from matching PSFex stars across entire frame
        # first set basename of first image
        if nima==0: base1 = base
        x_fratio, y_fratio, fratio, dra, ddec = get_fratio_radec(base1+'.psfexcat',
                                                                 base+'.psfexcat',
                                                                 base1+'.sexcat',
                                                                 base+'.sexcat')

        # loop nsubs to fill fratio_array - also needs to be done in
        # case fratio is determined from the entire image
        for nsub in range(nsubs):
            if nima==0:
                fratio_array[nima, nsub] = 1.
            else:
                # take local or full-frame values for fratio
                subcut = cuts_ima[nsub]
                index_sub = ((y_fratio > subcut[0]) & (y_fratio < subcut[1]) & 
                             (x_fratio > subcut[2]) & (x_fratio < subcut[3]))
                if fratio_local and any(index_sub):
                    # get median fratio from PSFex stars across subimage
                    fratio_mean, fratio_std, fratio_median = clipped_stats(
                        fratio[index_sub])
                else:
                    # else for the entire image
                    fratio_mean, fratio_std, fratio_median = clipped_stats(
                        fratio, nsigma=2)

                # fill fratio_array
                fratio_array[nima, nsub] = fratio_median
                if verbose:
                    print ('fratio_mean, fratio_median, fratio_std',
                           fratio_mean, fratio_median, fratio_std)
            

    start_time2 = os.times()

    print ('\nexecuting optimal co-addition ...')
    
    for nsub in range(nsubs):
    
        if timing: tloop = time.time()
        
        if verbose:
            print ('\nNsub:', nsub+1)
            print ('----------')

        # determine clipped mean, median and stddev in
        # this subimage of all images
        for nima in range(nimages):
            mean, stddev, median = clipped_stats(data_array[nima,nsub], nsigma=3)
            if verbose:
                print ('clipped mean, median, stddev', mean, median, stddev)

            # replace low values in subimages
            data_array[nima,nsub][data_array[nima,nsub] <= 0.] = median

            # subtract the median background
            data_array[nima,nsub] -= median

            # replace saturated pixel values with zero
            #data[nsub][data[nsub] > 0.95*satlevel] = 0.

            # fill sigma_array
            sigma_array[nima] = stddev

            
        # call optimal coaddition function
        data_R, data_PR = run_coaddition(data_array[:,nsub], P_array[:,nsub], sigma_array,
                                         fratio_array[:,nsub], nimages)
        
        # put sub images into output frames
        subcut = cuts_ima[nsub]
        fftcut = cuts_fft[nsub]
        y1 = subimage_border
        x1 = subimage_border
        y2 = subimage_border+subimage_size
        x2 = subimage_border+subimage_size
        data_R_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_R[y1:y2,x1:x2]
        data_PR_full[subcut[0]:subcut[1],subcut[2]:subcut[3]] = data_PR[y1:y2,x1:x2]

        if timing: print ('wall-time spent in nsub loop', time.time()-tloop)

    end_time = os.times()
    dt_usr  = end_time[2] - start_time2[2]
    dt_sys  = end_time[3] - start_time2[3]
    dt_wall = end_time[4] - start_time2[4]
    print
    print ("Elapsed user time in {0}:  {1:.3f} sec".format("optcoadd", dt_usr))
    print ("Elapsed CPU time in {0}:  {1:.3f} sec".format("optcoadd", dt_sys))
    print ("Elapsed wall time in {0}:  {1:.3f} sec".format("optcoadd", dt_wall))
    
    dt_usr  = end_time[2] - start_time1[2]
    dt_sys  = end_time[3] - start_time1[3]
    dt_wall = end_time[4] - start_time1[4]
    print
    print ("Elapsed user time in {0}:  {1:.3f} sec".format("total", dt_usr))
    print ("Elapsed CPU time in {0}:  {1:.3f} sec".format("total", dt_sys))
    print ("Elapsed wall time in {0}:  {1:.3f} sec".format("total", dt_wall))
    
    # write full new, ref, D and S images to fits
    fits.writeto('R.fits', data_R_full, clobber=True)
    fits.writeto('PR.fits', data_PR_full, clobber=True)
        
    # and display
    cmd = ['ds9', '-zscale', 'R.fits', 'PR.fits', 'swarp_combined.fits']
    result = subprocess.call(cmd)
    

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
    
def prep_optimal_subtraction_adapted (image, header, image_resampled, nsubs):

    print ('\nexecuting prep_optimal_subtraction ...')
    t = time.time()
    
    # read in data of resampled image
    with fits.open(image_resampled) as hdulist:
        data = hdulist[0].data
    # get gain, readnoise and pixscale from header
    gain = header[key_gain]
    readnoise = header[key_ron]
    pixscale = header[key_pixscale]
    # convert counts to electrons
    data *= gain

    # determine psf of image (not resampled) with get_psf function
    psf = get_psf_adapted (image, header, nsubs)

    # split full image into subimages
    ysize, xsize = refimage_ysize, refimage_xsize

    # determine cutouts
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(
        subimage_size, ysize, xsize)

    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border
    
    fftdata = np.zeros((nsubs, ysize_fft, xsize_fft), dtype='float32')
    for nsub in range(nsubs):
        subcutfft = cuts_ima_fft[nsub]
        fftcut = cuts_fft[nsub]
        fftdata[nsub][fftcut[0]:fftcut[1],fftcut[2]:fftcut[3]] = data[subcutfft[0]:subcutfft[1],
                                                                      subcutfft[2]:subcutfft[3]]
        
    if timing: print ('wall-time spent in prep_optimal_subtraction', time.time()-t)
    return fftdata, psf
    

################################################################################

def get_psf_adapted (image, ima_header, nsubs):

    """Function that takes in [image] and determines the actual Point
    Spread Function as a function of position from the full frame, and
    returns a cube containing the psf for each subimage in the full
    frame.

    """

    if timing: t = time.time()
    print ('\nexecuting get_psf ...')

    # determine image size from header
    xsize, ysize = ima_header['NAXIS1'], ima_header['NAXIS2']
    
    # run psfex on SExtractor output catalog
    sexcat = image.replace('.fits', '.sexcat')
    psfexcat = image.replace('.fits', '.psfexcat')
    if not os.path.isfile(psfexcat) or redo:
        print ('sexcat', sexcat)
        print ('psfexcat', psfexcat)
        result = run_psfex_adapted (sexcat, psfex_cfg, psfexcat)

    # read in PSF output binary table from psfex
    psfex_bintable = image.replace('.fits', '.psf')
    with fits.open(psfex_bintable) as hdulist:
        header = hdulist[1].header
        data = hdulist[1].data[0][0][:]

    # read in some header keyword values
    polzero1 = header['POLZERO1']
    polzero2 = header['POLZERO2']
    polscal1 = header['POLSCAL1']
    polscal2 = header['POLSCAL2']
    poldeg = header['POLDEG1']
    psf_fwhm = header['PSF_FWHM']
    psf_samp = header['PSF_SAMP']
    # [psf_size_config] is the size of the PSF as defined in the PSFex
    # configuration file ([PSF_SIZE] parameter), which is the same as
    # the size of the [data] array
    psf_size_config = header['PSFAXIS1']
    if verbose:
        print ('polzero1                   ', polzero1)
        print ('polscal1                   ', polscal1)
        print ('polzero2                   ', polzero2)
        print ('polscal2                   ', polscal2)
        print ('order polynomial:          ', poldeg)
        print ('PSF FWHM:                  ', psf_fwhm)
        print ('PSF sampling size (pixels):', psf_samp)
        print ('PSF size defined in config:', psf_size_config)
        
    # call centers_cutouts to determine centers
    # and cutout regions of the full image
    centers, cuts_ima, cuts_ima_fft, cuts_fft, sizes = centers_cutouts(subimage_size, ysize, xsize)
    ysize_fft = subimage_size + 2*subimage_border
    xsize_fft = subimage_size + 2*subimage_border

    # For all images, the PSF was determined from the original image,
    # while it will be applied to the images remapped/resampled to the
    # reference frame. This reference frame is not necessarily that of
    # one of the images. So the centers of the cutouts in the remapped
    # images need to be mapped back to those in the original image to
    # get the PSF from the proper coordinates.  Easiest to do this
    # using astropy.wcs, which would also take care of any potential
    # rotation and scaling.
    
    # first infer ra, dec corresponding to x, y pixel positions
    # (centers[:,1] and centers[:,0], respectively, in the
    # resampled image
    wcs = WCS(image.replace('.fits','_resamp.fits'))
    ra_temp, dec_temp = wcs.all_pix2world(centers[:,1], centers[:,0], 1)
    # then convert ra, dec back to x, y in the original image
    # using the .wcs file produced by Astrometry.net
    wcs = WCS(image.replace('_wcs.fits','.wcs'))
    centers[:,1], centers[:,0] = wcs.all_world2pix(ra_temp, dec_temp, 1)
        
    # [psf_size] is the PSF size in image pixels,
    # i.e. [psf_size_config] multiplied by the PSF sampling (roughly
    # 4-5 pixels per FWHM) which is automatically determined by PSFex
    # (PSF_SAMPLING parameter in PSFex config file set to zero)
    psf_size = int(np.ceil(psf_size_config * psf_samp))
    # if this is odd, make it even - for the moment this is because
    # the index range of the bigger image in which this psf is put
    # ([psf_ima_center]) assumes this is even
    if psf_size % 2 != 0:
        psf_size += 1
    # now change psf_samp slightly:
    psf_samp_update = float(psf_size) / float(psf_size_config)
    # [psf_ima] is the corresponding cube of PSF subimages
    psf_ima = np.ndarray((nsubs,psf_size,psf_size), dtype='float32')
    # [psf_ima_center] is [psf_ima] broadcast into images of xsize_fft
    # x ysize_fft
    psf_ima_center = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    # [psf_ima_shift] is [psf_ima_center] shifted - this is
    # the input PSF image needed in the zogy function
    psf_ima_shift = np.ndarray((nsubs,ysize_fft,xsize_fft), dtype='float32')
    
    # loop through nsubs and construct psf at the center of each
    # subimage, using the output from PSFex that was run on the full
    # image
    for nsub in range(nsubs):
        
        x = (centers[nsub,1] - polzero1) / polscal1
        y = (centers[nsub,0] - polzero2) / polscal2

        if nsubs==1:
            psf_ima_config = data[0]
        else:
            if poldeg==2:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + \
                          data[3] * y + data[4] * x * y + data[5] * y**2
            elif poldeg==3:
                psf_ima_config = data[0] + data[1] * x + data[2] * x**2 + data[3] * x**3 + \
                          data[4] * y + data[5] * x * y + data[6] * x**2 * y + \
                          data[7] * y**2 + data[8] * x * y**2 + data[9] * y**3

        if display:
            # write this psf to fits
            fits.writeto('psf_'+imtype+'_sub'+str(nsub)+'.fits', psf_ima_config, clobber=True)
            #result = show_image(psf_ima_config)

        # resample PSF image at image pixel scale
        psf_ima_resized = ndimage.zoom(psf_ima_config, psf_samp_update)
        psf_ima[nsub] = psf_ima_resized
        if verbose and nsub==1:
            print ('psf_samp, psf_samp_update', psf_samp, psf_samp_update)
            print ('np.shape(psf_ima_config)', np.shape(psf_ima_config))
            print ('np.shape(psf_ima)', np.shape(psf_ima))
            print ('np.shape(psf_ima_resized)', np.shape(psf_ima_resized))
            print ('psf_size ', psf_size)
        if display:
            # write this psf to fits
            fits.writeto('psf_resized_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_resized, clobber=True)
            #result = show_image(psf_ima_resized)

        # normalize to unity
        psf_ima_resized_norm = psf_ima_resized / np.sum(psf_ima_resized)
            
        # now place this resized and normalized PSF image at the
        # center of an image with the same size as the fftimage
        if ysize_fft % 2 != 0 or xsize_fft % 2 != 0:
            print ('WARNING: image not even in both dimensions!')
            
        xcenter_fft, ycenter_fft = int(xsize_fft/2), int(ysize_fft/2)
        if verbose and nsub==1:
            print ('xcenter_fft, ycenter_fft ', xcenter_fft, ycenter_fft)
            psf_ima_center[nsub,
                           ycenter_fft-int(psf_size/2):ycenter_fft+int(psf_size/2),
                           xcenter_fft-int(psf_size/2):xcenter_fft+int(psf_size/2)] \
                           = psf_ima_resized_norm
            
        if display:
            fits.writeto('psf_center_'+imtype+'_sub'+str(nsub)+'.fits',
                           psf_ima_center[nsub], clobber=True)            
            #result = show_image(psf_ima_center[nsub])

        # perform fft shift
        psf_ima_shift[nsub] = fft.fftshift(psf_ima_center[nsub])
        # Eran's function:
        #print np.shape(image_shift_fft(psf_ima_center[nsub], 1., 1.))
        #psf_ima_shift[nsub] = image_shift_fft(psf_ima_center[nsub], 0., 0.)

        #result = show_image(psf_ima_shift[nsub])

    if timing: print ('wall-time spent in get_psf', time.time() - t)

    return psf_ima_shift


################################################################################

def run_psfex_adapted (cat_in, file_config, cat_out):
    
    """Function that runs PSFEx on [cat_in] (which is a SExtractor output
       catalog in FITS_LDAC format) using the configuration file
       [file_config]"""

    if timing: t = time.time()

    if psf_sampling == 0:
        # provide new PSF_SIZE based on psf_radius, which is 2 *
        # [psf_radius] * FWHM / sampling factor. The sampling factor is
        # automatically determined in PSFex, and is such that FWHM /
        # sampling factor ~ 4-5, so:
        size = int(psf_radius*9+0.5)
        # make sure it's odd
        if size % 2 == 0: size += 1
        psf_size_config = str(size)+','+str(size)
    else:
        # use some reasonable default size
        psf_size_config = '45,45'

    if verbose:
        print ('psf_size_config', psf_size_config)
        
    # run psfex from the unix command line
    cmd = ['psfex', cat_in, '-c', file_config,'-OUTCAT_NAME', cat_out,
           '-PSF_SIZE', psf_size_config, '-PSF_SAMPLING', str(psf_sampling)]
    result = subprocess.call(cmd)    

    if timing: print ('wall-time spent in run_psfex', time.time()-t)
    

################################################################################

def run_coaddition(M, P, sigma, F, nimages):

    """Function that calculates optimal co-added image (R) and its PSF
       (RP) based on Eqs. 8 and 11 of Zackay & Ofek 2017b, based on
       the inputs (shape):
        - M (nimages, y, x): cube of background-subtracted images
        - P (nimages, y, x): corresponding PSFs of the images
        - sigma (nimages): standard deviation of background
        - F (nimages): flux factors

    """

    if timing: t = time.time()

    numerator = 0.
    denominator2 = 0.
    FR2 = 0.
    
    for j in range(nimages):
        Mj_hat = fft.fft2(M[j])
        Pj_hat = fft.fft2(P[j])
        Pj_hat_conj = np.conj(Pj_hat)
        Pj_hat2_abs = np.abs(Pj_hat**2)
        sigmaj2 = sigma[j]**2
        Fj = F[j]
        Fj2 = F[j]**2

        numerator += (Fj/sigmaj2) * Pj_hat_conj * Mj_hat 
        denominator2 += (Fj2/sigmaj2) * Pj_hat2_abs
        FR2 += (Fj2/sigmaj2)

    denominator = np.sqrt(denominator2)
    R_hat = numerator / denominator
    R = np.real(fft.ifft2(R_hat))

    PR_hat = denominator / np.sqrt(FR2)
    PR = np.real(fft.ifft2(PR_hat))
    
    if timing:
        print ('wall-time spent in optimal coaddition', time.time()-t)
        #print ('peak memory used in run_ZOGY in GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9)

    return R, PR


################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='build MeerLICHT/BlackGEM '
                                     'reference images')
    parser.add_argument('--telescope', type=str, default='ML1',
                        choices=['ML1', 'BG2', 'BG3', 'BG4'],
                        help='telescope name (ML1, BG2, BG3 or BG4); '
                        'default=\'ML1\'')

    parser.add_argument('--fits_table', type=str, default=None,
                        help='name binary fits table containing header keywords '
                        'MJD-OBS, OBJECT, FILTER, QC-FLAG, RA-CNTR, DEC-CNTR, '
                        'PSF-SEE, LIMMAG and S-BKGSTD of the possible images '
                        'to be included. If not provided, or the table does not '
                        'exist (yet), a table is constructed from the available '
                        'images and saved with this name.')

    parser.add_argument('--table_only', type=str2bool, default=False,
                        help='only prepare [fits_table]?; default=False')

    parser.add_argument('--date_start', type=str, default=None,
                        help='start date (noon) to include images, date string '
                        '(e.g. yyyymmdd) or days relative to noon today (negative '
                        'number); default=None')

    parser.add_argument('--date_end', type=str, default=None,
                        help='end date (noon) to include images, date string '
                        '(e.g. yyyymmdd) or days relative to noon today (negative '
                        'number); default=None')

    parser.add_argument('--field_IDs', type=str, default=None,
                        help='only consider images with this(these) field ID(s) '
                        '(can be multiple field IDs separated by a comma, '
                        'and with the optional use of unix wildcards, '
                        'e.g. 1600[0-5],16037,161??); can also be an ascii file '
                        'with the field ID(s) in the 1st column; default=None')

    parser.add_argument('--filters', type=str, default=None,
                        help='only consider this(these) filter(s), e.g. uqi')

    parser.add_argument('--qc_flag_max', type=str, default='orange',
                        choices=['green', 'yellow', 'orange', 'red'],
                        help='worst QC flag to consider; default=\'orange\'')

    parser.add_argument('--seeing_max', type=float, default=None,
                        help='[arcsec] maximum seeing to consider; default=None')

    parser.add_argument('--make_colfig', type=str2bool, default=False,
                        help='make color figures from uqi filters?; '
                        'default=False')

    parser.add_argument('--filters_colfig', type=str, default='iqu',
                        help='set of 3 filters to use for RGB color figures; '
                        'default=\'uqi\'')

    parser.add_argument('--extension', type=str, default=None,
                        help='extension to add to default reference folder name, '
                        'e.g. _alt; default: None')

    
    
    args = parser.parse_args()

    buildref (telescope = args.telescope,
              fits_table = args.fits_table,
              table_only = args.table_only,
              date_start = args.date_start,
              date_end = args.date_end,
              field_IDs = args.field_IDs,
              filters = args.filters,
              qc_flag_max = args.qc_flag_max,
              seeing_max = args.seeing_max,
              make_colfig = args.make_colfig,
              filters_colfig = args.filters_colfig,
              extension = args.extension)


################################################################################
