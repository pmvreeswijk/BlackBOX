import os
import gc
import pickle
import copy
import tempfile
import sys
import calendar
import warnings

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
#log.propagate = False

import set_zogy
import set_blackbox as set_bb

#sys.path.append("/Software/match2SSO")
import set_match2SSO as set_m2sso
import match2SSO as m2sso

# setting environment variable OMP_NUM_THREADS to number of threads,
# (used by e.g. astroscrappy); use value from environment variable
# SLURM_CPUS_PER_TASK if it is defined, otherwise set_bb.nthreads;
# needs to be done before numpy is imported in [zogy]
os.environ['OMP_NUM_THREADS'] = str(os.environ.get('SLURM_CPUS_PER_TASK',
                                                   set_bb.nthreads))

from zogy import *

import re   # Regular expression operations
import glob # Unix style pathname pattern expansion


from datetime import datetime, timedelta
from dateutil.tz import gettz
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle, SkyCoord, FK5, ICRS, get_body
from astropy.time import Time
from astropy import units as u
from astropy.visualization import ZScaleInterval as zscale
from astropy.visualization import astropy_mpl_style

import astroscrappy
from acstools.satdet import detsat, make_mask
import shutil
#from slackclient import SlackClient as sc
import ephem
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from qc import qc_check, run_qc_check
import platform

from ASTA import ASTA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(astropy_mpl_style)
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import colormaps
from matplotlib.colors import Normalize


import fitsio
from PIL import Image


# due to regular problems with downloading default IERS file (needed
# to compute UTC-UT1 corrections for e.g. sidereal time computation),
# Steven created a mirror of this file in a google storage bucket
#
# update on 2020-10-27: urls below not working properly now; default
# server from which to download finals2000A.all seems to have changed
# to (the mirror?):
# ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all, which
# is working fine at the moment, so do not define the urls below.
from astropy.utils import iers
iers.conf.iers_auto_url = (
    'https://maia.usno.navy.mil/ser7/finals2000A.all')
iers.conf.iers_auto_url_mirror = (
    'https://storage.googleapis.com/blackbox-auxdata/timing/finals2000A.all')
    #'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')

#from astroplan import moon

# to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

#from pympler import tracker
#import tracemalloc
#tracemalloc.start()

# commands to force the downloading of above IERS bulletin file in
# case a recent one (younger than 30 days) is not present in the cache
try:
    tnow = Time.now()
    tnow.ut1
except Exception as e:
    log.warning ('exception was raised while testing \'tnow.ut1\' at start of '
                 'blackbox; issue with IERS file?: {}'.format(e))


__version__ = '1.4.6'
keywords_version = '1.2.2'


################################################################################

def run_blackbox (telescope=None, mode=None, date=None, read_path=None,
                  recursive=False, imgtypes=None, filters=None, image=None,
                  image_list=None, master_date=None,
                  img_reduce=None, cat_extract=None, trans_extract=None,
                  force_reproc_new=None, name_genlog=None, keep_tmp=None):


    """Function that processes MeerLICHT or BlackGEM images, performs
    basic image reduction tasks such as overscan subtraction,
    flat-fielding and cosmic-ray rejection, and the feeds the image to
    zogy.py along with a reference image of the corresponding ML/BG
    field ID, to detect transients present in the image."""

    global tel, filts, types, proc_mode

    # in single-image mode, infer global parameter [tel] from the
    # starting characters of the image name
    if image is not None:
        tel_tmp = image.split('/')[-1][0:3]
        if tel_tmp in ['BG2', 'BG3', 'BG4']:
            tel = tel_tmp
        elif tel_tmp[0:2] == 'ML':
            tel = 'ML1'
        else:
            tel = telescope
    else:
        tel = telescope


    # filters and imgtypes
    filts = filters
    types = imgtypes
    if imgtypes is not None:
        types = imgtypes.lower()

    proc_mode = mode

    # define number of processes or tasks [nproc]; when running on the
    # ilifu/google slurm cluster the environment variable SLURM_NTASKS
    # should be set through --ntasks-per-node in the sbatch script;
    # otherwise use the value from the set_bb settings file
    nproc = int(os.environ.get('SLURM_NTASKS', get_par(set_bb.nproc,tel)))

    # update nthreads in set_bb with value of environment variable
    # OMP_NUM_THREADS set at the top of this module
    set_bb.nthreads = int(os.environ.get('OMP_NUM_THREADS', set_bb.nthreads))


    # update various parameters in set_bb if corresponding input
    # parameters are not None
    if img_reduce is not None:
        set_bb.img_reduce = str2bool(img_reduce)

    if cat_extract is not None:
        set_bb.cat_extract = str2bool(cat_extract)

    if trans_extract is not None:
        set_bb.trans_extract = str2bool(trans_extract)

    if force_reproc_new is not None:
        set_bb.force_reproc_new = str2bool(force_reproc_new)

    if keep_tmp is not None:
        set_bb.keep_tmp = str2bool(keep_tmp)


    # in night mode, force create_master to be True; N.B.: using
    # blackbox_slurm at ilifu and when running in the google cloud,
    # blackbox will be run in 'day' mode on single images
    if mode == 'night':
        set_bb.create_master = True


    if get_par(set_zogy.timing,tel):
        t_run_blackbox = time.time()


    # attach general logfile to logging
    ###################################

    # dealing with google cloud bucket?
    google_cloud = (get_par(set_bb.raw_dir,tel)[0:5] == 'gs://')


    # in google_cloud mode, do not keep general logfile; N.B.: this
    # general logfile is now created in separate scripts that submit
    # the jobs to Slurm
    if not google_cloud:

        if not isdir(get_par(set_bb.log_dir,tel)):
            os.makedirs(get_par(set_bb.log_dir,tel))

        global genlogfile
        genlogfile = None

        if name_genlog is not None:
            # check if path is provided
            fdir, fname = os.path.split(name_genlog)
            if len(fdir)>0 and isdir(fdir):
                log_dir = fdir
            else:
                log_dir = get_par(set_bb.log_dir,tel)

            genlogfile = '{}/{}'.format(log_dir, fname)


        elif mode == 'night':

            # in night mode, create general logfile based on date/time
            genlogfile = '{}/{}_{}.log'.format(
                get_par(set_bb.log_dir,tel), tel,
                Time.now().strftime('%Y%m%d_%H%M%S'))


        if genlogfile is not None:

            fileHandler = logging.FileHandler(genlogfile, 'a')
            fileHandler.setFormatter(logFormatter)
            fileHandler.setLevel('INFO')
            log.addHandler(fileHandler)
            log.info ('genlogfile created:      {}'.format(genlogfile))



    log.info ('processing mode:         {}'.format(mode))
    log.info ('test/staging/production: {}'.format(get_par(set_bb.proc_env,tel)))
    log.info ('number of processes:     {}'.format(nproc))
    log.info ('number of threads:       {}'.format(set_bb.nthreads))
    log.info ('switch img_reduce:       {}'.format(get_par(set_bb.img_reduce,tel)))
    log.info ('switch cat_extract:      {}'.format(get_par(set_bb.cat_extract,tel)))
    log.info ('switch trans_extract:    {}'.format(get_par(set_bb.trans_extract,tel)))
    log.info ('force reprocessing new:  {}'.format(get_par(set_bb.force_reproc_new,tel)))
    log.info ('keep temporary folders:  {}'.format(get_par(set_bb.keep_tmp,tel)))



    mem_use (label='in run_blackbox at start')


    # create master bias, dark and/or flat if [master_date] is specified
    if master_date is not None:
        create_masters (master_date, nproc=nproc)
        logging.shutdown()
        return


    # leave right away if none of the main processing switches are on
    if (not get_par(set_bb.img_reduce,tel) and
        not get_par(set_bb.cat_extract,tel) and
        not get_par(set_bb.trans_extract,tel)):

        log.info ('main processing switches img_reduce, cat_extract '
                  'and trans_extract all False, nothing left to do')
        logging.shutdown()
        return


    # [read_path] is assumed to be the full path to the directory with
    # raw images to be processed; if not provided as input parameter,
    # it is defined using the input [date] with the function
    # [get_path]
    if read_path is None:
        if date is not None:
            read_path, __ = get_path(date, 'read')
            log.info ('processing files from directory: {}'.format(read_path))
        elif image is not None:
            pass
        elif image_list is not None:
            pass
        else:
            # if [read_path], [date], [image] and [image_list] are all None, exit
            log.critical ('[read_path], [date], [image], [image_list] all None')
            logging.shutdown()
            return

    else:
        # if it is provided but does not exist, exit unless in night
        # mode in which case it will be created below
        if not isdir(read_path) and mode != 'night':
            log.critical ('[read_path] directory provided does not exist:\n{}'
                          .format(read_path))
            logging.shutdown()
            return

        else:
            # infer date from readpath: [some path]/yyyy/mm/dd in case
            # input read_path is defined but input date is not
            date = read_path.split('/')[-3:]


    # create global lock instance that can be used in
    # [blackbox_reduce] for certain blocks/functions to be accessed by
    # one process at a time
    global lock
    # use function from zogy to use same mp start method
    lock = get_mp_Lock()


    # for both day and night mode, create list of all
    # files present in [read_path], in image type order:
    # bias, dark, flat, object and other
    if image is None and image_list is None:
        biases, darks, flats, objects, others = sort_files(read_path, 'fits',
                                                           recursive=recursive)
        lists = [biases, darks, flats, objects, others]
        filenames = [name for sublist in lists for name in sublist]
    else:
        if mode == 'night':
            log.critical ('[image] or [image_list] should not be defined '
                          'in night mode')
            logging.shutdown()
            return

        elif image is not None:
            # if input parameter [image] is defined, the filenames
            # to process will contain a single image
            filenames = [image]
        elif image_list is not None:
            # if input parameter [image_list] is defined,
            # read the ascii files into filenames list
            with open(image_list, 'r') as f:
                filenames = [name.strip() for name in f if name[0]!='#']


    # split into 'day' or 'night' mode
    filename_reduced = None
    if mode == 'day':

        if len(filenames)==0:
            log.warning ('no files to reduce')


        # see https://docs.python.org/3/library/tracemalloc.html
        #snapshot1 = tracemalloc.take_snapshot()

        if nproc==1 or image is not None:

            # if only 1 process is requested, or [image] input
            # parameter is not None, run it without multiprocessing;
            # this will allow images to be shown on the fly if
            # [set_zogy.display] is set to True; something that is not
            # allowed (at least not on a macbook) when
            # multiprocessing.
            log.warning ('running single process')
            filenames_reduced = []
            for filename in filenames:
                filenames_reduced.append(try_blackbox_reduce(filename))

        else:
            # use [pool_func] to process list of files
            filenames_reduced = pool_func (try_blackbox_reduce, filenames,
                                           nproc=nproc)


        log.info ('{} filenames reduced: {}'.format(len(filenames_reduced),
                                                    filenames_reduced))

        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
        #    print(stat)


    elif mode == 'night':

        # if in night mode, check if anything changes in input directory
        # and if there is a new file, feed it to [blackbox_reduce]

        # [read_path] folder may not exist yet (e.g. no data have yet
        # been synced to it), which will cause watchdog to break, so
        # make sure it exists
        make_dir (read_path)


        # determine time of next sunrise
        obs = ephem.Observer()
        obs.lat = str(get_par(set_zogy.obs_lat,tel))
        obs.lon = str(get_par(set_zogy.obs_lon,tel))
        height = get_par(set_zogy.obs_height,tel)
        obs.elevation = height
        # correct apparent horizon for observer elevation, which is
        # not taken into account in ephem
        obs = adjust_horizon(obs, height)
        sunrise = obs.next_rising(ephem.Sun())


        # create queue for submitting jobs; use function from zogy to
        # use same mp start method
        queue = get_mp_Queue()


        # add files that are already present in the read_path
        # directory to the night queue, to reduce these first
        for filename in filenames:
            queue.put(filename)


        # create and setup observer, but do not start just yet
        observer = PollingObserver()
        observer.schedule(FileWatcher(queue), read_path, recursive=recursive)


        # create pool of workers
        results = []
        # use function from zogy to use same mp start method
        pool = get_mp_Pool(nproc)


        # start monitoring [read_path] for incoming files
        observer.start()


        # keep monitoring queue - which is being filled with new files
        # detected by watchdog - as long as it is nighttime or the
        # queue is not empty yet
        while ephem.now()-sunrise < ephem.hour or not queue.empty():

            if queue.empty():
                time.sleep(60)
            else:
                filename = get_file (queue)
                if filename is not None:
                    # process it by one of the workers
                    results.append(pool.apply_async(try_blackbox_reduce,
                                                    [filename]))


        log.info ('night has finished and queue is empty')


        # watchdog can be stopped
        observer.stop() #stop observer
        observer.join() #join observer


        # closing and joining pool of workers
        pool.close()
        pool.join()


        # create and email obslog
        log.info ('night processing has finished; creating and emailing obslog')
        try:
            create_obslog (date, email=True, tel=tel, weather_screenshot=True)
        except Exception as e:
            log.exception ('exception was raised in creating obslog: {}'
                           .format(e))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t_run_blackbox, label='in run_blackbox at very end')


    logging.shutdown()
    return


################################################################################

def adjust_horizon (observer, height):

    # 34arcmin due to atmospheric refraction (ephem uses top of Sun by
    # default, so no need for 16arcmin due to Sun apparent radius)
    horizon = -34/60

    # Earth's radius in m
    R = (1*u.earthRad).to(u.m).value
    horizon -= np.degrees(np.arccos((R/(R+height))))

    # set pressure to zero to discard ephem's internal refraction
    # calculation
    observer.pressure = 0
    observer.horizon = str(horizon)

    return observer


################################################################################

def get_file (queue):

    """Get file from queue after making sure it arrived completely; None
    is returned if the file is not a fits file or still having trouble
    reading the fits file even after waiting for 60s; otherwise the
    filename is returned.

    """

    # get event from queue
    event = queue.get(True)

    try:
        # get name of new file
        filename = str(event.src_path)
        filetype = 'new'

    except AttributeError as e:
        # instead of event, queue entry is a filename added in
        # [run_blackbox]
        filename = event
        filetype = 'pre-existing'


    log.info ('detected a {} file: {}'.format(filetype, filename))


    # only continue if a fits file
    if 'fits' not in filename:

        log.info ('{} is not a fits file; skipping it'.format(filename))
        filename = None

    else:

        # if filename is a temporary rsync copy (default
        # behaviour of rsync is to create a temporary file
        # starting with .[filename].[randomstr]; can be
        # changed with option "--inplace"), then let filename
        # refer to the eventual file created by rsync
        fn_head, fn_tail = os.path.split(filename)
        if fn_tail[0] == '.':
            filename = '{}/{}'.format(fn_head, '.'
                                      .join(fn_tail.split('.')[1:-1]))
            log.info ('changed filename from rsync temporary file {} to {}'
                      .format(event.src_path, filename))

        # this while loop below replaces the old [copying]
        # function; it times out after wait_max is reached
        wait_max = 180
        t0 = time.time()
        nsleep = 0
        while time.time()-t0 < wait_max:

            try:
                # read the file
                data = read_hdulist(filename)

            except:

                process = False
                if nsleep==0:
                    log.warning ('file {} has not completely arrived yet; '
                                 'will keep trying to read it in for {}s'
                                 .format(filename, wait_max))

                # give file a bit of time to arrive before next read attempt
                time.sleep(5)
                nsleep += 1

            else:
                # if fits file was read fine, set process flag to True
                process = True
                log.info ('successfully read file {} within {:.1f}s'
                          .format(filename, time.time()-t0))
                # and break out of while loop
                break


        if not process:
            log.info ('{}s limit for reading file reached, not processing {}'
                      .format(wait_max, filename))
            filename = None


    return filename


################################################################################

class FileWatcher(FileSystemEventHandler, object):
    '''Monitors directory for new files.

    :param queue: multiprocessing queue for new files
    :type queue: multiprocessing.Queue'''

    def __init__(self, queue):
        self._queue = queue

    def on_created(self, event):
        '''Action to take for new files.

        :param event: new event found
        :type event: event'''
        self._queue.put(event)


################################################################################

def create_masters (master_date, run_fpack=True, nproc=1):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    log.info ('creating master frames')

    # prepare list of all red/yyyy/mm/dd/bias and flat directories
    red_dir = get_par(set_bb.red_dir,tel)
    master_dir = get_par(set_bb.master_dir,tel)


    # [master_date] could be an existing text file, with evening dates
    # in the first column and optionally the filter in the second
    # column (only relevant for flats; if not provided, all filters
    # are considered); such a file can be used to create masters for a
    # list of specific dates
    list_filt = None
    if master_date is not None and isfile(master_date):

        # read ascii table
        table = Table.read(master_date, format='ascii', data_start=0)
        # table can contain 1 or 2 columns and can therefore not
        # pre-define column names, while with data_start=0 the entries
        # on the first line are taken as the column names
        cols = table.colnames

        # lists with evening dates and path including the yyyy/mm/dd
        # folders, but not further
        list_date_eve = []
        list_path = []
        for i in range(len(table)):
            # take out potential characters in table date column
            date_tmp = ''.join(e for e in str(table[cols[0]][i]) if e.isdigit())
            list_date_eve.append(date_tmp)
            list_path.append('{}/{}/{}/{}'.format(red_dir, date_tmp[0:4],
                                                  date_tmp[4:6], date_tmp[6:8]))

        # define list of filters if 2nd column is defined
        if len(cols)>1:
            list_filt = list(table[cols[1]])


    elif master_date is not None:

        # prepare master at [master_date]
        date_tmp = master_date
        list_date_eve = [master_date]
        list_path = ['{}/{}/{}/{}'.format(red_dir, date_tmp[0:4],
                                          date_tmp[4:6], date_tmp[6:8])]

    else:

        # if master_date is not specified, loop all
        # available [imgtype] folders in the reduced path
        year, month, day = None, None, None

        # master_date can be specified as yyyy, yyyymm or yyyymmdd (possibly
        # with any connecting character like '/' or '-'), so all
        # masters for specific day, month or year can be created
        if master_date is not None and master_date != '*':
            master_date = ''.join(e for e in master_date if e.isdigit())
            # only loop folders with specific year, month and/or day
            if len(master_date) >= 4:
                year = master_date[0:4]
                if len(master_date) >= 6:
                    month = master_date[4:6]
                    if len(master_date) >= 8:
                        day = master_date[6:8]


        # define search path
        search_path = '{}'.format(red_dir)
        if year is not None:
            search_path = '{}/{}'.format(search_path, year)
            if month is not None:
                search_path = '{}/{}'.format(search_path, month)
                if day is not None:
                    search_path = '{}/{}'.format(search_path, day)



        # infer folder depth to use
        if 'gs://' in red_dir:
            bucket_name, __ = get_bucket_name (red_dir)
            folder_tmp = red_dir.split(bucket_name)[-1]
            depth = folder_tmp.rstrip('/').count('/')
        else:
            depth = red_dir.rstrip('/').count('/')

        # add 3 to the depth for the yyyy/mm/dd folders
        depth += 3

        # use function [list_folders] to list all existing folders of
        # depth [depth]
        list_path = list_folders(search_path, depth=depth)

        # corresponding list of evening dates
        list_date_eve = [''.join(l.split('/')[-3:]) for l in list_path]



    # filts is a global variable determined by the [filters] input
    # to [run_blackbox]
    if filts is None:
        # if None set to all filters
        filts_temp = get_par(set_zogy.zp_default,tel).keys()
    else:
        # extract filters from [filts]
        filts_temp = re.sub(',|-|\.|\/', '', filts)


    # create list of master flats to process
    list_masters = []
    nfiles = len(list_path)
    for i in range(nfiles):
        # biases
        if types is None or 'bias' in types:
            list_masters.append('{}/bias/{}_bias_{}.fits'
                                .format(list_path[i], tel, list_date_eve[i]))

        # darks
        if types is None or 'dark' in types:
            list_masters.append('{}/dark/{}_dark_{}.fits'
                                .format(list_path[i], tel, list_date_eve[i]))

        # flats
        if types is None or 'flat' in types:
            # if input master_date is a file and 2nd column is defined, use
            # it for the filter(s)
            if list_filt is not None:
                filts_2loop = [f for f in list_filt[i] if f in filts_temp]
            else:
                filts_2loop = filts_temp

            # loop filters and create list of masters to multiprocess
            for filt in filts_2loop:
                list_masters.append('{}/flat/{}_flat_{}_{}.fits'
                                    .format(list_path[i], tel,
                                            list_date_eve[i], filt))


    # for google cloud, the masters have their separate bucket
    if master_dir[0:5] == 'gs://':
        list_masters = [l.replace(red_dir, master_dir) for l in list_masters]


    log.info ('list_masters: {}'.format(list_masters))


    # data shape is needed as input for [master_prep]
    data_shape = (get_par(set_bb.ny,tel) * get_par(set_bb.ysize_chan,tel),
                  get_par(set_bb.nx,tel) * get_par(set_bb.xsize_chan,tel))


    # use [pool_func] to process list of masters; pick_alt is set to
    # False as there is no need to look for an alternative master flat
    list_fits_master = pool_func (master_prep, list_masters, data_shape, True,
                                  False, tel, proc_mode, nproc=nproc)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in create_masters')


    return


################################################################################

def already_exists (filename, get_filename=False):

    file_list = [filename, '{}.fz'.format(filename), '{}.gz'.format(filename),
                 filename.replace('.fz',''), filename.replace('.gz','')]

    # unique entries:
    file_list = list(set(file_list))

    exists = False
    existing_file = filename

    for file_temp in file_list:
        if isfile(file_temp):
            exists = True
            existing_file = file_temp
            break

    if get_filename:
        return exists, existing_file
    else:
        return exists


################################################################################

def fpack (filename):

    """fpack fits images; skip fits tables"""

    try:

        # fits check if extension is .fits and not an LDAC fits file
        if filename.split('.')[-1] == 'fits' and '_ldac.fits' not in filename:
            header = read_hdulist(filename, get_data=False, get_header=True,
                                  ext_name_indices=0)

            # check if it is an image
            if int(header['NAXIS'])==2:
                # determine if integer or float image
                if int(header['BITPIX']) > 0:
                    cmd = ['fpack', '-D', '-Y', '-v', filename]
                else:
                    if 'Scorr' in filename or 'limmag' in filename:
                        quant = 2
                    elif 'Fpsf' in filename:
                        quant = 4
                    else:
                        quant = 16

                    cmd = ['fpack', '-q', str(quant), '-D', '-Y', '-v', filename]


                # if output fpacked file already exists, delete it
                filename_packed = '{}.fz'.format(filename)
                if isfile(filename_packed):
                    #os.remove(filename_packed)
                    remove_files([filename_packed])
                    log.warning ('fpacking over already existing file {}'
                                 .format(filename_packed))

                subprocess.run(cmd)
                filename = filename_packed


    except Exception as e:
        #log.exception (traceback.format_exc())
        log.exception ('exception was raised in fpacking of image {}: {}'
                       .format(filename,e))


    return filename


################################################################################

def create_jpg (filename, cmap='gray', ext='jpg'):

    """Create jpg image from fits"""

    mem_use (label='at start of create_jpg')

    try:

        image_jpg = '{}.{}'.format(filename.split('.fits')[0], ext)

        if not isfile(image_jpg):

            log.info ('saving {} to {}'.format(filename, image_jpg))

            # read input image
            data, header = read_hdulist(filename, get_header=True,
                                        dtype='float32')


            imgtype = header['IMAGETYP'].lower()
            file_str = image_jpg.split('/')[-1].split('.jpg')[0]
            title = 'file:{}   '.format(file_str)

            title_dict = {'object': ['OBJECT', 'FILTER', 'EXPTIME', 'QC-FLAG'],
                        'bias': ['IMAGETYP', 'FILTER', 'QC-FLAG'],
                        'dark': ['IMAGETYP', 'FILTER', 'QC-FLAG'],
                        'flat': ['IMAGETYP', 'FILTER', 'EXPTIME', 'QC-FLAG']}

            for key in title_dict[imgtype]:
                if key in header:
                    if key == 'EXPTIME':
                        title += '{}:{:.1f}s   '.format(key.lower(), header[key])
                    else:
                        title += '{}:{}   '.format(key.lower(), header[key])


            # label next to colorbar
            if 'Scorr' in filename:
                label = 'significance (sigma)'
            else:
                label = 'pixel value (e-)'


            fig, ax = plt.subplots(layout='constrained', figsize=(8.27,8.27))
            vmin, vmax = zscale(contrast=0.35).get_limits(data)
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            mapped_data = colormaps.get_cmap(cmap)(norm(data), bytes=True)
            plt.imshow(mapped_data, cmap=cmap, origin='lower')
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                         ax=ax, label=label, orientation='vertical',
                         fraction=0.05, pad=0.05)
            plt.title(title, fontsize=10)
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            plt.grid(None)
            plt.savefig(image_jpg, dpi=175)
            plt.close()

            mem_use (label='at end of create_jpg')

    except Exception as e:
        #log.exception (traceback.format_exc())
        log.exception ('exception was raised in creating jpg of image {}: {}'
                       .format(filename,e))


    return image_jpg


################################################################################

class WrapException(Exception):
    """see https://bugs.python.org/issue13831Ups"""

    def __init__(self):
        exc_type, exc_value, exc_tb = sys.exc_info()
        self.exception = exc_value
        self.formatted = ''.join(traceback.format_exception(exc_type, exc_value,
                                                            exc_tb))
    def __str__(self):
        return '{}\nOriginal traceback:\n{}'.format(Exception.__str__(self),
                                                    self.formatted)


################################################################################

def try_blackbox_reduce (filename):

    """This is a wrapper function to call [blackbox_reduce] below in a
    try-except statement in order to enable to show the complete
    exception traceback using [WrapException] above; this was not
    working before when using multiprocessing (nproc>1).  This
    construction should not be needed anymore when moving to python
    3.4+ as the complete traceback should be provided through the
    .get() method in [pool_func].

    """

    try:
        filename_reduced = blackbox_reduce (filename)

    except:
        filename_reduced = None
        raise WrapException()

    finally:

        # in case of exception in blackbox_reduce, make sure to close
        # the log and remove the tmp folder
        if filename_reduced is None:

            try:

                fn_red = get_filename_red (filename)
                tmp_path = '{}/{}'.format(get_par(set_bb.tmp_dir,tel), fn_red)

                # close the log
                logfile = '{}/{}.log'.format(tmp_path, fn_red)
                close_log(log, logfile)

                # running in the google cloud?
                #google_cloud = (filename[0:5]=='gs://')

                # remove tmp folder if not keeping tmp files; not so
                # urgent for MeerLICHT and provides opportunity to inspect
                # the tmp folder to find out what went wrong exactly
                # if google_cloud:
                clean_tmp (tmp_path, get_par(set_bb.keep_tmp,tel))

            except Exception as e:

                log.warning ('following exception occurred in '
                             'blackbox.try_blackbox_reduce when attempting to '
                             'close the log and remove the tmp folder for {}: {}'
                             .format(filename, e))


    return filename_reduced


################################################################################

def get_filename_red (fits_raw):

    """determine reduced filename from raw fits header"""

    # read header
    header = read_hdulist(fits_raw, get_data=False, get_header=True)

    # use [set_header] to update raw header so that also DATE-OBS
    # is updated to be mid-exposure time
    header = set_header(header, fits_raw, silent=True)

    # UT date (yyyymmdd) and time (hhmmss)
    utdate, uttime = get_date_time(header)

    # reduced filename without the full path nor fits extension
    filename_red = '{}_{}_{}_red'.format(tel, utdate, uttime)


    return filename_red


################################################################################

def blackbox_reduce (filename):

    """Function that takes as input a single raw fits image and works
       through chain of reduction steps, from correcting for the gain
       and overscan to running ZOGY on the reduced image.

    """

    if get_par(set_zogy.timing,tel):
        t_blackbox_reduce = time.time()
        mem_use (label='in blackbox_reduce at start')


    # running in the google cloud?
    google_cloud = (filename[0:5]=='gs://')


    # just read the header for the moment
    try:
        header = read_hdulist(filename, get_data=False, get_header=True)
    except Exception as e:
        #log.exception (traceback.format_exc())
        log.exception ('exception was raised in read_hdulist at top of '
                       '[blackbox_reduce]: {}; not processing {}'
                       .format(e, filename))
        return None


    # first header check using function [check_header1]
    header_ok = check_header1 (header, filename)
    if not header_ok:
        return None


    # determine the raw data path
    raw_path, __ = get_path(header['DATE-OBS'], 'read')


    # check quality control
    qc_flag = run_qc_check (header, tel)
    if qc_flag=='red':
        log.error ('red QC flag in image {}; returning without making '
                   'dummy catalogs'.format(filename))
        return None


    # if 'IMAGETYP' keyword not one of those specified in input parameter
    # [imgtypes] or complete set: ['bias', 'dark', 'flat', 'object']
    if types is not None:
        imgtypes2process = types
    else:
        imgtypes2process = ['bias', 'dark', 'flat', 'object']
    # then also return
    imgtype = header['IMAGETYP'].lower()
    if imgtype not in imgtypes2process:
        log.warning ('image type ({}) not in [imgtypes] ({}); not processing '
                     '{}'.format(imgtype, imgtypes2process, filename))
        return None


    # extend the header with some useful/required keywords
    try:
        header = set_header(header, filename)
    except Exception as e:
        #log.exception (traceback.format_exc())
        log.exception ('exception was raised during [set_header] of image {}: '
                       '{}; returning without making dummy catalogs'
                       .format(filename, e))
        return None


    # 2nd header check following [set_header] using function [check_header2]
    header_ok = check_header2 (header, filename)
    if not header_ok:
        return None


    # add additional header keywords
    header['PYTHON-V'] = (platform.python_version(), 'Python version used')
    header['BB-V'] = (__version__, 'BlackBOX version used')
    header['KW-V'] = (keywords_version, 'header keywords version used')
    header['BB-START'] = (Time.now().isot, 'start UTC date of BlackBOX image run')


    # defining various paths and output file names
    ##############################################

    # define [write_path] using the header DATE-OBS
    write_path, date_eve = get_path(header['DATE-OBS'], 'write')
    make_dir (write_path)
    bias_path = '{}/bias'.format(write_path)
    dark_path = '{}/dark'.format(write_path)
    flat_path = '{}/flat'.format(write_path)

    # master path is different for ilifu/Slurm and google cloud
    red_dir = get_par(set_bb.red_dir,tel)
    master_dir = get_par(set_bb.master_dir,tel)
    master_path = write_path.replace(red_dir, master_dir)

    # similar for the thumbnails
    thumbnails_dir = get_par(set_bb.thumbnails_dir,tel)
    thumbnails_path = write_path.replace(red_dir, thumbnails_dir)

   # UT date (yyyymmdd) and time (hhmmss)
    utdate, uttime = get_date_time(header)

    # define paths of different image types
    path = {'bias': bias_path,
            'dark': dark_path,
            'flat': flat_path,
            'object': write_path}
    filt = header['FILTER']

    # if exptime is not in the header or if it's 0 for a science
    # image, skip image
    if 'EXPTIME' in header:
        exptime = float(header['EXPTIME'])
        if ('IMAGETYP' in header and header['IMAGETYP'].lower()=='object'
            and int(exptime)==0):
            log.error ('science image {} with EXPTIME of zero; skipping image'
                       .format(filename))
            return None
    else:
        log.warning ('keyword EXPTIME not in header of {}; skipping image'
                     .format(filename))
        return None


    # if [only_filt] is specified, skip image if not relevant
    if filts is not None:
        if filt not in filts and imgtype != 'bias' and imgtype != 'dark':
            log.warning ('image filter ({}) not in [only_filters] ({}); '
                         'not processing {}'.format(filt, filts, filename))
            return None

    fits_out = '{}/{}_{}_{}.fits'.format(path[imgtype], tel, utdate, uttime)

    if imgtype == 'bias':
        make_dir (bias_path)

    elif imgtype == 'dark':
        make_dir (dark_path)

    elif imgtype == 'flat':
        make_dir (flat_path)
        fits_out = fits_out.replace('.fits', '_{}.fits'.format(filt))

    elif imgtype == 'object':

        # OBJECT is ML/BG field number padded with zeros (checked in
        # set_header)
        obj = header['OBJECT']

        fits_out = fits_out.replace('.fits', '_red.fits')
        fits_out_mask = fits_out.replace('_red.fits', '_mask.fits')

        # and reference image
        ref_path = '{}/{}'.format(get_par(set_bb.ref_dir,tel), obj)

        # previously:
        #ref_fits_in = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)

        # can use [list_files] to find [ref_fits_in] for both
        # ilifu/Slurm and google cloud, despite different names
        # (object or field ID was included in ref_path above):
        # [ref_path]/ML1_[filt]_red.fits.fz - ilifu/Slurm
        # and
        # [ref_path]/ML1_[obj]_[filt]_[date]_red.fits.fz - google
        # [ref_path]/BG_[obj]_[filt]_[date]_red.fits.fz - google

        # do not use telescope name here, to allow ML references to be
        # used for BG
        list_ref = list_files(ref_path, search_str='_{}_'.format(filt),
                              end_str='_red.fits.fz')


        # check if there is a reference image at all
        if len(list_ref)==0:

            ref_present = False
            ref_fits_in = None

        else:

            # there are three possibilities for reference images:
            # (1) ML or BG reference image is telescope-specific, so
            #     built only from images taken with a single telescope
            #     and its name starts with [tel]_; this is always the
            #     case for ML1 and for BG this needs to be set with
            #     input parameter mixBGs=False in buildref
            # (2) for BG: reference image starting with BG_ is
            #     typically built from a mix of BG telescopes
            #     (although it could also be from one single telescope
            #     if only those images were available or useful); this
            #     mixing of BGs is the default option in buildref for BG
            # (3) for BG: if no BG reference image available, use the
            #     one from ML

            # mask to check if any image in [list_ref] contains [tel]
            mask_tel = np.array([tel in it.split('/')[-1] for it in list_ref])

            # mask to check if any image in [list_ref] contains BG_
            mask_BG = np.array(['BG_' in it.split('/')[-1] for it in list_ref])

            if np.sum(mask_tel) > 0:
                # use option (1)
                list_ref = list(np.array(list_ref)[mask_tel])
            elif np.sum(mask_BG) > 0:
                # use option (2)
                list_ref = list(np.array(list_ref)[mask_BG])


            # sort the ones left in time and use the most recent one
            # (there should be only 1 left, but just in case)

            # !!!CHECK!!! N.B.: for ML: if the old ref image with name
            # ML1_[filt]_red.fits.fz is still present, that will be
            # picked if it is mixed with ref images with the new name
            # ML1_[obj]_[filt]_[data]_red.fits.fz
            ref_present = True
            ref_fits_in = sorted(list_ref)[-1].replace('.fz','')


            # check if the image being processed is not used as the
            # only image for the current referent image; if that is
            # the case, leave the function
            header_ref = read_hdulist(ref_fits_in, get_data=False,
                                      get_header=True)
            if header_ref['R-NUSED']==1 and header_ref['R-IM1'] in fits_out:
                log.warning ('this image {} is the current reference image '
                             'of field {} in filter {}; not processing it'
                             .format(fits_out.split('/')[-1], obj, filt))
                return None



    # prepare directory to store temporary files, which is set to the
    # tmp directory defined by [set_bb.tmp_dir] with subdirectory the
    # name of the reduced image without the .fits extension.
    tmp_path = '{}/{}'.format(get_par(set_bb.tmp_dir,tel),
                              fits_out.split('/')[-1].replace('.fits',''))


    # if running in the google cloud and not keep tmp files, clean up
    # tmp base folder, in case another job - which could be from
    # another telescope - left some files behind
    if google_cloud and not get_par(set_bb.keep_tmp,tel):

        # this is not permitted:
        #shutil.rmtree(get_par(set_bb.tmp_dir_base,tel))

        # instead, go through separate entries (files or
        # directories) in that folder and remove them one by one
        with os.scandir(get_par(set_bb.tmp_dir_base,tel)) as it:
            for entry in it:
                # only consider folders that start with 'BG';
                # otherwise running into trouble trying to
                # remove 'Constant.pm' or '.XIM-unix'
                if entry.is_dir() and entry.name.startswith('BG'):
                    log.info ('removing folder {}'.format(entry.path))
                    shutil.rmtree(entry.path, ignore_errors=True)



    # make tmp folder
    make_dir (tmp_path, empty=True)


    # prepare the logfile in [tmp_path]
    logfile = '{}/{}'.format(tmp_path, fits_out.split('/')[-1]
                             .replace('.fits','.log'))
    # attach logfile to logging
    fileHandler = logging.FileHandler(logfile, 'a')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel('INFO')
    log.addHandler(fileHandler)


    if imgtype == 'object':

        # output images and catalogs to refer to [tmp] directory
        new_fits = '{}/{}'.format(tmp_path, fits_out.split('/')[-1])
        new_fits_mask = new_fits.replace('_red.fits', '_mask.fits')
        fits_tmp_cat = new_fits.replace('.fits', '_cat.fits')
        fits_tmp_trans = new_fits.replace('.fits', '_trans.fits')
        # these are for copying files
        tmp_base = new_fits.split('_red.fits')[0]
        new_base = fits_out.split('_red.fits')[0]



    # check if reduction steps could be skipped
    file_present, fits_out_present = already_exists (fits_out, get_filename=True)
    if imgtype == 'object':
        mask_present = already_exists (fits_out_mask)
    else:
        # for non-object images, there is no mask
        mask_present = True

    # if reduced file and its possible mask exist, and img_reduce
    # and force_reproc_new flags are not both set to True, reduction
    # can be skipped
    if (file_present and mask_present and
        not (get_par(set_bb.img_reduce,tel) and
             get_par(set_bb.force_reproc_new,tel))):

        log.warning ('corresponding reduced {} image {} already exists; '
                     'skipping its reduction'
                     .format(imgtype, fits_out_present.split('/')[-1]))

        # copy relevant files to tmp folder for object images
        if imgtype == 'object':

            # indicate reduction is being skipped
            do_reduction = False

            log.info ('copying existing products to tmp folder')
            # copy files to tmp folder, unless both cat_extract and
            # trans_extract are turned off
            if (get_par(set_bb.cat_extract,tel) or
                get_par(set_bb.trans_extract,tel)):
                copy_files2keep(new_base, tmp_base,
                                get_par(set_bb.img_reduce_exts,tel),
                                move=False, run_fpack=False)

                # since an existing logfile was just copied to the tmp
                # folder, need to attach it to logging (again)
                fileHandler = logging.FileHandler(logfile, 'a')
                fileHandler.setFormatter(logFormatter)
                fileHandler.setLevel('INFO')
                log.addHandler(fileHandler)


        else:
            # for non-object images, leave function; if reduction steps would
            # not have been skipped, this would have happened before
            close_log(log, logfile)
            return fits_out

    else:

        # go through various reduction steps
        do_reduction = True

        if file_present:

            log.info ('forced reprocessing: removing all existing products '
                      'in reduced folder for {}'.format(filename))

            # this is a forced re-reduction; delete all corresponding
            # files in reduced folder as they will become obsolete
            # with this re-reduction
            if imgtype == 'object':
                files_2remove = list_files(new_base)
            else:
                # for biases and flats, just remove the reduced file
                # itself, its log and jpg
                file_jpg = '{}.jpg'.format(fits_out_present.split('.fits')[0])
                # logfile is defined to be in the tmp folder, so need
                # to define the output logfile here
                file_log = '{}.log'.format(fits_out_present.split('.fits')[0])
                files_2remove = [fits_out_present, file_log, file_jpg]


            # remove list of files
            remove_files(files_2remove, verbose=True)



        # write some info to the log
        if file_present:
            log.info('forced re-processing of {}'.format(filename))
        else:
            log.info('processing {}'.format(filename))

        log.info ('output file:       {}'.format(fits_out))
        log.info ('image type:        {}'.format(imgtype))
        log.info ('filter:            {}'.format(filt))
        log.info ('exptime:           {:.1f}s'.format(exptime))
        if imgtype == 'object':
            log.info ('OBJECT (field ID): {}'.format(obj))

        log.info ('write_path:        {}'.format(write_path))
        log.info ('bias_path:         {}'.format(bias_path))
        log.info ('dark_path:         {}'.format(dark_path))
        log.info ('flat_path:         {}'.format(flat_path))
        log.info ('master_path:       {}'.format(master_path))
        if imgtype == 'object':
            log.info ('tmp_path:          {}'.format(tmp_path))
            log.info ('ref_path:          {}'.format(ref_path))
            log.info ('thumbnails_path:   {}'.format(thumbnails_path))


        # image log file
        header['LOG-IMA'] = (logfile.split('/')[-1], 'name image logfile')

        # now also read in the raw image data
        try:
            data = read_hdulist(filename, dtype='float32')
        except:
            log.exception('problem reading image {}; leaving function '
                          'blackbox_reduce'.format(filename))
            close_log(log, logfile)
            clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
            return None


        # determine number of pixels with infinite/nan values
        mask_infnan = ~np.isfinite(data)
        n_infnan = np.sum(mask_infnan)
        header['N-INFNAN'] = (n_infnan, 'number of pixels with infinite/nan '
                              'values')
        if n_infnan > 0:
            log.warning('{} pixels with infinite/nan values; replacing '
                        'with zero'.format(n_infnan))
            data[mask_infnan] = 0


        #snapshot1 = tracemalloc.take_snapshot()


        # gain correction
        #################
        try:
            log.info('correcting for the gain')
            gain_processed = False
            gain_corr(data, header, tel=tel)
        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [gain_corr] of image {}: '
                          '{}'.format(filename, e))
        else:
            gain_processed = True
        finally:
            header['GAIN'] = (1.0, '[e-/ADU] effective gain all channels')
            header['GAIN-P'] = (gain_processed, 'corrected for gain?')


        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
        #    print(stat)

        if get_par(set_zogy.display,tel):
            ds9_arrays(gain_cor=data)



        # old crosstalk correction
        ##########################
        if False and imgtype == 'object':
            # not needed for biases, darks or flats
            try:
                log.info('correcting for the crosstalk')
                xtalk_processed = False
                crosstalk_file = get_par(set_bb.crosstalk_file,tel)
                # data array is corrected in place
                xtalk_corr_old (data, crosstalk_file)
            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [xtalk_corr] of image '
                              '{}: {}'.format(filename, e))
            else:
                xtalk_processed = True
            finally:
                header['XTALK-P'] = (xtalk_processed, 'corrected for crosstalk?')
                header['XTALK-F'] = (crosstalk_file.split('/')[-1],
                                     'name crosstalk coefficients file')


            if get_par(set_zogy.display,tel):
                ds9_arrays(Xtalk_cor=data)



        # overscan correction
        #####################
        try:
            log.info('correcting for the overscan')
            os_processed = False
            data = os_corr(data, header, imgtype, tel=tel)
        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [os_corr] of image {}: {}'
                          .format(filename, e))
        else:
            os_processed = True
        finally:
            header['OS-P'] = (os_processed, 'corrected for overscan?')


        if get_par(set_zogy.display,tel):
            ds9_arrays(os_cor=data)



        # non-linearity correction
        ##########################
        nonlin_corr_processed = False
        header['NONLIN-P'] = (nonlin_corr_processed, 'corrected for '
                              'non-linearity?')
        header['NONLIN-F'] = ('None', 'name non-linearity correction file')

        if imgtype != 'bias' and get_par(set_bb.correct_nonlin,tel):

            try:
                log.info('correcting for the non-linearity')
                nonlin_corr_file = get_par(set_bb.nonlin_corr_file,tel)
                data = nonlin_corr(data, nonlin_corr_file)
                header['NONLIN-F'] = (nonlin_corr_file.split('/')[-1],
                                      'name non-linearity correction file')
            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [nonlin_corr] of '
                              'image {}: {}'.format(filename, e))
            else:
                nonlin_corr_processed = True
            finally:
                header['NONLIN-P'] = nonlin_corr_processed


        # if IMAGETYP=bias, write [data] to fits and return
        if imgtype == 'bias':
            # call [run_qc_check] to update header with any QC flags
            run_qc_check (header, tel)
            # write fits
            fits_out = write_fits (fits_out, data.astype('float32'), header,
                                   tel=tel)
            # close down logging and leave
            close_log(log, logfile)

            return fits_out



        # master bias creation
        ######################
        try:
            # put an multi-processing lock on this block so that only 1
            # process at a time can create the master bias
            lock.acquire()

            # prepare or point to the master bias
            fits_master = '{}/bias/{}_bias_{}.fits'.format(master_path, tel,
                                                           date_eve)
            fits_mbias = master_prep (fits_master, data.shape,
                                      get_par(set_bb.create_master,tel),
                                      tel=tel, proc_mode=proc_mode)

        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during bias [master_prep] of '
                          'master {}: {}'.format(fits_master, e))

        finally:
            lock.release()



        # master bias subtraction
        #########################
        mbias_processed = False
        header['MBIAS-P'] = (mbias_processed, 'corrected for master bias?')
        header['MBIAS-F'] = ('None', 'name of master bias applied')

        # check if mbias needs to be subtracted
        if fits_mbias is not None and get_par(set_bb.subtract_mbias,tel):

            try:
                # and subtract it from the flat or object image
                log.info('subtracting the master bias')
                data_mbias, header_mbias = read_hdulist(fits_mbias,
                                                        get_header=True)
                data -= data_mbias
                del data_mbias
                header['MBIAS-F'] = fits_mbias.split('/')[-1].split('.fits')[0]

                # for object image, add number of days separating
                # image and master bias
                if imgtype == 'object':
                    mjd_obs = header['MJD-OBS']
                    mjd_obs_mb = header_mbias['MJD-OBS']
                    header['MB-NDAYS'] = (
                        np.abs(mjd_obs-mjd_obs_mb),
                        '[days] time between image and master bias used')

            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during master bias '
                              'subtraction of image {}: {}'.format(filename, e))
            else:
                mbias_processed = True
            finally:
                header['MBIAS-P'] = mbias_processed


        # display
        if get_par(set_zogy.display,tel):
            ds9_arrays(bias_sub=data)


        # if IMAGETYP=dark, write [data] to fits and return
        if imgtype == 'dark':
            # call [run_qc_check] to update header with any QC flags
            run_qc_check (header, tel)
            # write fits
            fits_out = write_fits (fits_out, data.astype('float32'), header,
                                   tel=tel)
            # close down logging and leave
            close_log(log, logfile)

            return fits_out



        # master dark creation
        ######################
        if get_par(set_bb.create_mdark,tel):

            try:
                # put an multi-processing lock on this block so that only 1
                # process at a time can create the master bias
                lock.acquire()

                # prepare or point to the master dark
                fits_master = '{}/dark/{}_dark_{}.fits'.format(master_path, tel,
                                                               date_eve)
                fits_mdark = master_prep (fits_master, data.shape,
                                          get_par(set_bb.create_master,tel),
                                          tel=tel, proc_mode=proc_mode)

            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during dark [master_prep] of '
                              'master {}: {}'.format(fits_master, e))

            finally:
                lock.release()



        # create initial mask array
        ###########################
        if imgtype == 'object' or imgtype == 'flat':
            try:
                log.info('preparing the initial mask')
                mask_processed = False
                data_mask, header_mask = mask_init (data, header, filt, imgtype)
            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during [mask_init] for image '
                              '{}: {}'.format(filename, e))
            else:
                mask_processed = True
            finally:
                header['MASK-P'] = (mask_processed, 'mask image created?')


            if get_par(set_zogy.display,tel):
                    ds9_arrays(mask=data_mask)



        # if IMAGETYP=flat, write [data] to fits and return
        if imgtype == 'flat':

            # first add some image statistics to header
            if os_processed:
                log.info ('data.dtype: {}'.format(data.dtype))
                get_flatstats (data, header, data_mask, tel=tel)

            # call [run_qc_check] to update header with any QC flags
            run_qc_check (header, tel)
            # write fits
            fits_out = write_fits (fits_out, data.astype('float32'), header,
                                   tel=tel)
            # close down logging and leave
            close_log(log, logfile)

            return fits_out



        # master flat creation
        ######################
        try:
            # put an multi-processing lock on this block so that only 1
            # process at a time can create the master flat
            lock.acquire()

            # prepare or point to the master flat
            fits_master = '{}/flat/{}_flat_{}_{}.fits'.format(master_path, tel,
                                                              date_eve, filt)
            fits_mflat = master_prep (fits_master, data.shape,
                                      get_par(set_bb.create_master,tel),
                                      tel=tel, proc_mode=proc_mode)

        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during flat [master_prep] of '
                          'master {}: {}'.format(fits_master, e))

        finally:
            lock.release()



        # master flat division
        ######################
        mflat_processed = False
        header['MFLAT-P'] = (mflat_processed, 'corrected for master flat?')
        header['MFLAT-F'] = ('None', 'name of master flat applied')

        if fits_mflat is not None:
            try:
                # and divide the object image by the master flat
                log.info('dividing by the master flat')
                data_mflat, header_mflat = read_hdulist(fits_mflat,
                                                        get_header=True)
                data /= data_mflat
                del data_mflat
                header['MFLAT-F'] = (fits_mflat.split('/')[-1].split('.fits')[0],
                                     'name of master flat applied')

                # for object image, add number of days separating
                # image and master bias
                if imgtype == 'object':
                    mjd_obs = header['MJD-OBS']
                    mjd_obs_mf = header_mflat['MJD-OBS']
                    header['MF-NDAYS'] = (
                        np.abs(mjd_obs-mjd_obs_mf),
                        '[days] time between image and master flat used')

            except Exception as e:
                #log.exception(traceback.format_exc())
                log.exception('exception was raised during master flat division '
                              'of image {}: {}'.format(filename, e))
            else:
                mflat_processed = True
            finally:
                header['MFLAT-P'] = (mflat_processed, 'corrected for master flat?')




        # PMV 2018/12/20: fringe correction is not yet done, but
        # still add these keywords to the header
        header['MFRING-P'] = (False, 'corrected for master fringe map?')
        header['MFRING-F'] = ('None', 'name of master fringe map applied')
        header['FRRATIO'] = ('None', 'fringe ratio (science/fringe map) applied')


        if get_par(set_zogy.display,tel):
            #ds9_arrays(flat_cor=data)
            data_precosmics = np.copy(data)
            data_mask_precosmics = np.copy(data_mask)


        # cosmic ray detection and correction
        #####################################
        try:
            log.info('detecting cosmic rays')
            cosmics_processed = False
            data, data_mask = cosmics_corr(data, header, data_mask, header_mask)
        except Exception as e:
            header['NCOSMICS'] = ('None', '[/s] number of cosmic rays identified')
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [cosmics_corr] of image '
                          '{}: {}'.format(filename, e))
        else:
            cosmics_processed = True
        finally:
            header['COSMIC-P'] = (cosmics_processed, 'corrected for cosmic rays?')


        if get_par(set_zogy.display,tel):
            value_cosmic = get_par(set_zogy.mask_value['cosmic ray'],tel)
            mask_cosmics = (data_mask & value_cosmic != 0)
            data_mask_cosmics = np.zeros_like (mask_cosmics, dtype='uint8')
            data_mask_cosmics[mask_cosmics] = value_cosmic
            log.info ('number of cosmics per second: {}'
                      .format(header['NCOSMICS']))
            ds9_arrays(data=data_precosmics,
                       mask_cosmics=data_mask_cosmics,
                       cosmic_cor=data)




        # new crosstalk correction
        ##########################
        try:
            log.info('correcting for the crosstalk')
            xtalk_processed = False
            crosstalk_file = get_par(set_bb.crosstalk_file,tel)
            # data array is corrected in place; data_mask remains unchanged
            xtalk_corr (data, crosstalk_file, data_mask)
        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [xtalk_corr] of image '
                          '{}: {}'.format(filename, e))
        else:
            xtalk_processed = True
        finally:
            header['XTALK-P'] = (xtalk_processed, 'corrected for crosstalk?')
            header['XTALK-F'] = (crosstalk_file.split('/')[-1],
                                 'name crosstalk coefficients file')




        # satellite trail detection
        ###########################
        try:
            sat_processed = False
            if get_par(set_bb.detect_sats,tel):
                log.info('detecting satellite trails')

                if get_par(set_bb.use_asta,tel):

                    # new method using Fiore's ASTA package
                    # (https://github.com/FiorenSt/ASTA)
                    data_mask = run_asta(data, header, data_mask, header_mask,
                                         tmp_path)
                else:

                    # old method using acstools
                    data_mask = sat_detect(data, header, data_mask, header_mask,
                                           tmp_path)



        except Exception as e:
            header['NSATS'] = ('None', 'number of satellite trails identified')
            #log.exception(traceback.format_exc())
            if get_par(set_bb.use_asta,tel):
                modname = run_asta
            else:
                modname = sat_detect

            log.exception('exception was raised during [{}] of image {}: '
                          '{}'.format(modname, filename, e))
        else:
            if get_par(set_bb.detect_sats,tel):
                sat_processed = True
        finally:
            header['SAT-P'] = (sat_processed, 'processed for satellite trails?')


        # add some more info to mask header
        result = mask_header(data_mask, header_mask)

        # set edge pixel values to zero
        value_edge = get_par(set_zogy.mask_value['edge'],tel)
        mask_edge = (data_mask & value_edge == value_edge)
        if False:
            data[mask_edge] = 0


        # set edge pixels to median of corresponding channel to avoid
        # initial source-extractor run leading to a wrong background
        # estimation near the edge due to the sudden jump in level
        __, __, __, __, data_sec_red = (define_sections(np.shape(data), tel=tel))
        nchans = np.shape(data_sec_red)[0]
        for i_chan in range(nchans):
            # channel section
            sec_chan = data_sec_red[i_chan]
            mask_edge_chan = mask_edge[sec_chan]
            data[sec_chan][mask_edge_chan] = np.median(data[sec_chan])



        # write data and mask to output images in [tmp_path] and
        # add name of reduced image and corresponding mask in header just
        # before writing it
        log.info('writing reduced image and mask to {}'.format(tmp_path))
        redfile = fits_out.split('/')[-1].split('.fits')[0]
        header['REDFILE'] = (redfile, 'BlackBOX reduced image name')
        header['MASKFILE'] = (redfile.replace('_red', '_mask'),
                              'BlackBOX mask image name')
        header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(new_fits, data.astype('float32'), header, overwrite=True)
        header_mask['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(new_fits_mask, data_mask.astype('uint8'), header_mask,
                     overwrite=True)
        del data, data_mask


        # also write separate header fits file - done just below the QC check
        #hdulist = fits.HDUList(fits.PrimaryHDU(header=header))
        #hdulist.writeto(new_fits.replace('.fits', '_hdr.fits'), overwrite=True)


        # check quality control
        qc_flag = run_qc_check (header, tel, check_key_type='full')

        # update [new_fits] header with qc-flags
        header_update = read_hdulist(new_fits, get_data=False, get_header=True)
        for key in header:
            if 'QC' in key or 'DUMCAT' in key:
                log.info ('updating header keyword {} with: {} for image {}'
                          .format(key, header[key], new_fits))
                header_update[key] = (header[key], header.comments[key])


        # update fits header and create separate header file
        update_imcathead (new_fits, header_update, create_hdrfile=True)


        # if header of object image contains a red flag, create dummy
        # binary fits catalogs (both 'new' and 'trans') and return,
        # skipping zogy's [optimal subtraction] below
        if qc_flag=='red':
            log.error('red QC flag in image {}; making dummy catalogs and '
                      'returning'.format(fits_out))
            run_qc_check (header, tel, cat_type='new', cat_dummy=fits_tmp_cat,
                          check_key_type='full')
            run_qc_check (header, tel, cat_type='trans', cat_dummy=fits_tmp_trans,
                          check_key_type='trans')

            # verify headers of catalogs
            verify_header (fits_tmp_cat, ['raw','full'])
            verify_header (fits_tmp_trans, ['raw','full','trans'])

            # run match2SSO to find known asteroids in the observation
            call_match2SSO(fits_tmp_trans, tel)

            # copy selected output files to new directory
            list_2keep = get_par(set_bb.all_2keep,tel)
            copy_files2keep(tmp_base, new_base, list_2keep,
                            move=(not get_par(set_bb.keep_tmp,tel)))

            # only for ML, create symbolic links in alternative
            # directory structure if transient catalog is involved;
            # turn off for now
            if False:
                if tel=='ML1' and '_trans.fits' in list_2keep:
                    create_symlinks (new_base, obj, filt)

            # close log and remove tmp folder
            close_log(log, logfile)
            clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
            return fits_out


    # end of if block with reduction steps

    if False:
        t_gc = time.time()
        ncollected = gc.collect()
        log.info ('garbage collector: collected {} objects in {:.2f}s'
                  .format(ncollected, time.time()-t_gc))


    # block dealing with main processing switches
    #############################################

    # if both catalog and transient extraction are switched off, then
    # no need to execute [optimal_subtraction]
    if (not get_par(set_bb.cat_extract,tel) and
        not get_par(set_bb.trans_extract,tel)):

        log.info('main processing switches cat_extract and trans_extract are off, '
                 'nothing left to do for {}'.format(filename))


        if do_reduction:

            # verify image header
            verify_header (new_fits, ['raw'])

            # if reduction steps were performed, copy selected output
            # files to new directory and clean up tmp folder if needed
            copy_files2keep(tmp_base, new_base,
                            get_par(set_bb.img_reduce_exts,tel),
                            move=(not get_par(set_bb.keep_tmp,tel)))
            close_log(log, logfile)
            clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
            return fits_out

        else:

            # even if reduction steps were skipped, tmp folder is
            # still present
            close_log(log, logfile)
            clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
            return None


    elif not get_par(set_bb.force_reproc_new,tel):

        # stop processing here if cat_extract and/or trans_extract
        # products exist in reduced folder
        #new_list = glob.glob('{}*'.format(new_base))
        new_list = list_files(new_base)
        ext_list = []

        fits_cat = '{}_red_cat.fits'.format(new_base)
        fits_trans = '{}_red_trans.fits'.format(new_base)

        if get_par(set_bb.trans_extract,tel) and ref_present:
            # if [trans_extract] is set to True and a reference image
            # is present, both the cat_extract and trans_extract
            # products should be present, even when [cat_extract] is
            # set to False
            ext_list += get_par(set_bb.cat_extract_exts,tel)
            ext_list += get_par(set_bb.trans_extract_exts,tel)
            text = 'cat_extract and trans_extract'

            # check if transient catalog is a dummy
            dumcat = is_dumcat(fits_trans)

            if get_par(set_bb.cat_extract,tel):
                # check if full-source catalog is also a dummy
                dumcat &= is_dumcat(fits_cat)


        elif get_par(set_bb.cat_extract,tel):
            ext_list += get_par(set_bb.cat_extract_exts,tel)
            text = 'cat_extract'

            # check if full-source catalog is a dummy
            dumcat = is_dumcat(fits_cat)


        # N.B.: note that the presence of sso files are not checked
        # for, so the reduction will not be redone if they are not
        # present; they can easily be redone in day mode of match2sso
        present = (np.array([ext in fn or 'sso' in ext
                             for ext in ext_list for fn in new_list])
                   .reshape(len(ext_list),-1).sum(axis=1))
        #log.info ('present: {}'.format(present))


        if np.all(present) or dumcat:

            if not dumcat:
                log.info ('force_reproc_new is False and all {} data products '
                          'already present in reduced folder (ref image present?'
                          ': {}); nothing left to do for {}'
                          .format(text, ref_present, filename))
            else:
                log.info ('force_reproc_new is False and full-source and/or '
                          'transient catalog is a dummy; nothing left to do for '
                          '{}'.format(filename))

            close_log(log, logfile)
            clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))

            if do_reduction:
                return fits_out
            else:
                return None

        else:

            log.info ('copying existing {} data products from reduced to tmp '
                      'folder to avoid repeating processing steps for {}'
                      .format(text, filename))

            # otherwise, copy cat_extract products and trans_extract
            # to tmp folder and continue
            copy_files2keep(new_base, tmp_base, ext_list, move=False,
                            run_fpack=False)


    elif get_par(set_bb.force_reproc_new,tel) and not do_reduction:

        # if [force_reproc_new]=True, then depending on exact
        # switches, remove relevant files from reduced folder and copy
        # files to the tmp folder; this is not needed if basic
        # reduction was performed, i.e. do_reduction=True
        #new_list = glob.glob('{}*'.format(new_base))
        new_list = list_files(new_base)
        ext_list = []


        # should full-source catalog be empty, also redo the
        # catalog extraction as this is a forced rereduction
        fits_cat = '{}_red_cat.fits'.format(new_base)
        dumcat = is_dumcat(fits_cat)
        if not isfile(fits_cat):
            # if full-source catalog does not exist, force
            # the re-extraction
            dumcat = True

        if dumcat:
            log.info ('full-source catalog is a dummy (zero entries) or does '
                      'not exist for {}; re-extracting it'.format(filename))

        # if cat_extract=True or full-source catalog is a dummy
        # catalog, then remove all cat and trans products
        if get_par(set_bb.cat_extract,tel) or dumcat:

            log.info ('forced reprocessing: removing all existing cat_extract '
                      'and trans_extract products in reduced folder for {}'
                      .format(filename))

            ext_list += get_par(set_bb.cat_extract_exts,tel)
            ext_list += get_par(set_bb.trans_extract_exts,tel)


            # uncompress new_fits if needed
            __, file_tmp = already_exists (new_fits, get_filename=True)
            if '.fz' in file_tmp:
                unzip (file_tmp)


            # clear any pre-existing qc-flags from [new_fits] header,
            # including keywords that determine whether [get_back],
            # [run_wcs] and [format_cat] are rerun
            header_update = read_hdulist(new_fits, get_data=False,
                                         get_header=True)
            keys = ['DUMCAT', 'QC-FLAG', 'QCRED', 'QCORA', 'QCYEL',
                    'FORMAT-P', 'CTYPE1', 'CTYPE2']
            # don't add BKG-SUB as otherwise background could be
            # determined from image that was already background
            # subtracted and then the original background will be
            # lost , 'BKG-SUB']
            for key in keys:
                if 'QCRED' in key or 'QCORA' in key or 'QCYEL' in key:
                    keys2del = ['{}{}'.format(key[0:5], i)
                                for i in range(1,100)]
                else:
                    keys2del = [key]

                for key2del in keys2del:
                    if key2del in header_update:
                        log.info ('deleting keyword {} from header of {}'
                                  .format(key2del, new_fits))
                        del header_update[key2del]
                    else:
                        break

            # update fits header and create separate header file
            update_imcathead (new_fits, header_update, create_hdrfile=True)


        elif get_par(set_bb.trans_extract,tel):

            log.info ('forced reprocessing: removing all existing '
                      'trans_extract products in reduced folder for {}'
                      .format(filename))

            # only remove trans_extract products
            ext_list += get_par(set_bb.trans_extract_exts,tel)

            # but before doing that, copy the cat_extract products to
            # the tmp folder, as the cat_extract can be skipped
            copy_files2keep(new_base, tmp_base,
                            get_par(set_bb.cat_extract_exts,tel),
                            move=False, run_fpack=False)



        # now files in reduced folder can be removed
        files_2remove = [fn for fn in new_list for ext in ext_list if ext in fn]
        lock.acquire()
        #for file_2remove in files_2remove:
        #    log.info ('removing existing {}'.format(file_2remove))
        #    os.remove(file_2remove)
        remove_files(files_2remove, verbose=True)
        lock.release()



    # if cat_extract is True while img_reduce is False, delete the jpg
    # previously created, as it may include the background which is
    # removed in cat_extract
    if get_par(set_bb.cat_extract,tel) and not get_par(set_bb.img_reduce,tel):
        file_jpg = '{}_red.jpg'.format(new_base)
        if isfile(file_jpg):
            #os.remove (file_jpg)
            remove_files ([file_jpg])


    # before continuing, zipped files in tmp folder need to be
    # unzipped/funpacked for optimal_subtraction to process them
    #tmp_files = glob.glob('{}*.fz'.format(tmp_base))
    tmp_files = list_files(tmp_base, end_str='.fz')


    if len(tmp_files) > 0:
        log.info ('unpacking files in tmp folder for {}'.format(filename))
        for tmp_file in tmp_files:
            # and funpack/unzip if necessary
            tmp_file = unzip(tmp_file, put_lock=False)


    # run zogy's [optimal_subtraction]
    ##################################
    log.info ('running optimal image subtraction')


    # change to [tmp_path]; only necessary if making plots as
    # PSFEx is producing its diagnostic output fits and plots in
    # the current directory
    if get_par(set_zogy.make_plots,tel):
        os.chdir(tmp_path)


    # some info on ref image
    log.info ('ref image {} for {} already present?: {}'
              .format(ref_fits_in, filename, ref_present))
    log.info ('input [create_ref] switch set to: {}'
              .format(get_par(set_bb.create_ref,tel)))


    # in case transient extraction step is switched off, or in case
    # the ref image does not exist and it is not to be created from
    # this image, run zogy on new image only
    if (not get_par(set_bb.trans_extract,tel) or
        (not get_par(set_bb.create_ref,tel) and not ref_present)):

        log.info('set_bb.trans_extract={}; processing new image only, '
                 'without comparison to ref image'
                 .format(get_par(set_bb.trans_extract,tel)))
        log.info('new_fits: {}'.format(new_fits))
        log.info('new_fits_mask: {}'.format(new_fits_mask))

        try:
            zogy_processed = False
            header_new = optimal_subtraction(
                new_fits=new_fits, new_fits_mask=new_fits_mask,
                set_file='set_zogy', verbose=None, redo_new=None,
                nthreads=set_bb.nthreads, telescope=tel,
                keep_tmp=get_par(set_bb.keep_tmp,tel))


            # strip header from image- or table-dependent keywords
            header_new.strip()

            # add offset between RA/DEC-CNTR coords and ML/BG field
            # definition to the header
            radec_offset (header_new, filename)


        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [optimal_subtraction] for '
                          'new-only image {}: {}'.format(new_fits, e))
        else:
            zogy_processed = True
        finally:
            if not zogy_processed:
                # copy selected output files to red directory and
                # remove tmp folder corresponding to the image
                log.error ('due to unexpected exception, saving just the image '
                           'reduction products and returning from '
                           '[blackbox_reduce] prematurely')
                copy_files2keep(tmp_base, new_base,
                                get_par(set_bb.img_reduce_exts,tel),
                                move=(not get_par(set_bb.keep_tmp,tel)))

                #clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
                # keeping tmp folder to be able to help understand exception
                close_log(log, logfile)
                clean_tmp(tmp_path, True)
                return None

            else:
                # feed [header_new] to [run_qc_check], and make
                # dummy catalogs if there is a red flag
                try:
                    qc_flag = run_qc_check (header_new, tel,
                                            check_key_type='full')
                except Exception as e:
                    #log.exception(traceback.format_exc())
                    log.exception('exception was raised during [run_qc_check] '
                                  'for new-only image {}: {}'.format(new_fits, e))

                if qc_flag=='red':
                    log.error('red QC flag in [header_new] returned by new-only '
                              '[optimal_subtraction]; making dummy catalogs')
                    run_qc_check (header_new, tel, cat_type='new',
                                  cat_dummy=fits_tmp_cat,
                                  check_key_type='full')
                    # make copy to avoid keywords related to transient
                    # catalog (TQC-FLAG and TDUMCAT) being added to
                    # [header_new]
                    header_trans = header_new.copy()
                    run_qc_check (header_trans, tel, cat_type='trans',
                                  cat_dummy=fits_tmp_trans,
                                  check_key_type='trans')

                else:
                    # update full-source catalog header with latest
                    # qc-flags; transient catalog not needed
                    log.info ('updating new catalog header with QC flags')
                    update_imcathead (fits_tmp_cat, header_new,
                                      create_hdrfile=True)


                # update reduced image header with extended header
                # from ZOGY's optimal_subtraction; needs to be done
                # also when there is a red flag - not needed for
                # catalog header as the dummy catalogs _cat.fits and
                # _trans.fits will be created in function [qc]
                update_imcathead (new_fits, header_new, create_hdrfile=True)


    else:

        # block that runs zogy on two images: new and ref

        # copy all relevant reference files to the tmp folder; could
        # make symbolic links but copying is cleaner to avoid editing
        # the files (at least the header is updated by zogy in
        # function prep_optimal_subtraction)
        ref_files = list_files(ref_fits_in.split('_red.fits')[0])
        for ref_file in ref_files:

            tmp_file = '{}/{}'.format(tmp_path, ref_file.split('/')[-1])

            # copy the file to tmp_path
            copy_file (ref_file, tmp_file)
            # and unzip if needed
            unzip(tmp_file, put_lock=False)



        ref_fits = '{}/{}'.format(tmp_path, ref_fits_in.split('/')[-1])
        ref_fits_mask = ref_fits.replace('_red.fits', '_mask.fits')

        log.info('new_fits: {}'.format(new_fits))
        log.info('new_fits_mask: {}'.format(new_fits_mask))
        log.info('ref_fits: {}'.format(ref_fits))
        log.info('ref_fits_mask: {}'.format(ref_fits_mask))


        try:
            zogy_processed = False
            header_new, header_trans = optimal_subtraction(
                new_fits=new_fits, ref_fits=ref_fits, new_fits_mask=new_fits_mask,
                ref_fits_mask=ref_fits_mask, set_file='set_zogy',
                verbose=None, redo_new=None, redo_ref=None,
                nthreads=set_bb.nthreads, telescope=tel,
                keep_tmp=get_par(set_bb.keep_tmp,tel))


            # strip headers from image- or table-dependent keywords
            header_new.strip()
            header_trans.strip()

            # add offset between RA/DEC-CNTR coords and ML/BG field
            # definition to the new header
            radec_offset (header_new, filename)


        except Exception as e:
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during [optimal_subtraction] for '
                          'new image {} and reference image {}: {}'
                          .format(new_fits, ref_fits, e))
        else:
            zogy_processed = True
        finally:
            if not zogy_processed:
                # copy selected output files to red directory and
                # remove tmp folder corresponding to the image
                log.error ('due to unexpected exception, saving just the image '
                           'reduction products and returning from '
                           '[blackbox_reduce] prematurely')
                copy_files2keep(tmp_base, new_base,
                                get_par(set_bb.img_reduce_exts,tel),
                                move=(not get_par(set_bb.keep_tmp,tel)))

                # remove cat_extract and trans_extract products?

                #clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))
                # keeping tmp folder to be able to help understand exception
                close_log(log, logfile)
                clean_tmp(tmp_path, True)
                return None

            else:
                # feed [header_new] to [run_qc_check], and if there
                # is a red flag: make output dummy catalog
                qc_flag = run_qc_check (header_new, tel,
                                        check_key_type='full')
                if qc_flag=='red':
                    log.error('red QC flag in [header_new] returned by new '
                              'vs. ref [optimal_subtraction]: making dummy '
                              'full-source catalog')
                    run_qc_check (header_new, tel, cat_type='new',
                                  cat_dummy=fits_tmp_cat,
                                  check_key_type='full')
                else:
                    log.info ('updating new catalog header with QC flags')
                    # update full-source catalog fits header with latest
                    # qc-flags
                    update_imcathead (fits_tmp_cat, header_new,
                                      create_hdrfile=True)


                # same for transient catalog; N.B. header.strip() will
                # strip header in place, and return None, so use
                # header.copy(strip=True) to return stripped header
                header_newtrans = header_new.copy()
                header_newtrans.update(header_trans)

                tqc_flag = run_qc_check (header_newtrans, tel,
                                         check_key_type='trans')
                if qc_flag=='red' or tqc_flag=='red':
                    log.error('red transient QC flag in [header_newtrans] '
                              'returned by new vs ref [optimal_subtraction]: '
                              'making dummy transient catalog')
                    run_qc_check (header_newtrans, tel, cat_type='trans',
                                  cat_dummy=fits_tmp_trans,
                                  check_key_type='trans')
                else:
                    # update transient catalog header with latest qc-flags
                    log.info ('updating trans catalog header with QC flags')
                    try:
                        # first try using astropy
                        update_imcathead (fits_tmp_trans, header_newtrans,
                                          create_hdrfile=True, use_fitsio=False)
                    except Exception as e:
                        # in case of an exception due to a very large
                        # transient catalog, resort to fitsio
                        log.info ('exception occurred while updating transient '
                                  'catalog {} using astropy; now trying with '
                                  'fitsio: {}'.format(fits_tmp_trans, e))
                        update_imcathead (fits_tmp_trans, header_newtrans,
                                          create_hdrfile=True, use_fitsio=True)


                    # also update the trans_light header
                    fits_tmp_light = fits_tmp_trans.replace('.fits',
                                                            '_light.fits')
                    if isfile(fits_tmp_light):
                        update_imcathead (fits_tmp_light, header_newtrans)
                    else:
                        log.warn ('{} does not exist'.format(fits_tmp_light))



                # update reduced new image header with extended header
                # from ZOGY's optimal_subtraction; no need to update
                # the ref image header
                update_imcathead (new_fits, header_new, create_hdrfile=True)



    # verify headers of catalogs
    verify_header (fits_tmp_cat, ['raw','full'])
    verify_header (fits_tmp_trans, ['raw','full','trans'])


    # run match2SSO to find known asteroids in the observation
    call_match2SSO(fits_tmp_trans, tel)


    # if transient catalog exists
    if isfile(fits_tmp_trans):

        # create png thumbnails for database
        if get_par(set_bb.save_thumbnails_pngs,tel):
            #and qc_flag != 'red' and tqc_flag != 'red':
            dir_dest = '{}/{}'.format(thumbnails_path, tmp_base.split('/')[-1])
            save_png_thumbnails (fits_tmp_trans, dir_dest,
                                 nthreads=set_bb.nthreads)


        # if not keeping thumbnails as columns in transient catalog,
        # make a copy of the light version
        fits_tmp_light = fits_tmp_trans.replace('.fits', '_light.fits')
        save_thumbnails = get_par(set_bb.save_thumbnails,tel)
        if not save_thumbnails and os.path.exists(fits_tmp_light):
            copy_file (fits_tmp_light, fits_tmp_trans)



    # list of files to copy/move to reduced folder; need to include
    # the img_reduce products in any case because the header will have
    # been updated with fresh QC flags
    list_2keep = copy.deepcopy (get_par(set_bb.img_reduce_exts,tel))
    # source extraction products
    if get_par(set_bb.cat_extract,tel):
        list_2keep += get_par(set_bb.cat_extract_exts,tel)
    elif qc_flag == 'red':
        # make sure to copy dummy source catalog in case of a red flag
        list_2keep += ['_cat.fits']
        list_2keep += ['_cat_hdr.fits']


    # transient extraction products
    if get_par(set_bb.trans_extract,tel):
        list_2keep += get_par(set_bb.trans_extract_exts,tel)
    elif qc_flag == 'red':
        # make sure to copy dummy source catalog in case of a red flag
        list_2keep += ['_trans.fits']
        list_2keep += ['_trans_hdr.fits']
        list_2keep += ['_trans_light.fits']


    # copy/move files over
    copy_files2keep(tmp_base, new_base, list_2keep,
                    move=(not get_par(set_bb.keep_tmp,tel)))


    # if original filename contains ADC, save any *dRADEC* files from
    # tmp to /idia/projects/meerlicht/ADCtests/tel_yyyymmdd or
    # /home/sa_105685508700717199458/ADCtests/tel_yyyymmdd folders
    if 'adc' in filename.lower() and tel in ['ML1', 'BG2', 'BG3', 'BG4']:

        if tel == 'ML1':
            dest_dir = ('/idia/projects/meerlicht/ADCtests/{}_{}'
                        .format(tel, date_eve))
        else:
            dest_dir = ('/home/sa_105685508700717199458/ADCtests/{}_{}'
                        .format(tel, date_eve))

        # make directory
        make_dir (dest_dir)


        #adc_files = glob.glob('{}*dRADEC*'.format(tmp_base))
        adc_files = list_files(tmp_base, search_str='dRADEC')


        # add header file - needs to come from new_base as it might
        # have been moved over
        adc_files.append('{}_red_hdr.fits'.format(new_base))
        for adc_file in adc_files:
            copy_file (adc_file, dest_dir+'/')



    # only for ML, create symbolic links in alternative directory
    # structure if transient catalog is involved; turn off for now
    if False:
        if tel=='ML1' and '_trans.fits' in list_2keep:
            create_symlinks (new_base, obj, filt)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t_blackbox_reduce,
                           label='in blackbox_reduce at end')

    close_log(log, logfile)
    clean_tmp(tmp_path, get_par(set_bb.keep_tmp,tel))

    return fits_out


################################################################################

def save_png_thumbnails (fits_trans, dir_dest, nthreads=1):

    """function to convert thumbnails in transient catalog
    [fits_trans] to separate png files (to be used by the Database)
    with names [number]_RED.png, [number]_REF.png, [number]_D.png,
    [number]_SCORR.png, where number is the row number - also
    indicated by the NUMBER column - in [fits_trans]. The pngs are
    copied/moved to [dest_folder].

    """

    log.info ('running save_png_thumbnails')

    if get_par(set_zogy.timing,tel):
        t = time.time()

    # to avoid reading potentially very large transient fits table in
    # one go, use fitsio to first read all rows but only for a
    # specific column; output data is a numpy recarray, which is
    # easily converted to an astropy table
    table = Table(fitsio.read(fits_trans, ext=-1, columns=['NUMBER']))
    nrows = len(table)


    if nrows > 0:

        # define thumbnail columns to save
        cols2save = ['RED', 'REF', 'D', 'SCORR']
        cols2save = ['THUMBNAIL_{}'.format(c) for c in cols2save]


        # define tmp and destination folders
        dir_tmp = os.path.dirname(fits_trans)


        # use multiprocessing to process rows in fits_trans, creating
        # pngs in tmp folder
        pool_func (save_thumbs_row, range(nrows), fits_trans,
                   cols2save, dir_tmp, nproc=nthreads)


        # search string to identify the pngs created (to distinguish
        # them from other png files in tmp folder)
        search_str = '{}/[0-9]*_[DRS]*.png'.format(dir_tmp)


        # make sure destination folder is empty, otherwise different
        # reductions of the same image might lead to a mix of pngs
        if isdir(dir_dest):
            log.warning ('removing all existing files in {}'.format(dir_dest))
            if dir_dest[0:5] == 'gs://':
                #cmd = ['gcloud', 'storage', 'rm', '{}/*'.format(dir_dest)]
                cmd = ['gsutil', 'rm', '{}/*'.format(dir_dest)]
                result = subprocess.run(cmd)
            else:
                make_dir (dir_dest, empty=True)


        # copy or move to destination folder; if the destination is a
        # Google Cloud bucket, then copying one by one just after
        # creation in function save_thumbs_row() is very slow (about
        # 1min for 100 files), so best to copy/move them with single
        # command here
        move = (not get_par(set_bb.keep_tmp,tel))
        if dir_dest[0:5] == 'gs://':

            if move:
                cp_cmd = 'mv'
            else:
                cp_cmd = 'cp'


            # gsutil command (not actively supported anymore)
            cmd = ['gsutil', '-m', '-q', cp_cmd, search_str, dir_dest]
            # gcloud storage alternative
            #cmd = ['gcloud', 'storage', cp_cmd, search_str, dir_dest]

            result = subprocess.run(cmd)

        else:

            # if not in Google cloud, copy files one by one
            filenames_png = glob.glob(search_str)

            for png_tmp in filenames_png:
                png_dest = '{}/{}'.format(dir_dest, png_tmp.split('/')[-1])
                copy_file (png_tmp, png_dest, move=move, verbose=False)


    else:
        log.warning ('zero rows in {}; no thumbnails to save'
                     .format(fits_trans))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in save_png_thumbnails')


################################################################################

def save_thumbs_row (i_row, fits_trans, cols2save, dir_tmp, move=False):

    # read all columns of fits_trans row with index i_row
    table_row = Table(fitsio.read(fits_trans, ext=-1, rows=i_row))[0]


    # transient number in catalog
    #number = i_row+1
    number = table_row['NUMBER']

    # loop thumbnails
    for col in cols2save:

        # fetch data array from table column and flip/scale it
        data = np.flipud(table_row[col])
        vmin, vmax = zscale().get_limits(data)
        data = scale_data(data, vmin, vmax)

        # save to file in tmp folder
        fn = '{}_{}.png'.format(number, col.split('_')[-1])
        png_tmp = '{}/{}'.format(dir_tmp, fn)
        image = Image.fromarray(data)
        image.save(png_tmp)



################################################################################

def scale_data (data, vmin, vmax):

    """scale input (float) data array between the values vmin and vmax
    to an integer (uint8) range from 0 to 255"""

    # scale data
    data -= vmin
    data /= (vmax-vmin)
    data *= 255
    data[data<0] = 0
    data[data>255] = 255

    return data.astype('uint8')


################################################################################

def create_symlinks (new_base, obj, filt):

    # extensions for which to create symbolic links
    symlink_exts = ['_cat.fits', '_cat_hdr.fits', '_trans.fits',
                    '_trans_hdr.fits', '_trans_light.fits']

    src_files = ['{}_red{}'.format(new_base, ext) for ext in symlink_exts]
    files_present = np.array([isfile(f) for f in src_files])

    # only creating symbolic links if all extensions are present
    if np.all(files_present):

        # create target directory if not already existing
        target_dir = '{}/fields/{}/{}'.format(get_par(set_bb.red_dir,tel),
                                              obj, filt)
        make_dir (target_dir)


        for src_file in src_files:

            src_dir, src_name = os.path.split(src_file)
            target_file = '{}/{}'.format(target_dir, src_name)

            try:
                os.symlink (src_file, target_file)
                log.info ('creating symbolic link: {} -> {}'
                          .format(target_file, src_file))

            except FileExistsError:
                log.error ('target_file {} already exist; symbolic link not '
                           'made for {}'.format(target_file, src_file))

    else:

        log.warning ('these source files do not exist: {}; not creating any '
                     'symbolic link for files with new_base {}'
                     .format(np.array(src_files)[~files_present], new_base))


    return


################################################################################

def is_dumcat (fits_cat):

    dumcat = False
    if isfile(fits_cat):
        header_cat = read_hdulist(fits_cat, get_data=False, get_header=True)
        if 'DUMCAT' in header_cat:
            dumcat = header_cat['DUMCAT']
        elif 'NAXIS2' in header_cat and header_cat['NAXIS2']==0:
            dumcat = True

    else:
        log.warning ('catalog {} does not exist'.format(fits_cat))

    return dumcat


################################################################################

def verify_header (filename, htypes=None):

    """function to verify the presence of keywords in the header of the
       input fits file [filename], where the type of header to check
       is determined by [htypes]. The latter can be a string or list
       of strings with one or more of the following relevant values:
       'raw', 'bias', 'mbias', 'flat', 'mflat', 'mask', 'full', 'ref'
       or 'trans'.

    """

    # dictionary
    dict_head = {
        # raw header
        # commenting out SIMPLE, BSCALE and BZERO - basic keywords
        # that will be present in images but not in binary fits tables
        #'SIMPLE':   {'htype':'raw', 'dtype':bool,  'DB':False, 'None_OK':True},
        #'BSCALE':   {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        #'BZERO':    {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'BITPIX':   {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'NAXIS':    {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'NAXIS1':   {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'NAXIS2':   {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'BUNIT':    {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        #'CCD-AMP':  {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'SET-TEMP': {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'CCD-TEMP': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'XBINNING': {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'YBINNING': {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        #'CCD-SET':  {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'ALTITUDE': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'AZIMUTH':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'DOMEAZ':   {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'RADESYS':  {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'EPOCH':    {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'RA':       {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'RA-REF':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        #'RA-TEL':   {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'DEC':      {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'DEC-REF':  {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        #'DEC-TEL':  {'htype':'raw', 'dtype':float, 'DB':False, 'None_OK':True},
        'HA':       {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'FLIPSTAT': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'EXPTIME':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'ISTRACKI': {'htype':'raw', 'dtype':bool,  'DB':False, 'None_OK':True},
        'ACQSTART': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'ACQEND':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'GPSSTART': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'GPSEND':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'GPS-SHUT': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'DATE-OBS': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'MJD-OBS':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'LST':      {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'UTC':      {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'TIMESYS':  {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'ORIGIN':   {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'MPC-CODE': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'TELESCOP': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'CL-BASE':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'RH-MAST':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'RH-DOME':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'RH-AIRCO': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'RH-PIER':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'PRESSURE': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-PIER':   {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-DOME':   {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-ROOF':   {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-AIRCO':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-MAST':   {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-STRUT':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-CRING':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-SPIDER': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-FWN':    {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-FWS':    {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-M2HOLD': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-GUICAM': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-M1':     {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-CRYWIN': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-CRYGET': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-CRYCP':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'PRES-CRY': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'WINDAVE':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'WINDGUST': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'WINDDIR':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'SITELAT':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'SITELONG': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'ELEVATIO': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        #'WEATIME':  {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'FILTER':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        #'FILTERID': {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'CCD-ID':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'CONTROLL': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'DETSPEED': {'htype':'raw', 'dtype':int,   'DB':True,  'None_OK':True},
        'CCD-NW':   {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'CCD-NH':   {'htype':'raw', 'dtype':int,   'DB':False, 'None_OK':True},
        'INSTRUME': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'FOCUSPOS': {'htype':'raw', 'dtype':int,   'DB':True,  'None_OK':True},
        'IMAGETYP': {'htype':'raw', 'dtype':str,   'DB':False, 'None_OK':True},
        'OBJECT':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'AIRMASS':  {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':False},
        'ORIGFILE': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':False},
        'OBSERVER': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'ABOTVER':  {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'PROGNAME': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'PROGID':   {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'GUIDERST': {'htype':'raw', 'dtype':str,   'DB':True,  'None_OK':True},
        'GUIDERFQ': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'TRAKTIME': {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'ADCX':     {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        'ADCY':     {'htype':'raw', 'dtype':float, 'DB':True,  'None_OK':True},
        #
        # full header
        'BB-V':     {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':False},
        'BB-START': {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':False},
        'KW-V':     {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':False},
        #'LOG':      {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'LOG-IMA':  {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'N-INFNAN': {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'XTALK-P':  {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'XTALK-F':  {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'NONLIN-P': {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'NONLIN-F': {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'GAIN-P':   {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'GAIN':     {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'GAIN1':    {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'GAIN16':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'OS-P':     {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'BIASMEAN': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'BIASM1':   {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'BIASM16':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'RDNOISE':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'RDN1':     {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'RDN16':    {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'BIAS1A0':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BIAS1A1':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'VFITOK1':  {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'BIAS16A0': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BIAS16A1': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'VFITOK16': {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'MBIAS-P':  {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'MBIAS-F':  {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'MB-NDAYS': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'SATURATE': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'NOBJ-SAT': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'MFLAT-P':  {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'MFLAT-F':  {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'MF-NDAYS': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'MFRING-P': {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'MFRING-F': {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'FRRATIO':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'COSMIC-P': {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'NCOSMICS': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'SAT-P':    {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'NSATS':    {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'REDFILE':  {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'MASKFILE': {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'S-P':      {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'S-V':      {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'S-NOBJ':   {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'S-FWHM':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'S-FWSTD':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'S-SEEING': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-SEESTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-ELONG':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-ELOSTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-BKG':    {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-BKGSTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'S-VIGNET': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'BKG-CORR': {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'BKG-CHI2': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BKG-CF1':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BKG-CF16': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BKG-FDEG': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'BKG-FC0':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'A-P':      {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'A-V':      {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'A-INDEX':  {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'A-PSCALE': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-PSCALX': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'A-PSCALY': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'A-ROT':    {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-ROTX':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'A-ROTY':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'A-CAT-F':  {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'A-NAST':   {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'A-TNAST':  {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'A-NAMAX':  {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'A-DRA':    {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-DRASTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-DDEC':   {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-DDESTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'A-NA-OFF': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PSF-P':    {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'PSF-V':    {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        'PSF-RAD':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-RADP': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-SIZE': {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'PSF-FRAC': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-SAMP': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-CFGS': {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'PSF-NOBJ': {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'PSF-FIX':  {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'PSF-PLDG': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PSF-CHI2': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PSF-FWHM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-SEE':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PSF-PMIN': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-PMAX': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-PMED': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-PSTD': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-BMIN': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-BMAX': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-BMED': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-BSTD': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMNM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMXM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMDM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-ESTM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMNM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMXM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMDM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FSTM': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMNG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMXG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-EMDG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-ESTG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMNG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMXG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FMDG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PSF-FSTG': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PC-P':     {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'PC-CAT-F': {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':True},
        'PC-NCAL':  {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'PC-TNCAL': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        #'PC-FNCAL': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        #'PC-NCMAX': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PC-NCMIN': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        #'PC-ZPCHN': {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'PC-ZPFDG': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PC-ZPF0':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PC-TNSUB': {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PC-NSUB':  {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'PC-MZPD':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PC-MZPS':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PC-MZPE':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PC-ZPDEF': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'PC-ZP':    {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PC-ZPSTD': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PC-ZPERR': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PC-EXTCO': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'AIRMASSC': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'RA-CNTR':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'DEC-CNTR': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'AZ-CNTR':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'ALT-CNTR': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'BJD-OBS':  {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'PC-AIRM':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'NSIGMA':   {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'LIMEFLUX': {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'LIMFNU':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'LIMMAG':   {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'NOBJECTS': {'htype':'full', 'dtype':int,   'DB':True,  'None_OK':True},
        'NGAIA':    {'htype':'full', 'dtype':int,   'DB':False, 'None_OK':True},
        'MAG-SAT':  {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'RADECOFF': {'htype':'full', 'dtype':float, 'DB':True,  'None_OK':True},
        'RAOFF':    {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'DECOFF':   {'htype':'full', 'dtype':float, 'DB':False, 'None_OK':True},
        'FORMAT-P': {'htype':'full', 'dtype':bool,  'DB':False, 'None_OK':True},
        'DUMCAT':   {'htype':'full', 'dtype':bool,  'DB':True,  'None_OK':False},
        'QC-FLAG':  {'htype':'full', 'dtype':str,   'DB':True,  'None_OK':False},
        'DATEFILE': {'htype':'full', 'dtype':str,   'DB':False, 'None_OK':True},
        #
        # transient header
        'SWARP-P':  {'htype':'trans', 'dtype':bool,  'DB':True,  'None_OK':False},
        'SWARP-V':  {'htype':'trans', 'dtype':str,   'DB':False, 'None_OK':True},
        'Z-REF':    {'htype':'trans', 'dtype':str,   'DB':False, 'None_OK':True},
        'Z-DXYLOC': {'htype':'trans', 'dtype':bool,  'DB':False, 'None_OK':True},
        'Z-DX':     {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-DY':     {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-DXSTD':  {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-DYSTD':  {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-FNRLOC': {'htype':'trans', 'dtype':bool,  'DB':False, 'None_OK':True},
        'Z-FNR':    {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-FNRSTD': {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-FNRERR': {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-P':      {'htype':'trans', 'dtype':bool,  'DB':True,  'None_OK':False},
        'Z-V':      {'htype':'trans', 'dtype':str,   'DB':False, 'None_OK':True},
        'Z-SIZE':   {'htype':'trans', 'dtype':int,   'DB':False, 'None_OK':True},
        'Z-BSIZE':  {'htype':'trans', 'dtype':int,   'DB':False, 'None_OK':True},
        'Z-SCMED':  {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-SCSTD':  {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'Z-FPEMED': {'htype':'trans', 'dtype':float, 'DB':False, 'None_OK':True},
        'Z-FPESTD': {'htype':'trans', 'dtype':float, 'DB':False, 'None_OK':True},
        'T-NSIGMA': {'htype':'trans', 'dtype':int,   'DB':True,  'None_OK':True},
        'T-LFLUX':  {'htype':'trans', 'dtype':float, 'DB':False, 'None_OK':True},
        'T-NTRANS': {'htype':'trans', 'dtype':int,   'DB':True,  'None_OK':True},
        'T-FTRANS': {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-LFNU':   {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-LMAG':   {'htype':'trans', 'dtype':float, 'DB':True,  'None_OK':True},
        'T-NFAKE':  {'htype':'trans', 'dtype':int,   'DB':False, 'None_OK':True},
        'T-FAKESN': {'htype':'trans', 'dtype':float, 'DB':False, 'None_OK':True},
        'MC-P':     {'htype':'trans', 'dtype':bool,  'DB':True,  'None_OK':False},
        'MC-V':     {'htype':'trans', 'dtype':str,   'DB':False, 'None_OK':True},
        'MC-MODEL': {'htype':'trans', 'dtype':str,   'DB':False, 'None_OK':True},
        'TDUMCAT':  {'htype':'trans', 'dtype':bool,  'DB':True,  'None_OK':False},
        'TQC-FLAG': {'htype':'trans', 'dtype':str,   'DB':True,  'None_OK':False},
    }

    # read header of filename
    if isfile (filename):
        header = read_hdulist (filename, get_data=False, get_header=True)
    else:
        # return success=False if it does not exist
        log.warning ('file {} does not exist; not able to verify its header'
                     .format(filename))
        return False


    # force [htypes] to be a list
    htypes_list = list(htypes)

    # loop keys in dict_head
    for key in dict_head.keys():

        # only check keywords with htype matching the input [htypes]
        if dict_head[key]['htype'] not in htypes_list:
            continue

        # check that key is present in header
        if key in header:

            # provide warning if dtype not as expected and header
            # keyword value is not 'None'
            if (dict_head[key]['dtype'] != type(header[key]) and
                header[key] != 'None'):
                log.warning ('dtype of keyword {}: {} does not match the '
                             'expected dtype: {} in header of {}'
                             .format(key, type(header[key]),
                                     dict_head[key]['dtype'], filename))

            # if key goes to DataBase and value is 'None' or None
            # while 'None_OK' is False, raise an exception
            if (dict_head[key]['DB'] and not dict_head[key]['None_OK'] and
                (header[key] is None or header[key] == 'None')):
                msg = ('DataBase keyword {} not allowed to have \'None\' or '
                       'None value in header of {}'.format(key, filename))
                log.error (msg)
                raise ValueError (msg)

        else:
            msg = 'keyword {} not present in header of {}'.format(key, filename)
            # if keyword will be ingested into the database, raise an exception
            if dict_head[key]['DB']:
                log.error (msg)
                raise KeyError (msg)

            else:
                log.warning (msg)


    return


################################################################################

def call_match2SSO(filename, tel):

    """Function to call the match2SSO software in order to find known asteroids
       in the transient catalog specified in [filename].

    """

    # select light version of catalogue if it exists
    fits_for_m2sso = filename.replace('.fits', '_light.fits')
    if not isfile(fits_for_m2sso):
        fits_for_m2sso = filename

    # run match2SSO on catalogue
    if isfile(fits_for_m2sso):
        m2sso.run_match2SSO(tel=tel, mode='night', cat2process=fits_for_m2sso,
                            date2process=None, list2process=None, logname=None,
                            overwrite=False)
    return


################################################################################

def update_cathead (filename, header):

    """obsolete: replaced by zogy.update_imcathead"""


    if False:
        # CHECK!!! temporarily making copy of transient
        # catalog before header is updated
        shutil.copy2(filename, filename.replace('.fits', '_tmpcopy.fits'))
        # also save fits header
        hdulist = fits.HDUList(fits.PrimaryHDU(header=header))
        hdulist.writeto(filename.replace('.fits', '_tmphdr.fits'))


    use_fitsio = False
    if use_fitsio:
        # played with fitsio to update headers in an attempt to use
        # less disk space: astropy appears to make a copy when
        # updating the file. However, the fitsio headers are slightly
        # different from before, and would need to update all of the
        # headers, also in zogy.py, which is too much hassle for the
        # advantage, at least for now
        with fitsio.FITS(filename, 'rw') as hdulist:
            hdulist[-1].write_keys(dict(header))

        # read updated header - used below - with astropy
        header_update = read_hdulist(filename, get_data=False, get_header=True)

    else:

        with fits.open(filename, 'update', memmap=True) as hdulist:
            # if existing header is minimal (practically only table
            # columns), which can be the case if
            # zogy.optimal_subtraction did not reach the end (due to
            # an issue in the transient extraction) and QC-FLAG is not
            # red (in which case a dummy catalog with header would
            # have been created before this function is called), then
            # add the full input [header] to it; FORMAT-P is only in
            # header if end of zogy.optimal_subtraction was reached
            if 'FORMAT-P' not in hdulist[-1].header:
                hdulist[-1].header += header
            else:
                for key in header:
                    if ('QC' in key or 'DUMCAT' in key or 'RAOFF' in key or
                        'DECOFF' in key):
                        hdulist[-1].header[key] = (header[key],
                                                   header.comments[key])


            header_update = hdulist[-1].header


    # in case of transient catalog, also update the trans_light header
    if 'trans' in filename:
        transcat_light = filename.replace('.fits', '_light.fits')
        # check if it exists
        if isfile(transcat_light):
            with fits.open(transcat_light, 'update', memmap=True) as hdulist:
                # N.B.: cannot simply copy updated header above
                # (header_update) as that will also define the
                # thumbnail colmns while these are not present in the
                # light version
                #hdulist[-1].header = header_update
                for key in header:
                    if ('QC' in key or 'DUMCAT' in key or 'RAOFF' in key or
                        'DECOFF' in key):
                        hdulist[-1].header[key] = (header[key],
                                                   header.comments[key])

        else:
            log.warning ('file {} does not exist'.format(transcat_light))


    # create separate header file with updated header
    hdulist = fits.HDUList(fits.PrimaryHDU(header=header_update))
    hdulist.writeto(filename.replace('.fits', '_hdr.fits'), overwrite=True,
                    output_verify='ignore')


    mem_use (label='in update_cathead')

    return


################################################################################

def update_imhead (filename, header, create_hdrfile=True):

    """obsolete: replaced by zogy.update_imcathead"""

    # update image header with extended header from ZOGY's
    # optimal_subtraction

    header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')

    if '.fz' not in filename:

        with fits.open(filename, 'update', memmap=True) as hdulist:
            hdulist[-1].header = header

    else:

        # for an fpacked image, above update is not possible; use
        # function [copy_header] instead
        copy_header (filename, header)


    if create_hdrfile:
        # create separate header file
        hdulist = fits.HDUList(fits.PrimaryHDU(header=header))
        hdulist.writeto(filename.replace('.fz','').replace('.fits', '_hdr.fits'),
                        overwrite=True, output_verify='ignore')


    mem_use (label='in update_imhead')
    return


################################################################################

def copy_header (fits_dest, header_src):

    """obsolete: replaced by zogy.update_imcathead"""

    """function to copy header of [fits_src] to [fits_dest]; all keywords
       starting from [key_start] are first removed from the header of
       [fits_dest], and then the corresponding keywords are copied
       over from the header of [fits_src].

       Mainly meant to replace the header of an fpacked fits image
       using the header of its funpacked counterpart that was updated,
       without touching its data - if the data was left
       unchanged. This is to avoid unnecessary repeated funpacking and
       fpacking images, leading to loss of precision.

    """

    # open fits_dest for updating
    with fits.open(fits_dest, mode='update') as hdulist:
        header_dest = hdulist[-1].header

        # delete hdr_dest keys
        process_keys (header_dest)

        # copy keys
        process_keys (header_dest, header_src)


    return


################################################################################

def process_keys (hdr_dest, hdr_src=None, key_start='BUNIT'):

    """obsolete: replaced by zogy.update_imcathead"""

    # if hdr_src is not defined, loop through hdr_dest to delete the
    # keywords; if it is, loop through the hdr_src keys to copy them
    # to hdr_dest
    if hdr_src is None:
        hdr = hdr_dest
    else:
        hdr = hdr_src

    process = False
    for key in list(hdr.keys()):

        if key_start in key:
            # start deleting/copying
            process = True

        if process:
            if hdr_src is None:
                # delete key; need to check if it exists, as e.g. the
                # key HISTORY has multiple entries and may have all
                # been deleted already by a single del command
                if key in list(hdr.keys()):
                    del hdr_dest[key]
            else:
                # copy from hdr_src to hdr_dest
                try:
                    hdr_dest.append((key, hdr_src[key], hdr_src.comments[key]))
                except Exception as e:
                    log.error ('failed to copy key {} due to ValueError: {}'
                               .format(key, e))


    return


################################################################################

def create_obslog (date, email=True, tel=None, weather_screenshot=True):

    # extract table with various observables/keywords from the headers
    # of all raw/reduced files of a particular (evening) date,
    # e.g. ORIGFILE, IMAGETYP, DATE-OBS, PROGNAME, PROGID, OBJECT,
    # FILTER, EXPTIME, RA, DEC, AIRMASS, FOCUSPOS, image quality
    # (PSF-FWHM), QC-FLAG, ..
    #
    # if email==True, also send an email to people that are
    # interested; the email parameters such as sender, recipients,
    # etc., can be defined in BlackBOX settings file


    date_eve = ''.join(e for e in date if e.isdigit())
    if len(date_eve) != 8:
        log.error ('input date to function create_obslog needs to consist of '
                   'at least 8 digits, yyyymmdd, where the year, month and '
                   'day can be connected with any type of character, e.g. '
                   'yyyy/mm/dd or yyyy-mm-dd, etc.')
        return


    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])
    red_path = get_par(set_bb.red_dir,tel)
    full_path = '{}/{}'.format(red_path, date_dir)

    log.info ('full_path: {}'.format(full_path))


    # collect biases, darks, flats and science frames in different
    # lists, using [list_files] as little as possible
    log.info ('collecting lists of reduced bias, dark, flat and science frames')
    all_files_list = list_files('{}'.format(full_path), search_str='.fits',
                                recursive=True)
    bias_list = [fn for fn in all_files_list if '/bias/' in fn]
    dark_list = [fn for fn in all_files_list if '/dark/' in fn]
    flat_list = [fn for fn in all_files_list if '/flat/' in fn]
    object_list = [fn for fn in all_files_list if '_red.fits' in fn]


    filenames = [bias_list, dark_list, flat_list, object_list]
    # clean up [filenames]
    filenames = [f for sublist in filenames for f in sublist]


    # number of different reduced files
    nred = len(filenames)
    nbias_red = len(bias_list)
    ndark_red = len(dark_list)
    nflat_red = len(flat_list)
    nobject_red = len(object_list)


    # collect raw image list
    raw_path = get_par(set_bb.raw_dir,tel)
    #raw_list = glob.glob('{}/{}/*.fits*'.format(raw_path, date_dir))
    log.info ('collecting list of raw frames')
    raw_list = list_files('{}/{}'.format(raw_path, date_dir), search_str='.fits')

    # number of different raw files
    nraw = len(raw_list)
    nbias_raw = 0
    ndark_raw = 0
    nflat_raw = 0
    nobject_raw = 0
    for f in raw_list:
        if 'bias' in f.lower():
            nbias_raw += 1
        elif 'dark' in f.lower():
            ndark_raw += 1
        elif 'flat' in f.lower():
            nflat_raw += 1
        elif 'singleobservation' not in f.lower():
            nobject_raw += 1


    # maximum filename length for column format
    #max_length = max([len(f.strip()) for f in filenames])

    # keywords to add to table
    keys = ['ORIGFILE', 'IMAGETYP', 'DATE-OBS', 'PROGNAME', 'PROGID', 'OBJECT',
            'FILTER', 'EXPTIME', 'RA', 'DEC', 'AIRMASS', 'FOCUSPOS',
            'S-SEEING', 'CL-BASE', 'RH-MAST', 'WINDAVE', 'LIMMAG', 'QC-FLAG',
            'QCRED1', 'QCRED2', 'QCRED3']
    formats = {#'ORIGFILE': '{:60}',
        #'IMAGETYP': '{:<8}',
        'DATE-OBS': '{:.19}',
        'EXPTIME': '{:.1f}',
        'RA': '{:.3f}',
        'DEC': '{:.3f}',
        'AIRMASS': '{:.3f}',
        'S-SEEING': '{:.4}',
        'CL-BASE': '{:.4}',
        'RH-MAST': '{:.4}',
        'WINDAVE': '{:.4}',
        'LIMMAG': '{:.5}'
    }


    # loop input list of filenames
    log.info ('reading headers of reduced frames')
    rows = []
    for filename in filenames:

        fn_hdr = filename.replace('.fits.fz', '_hdr.fits')
        if isfile (fn_hdr):
            file2read = fn_hdr
        else:
            file2read = filename


        # read file header
        header = read_hdulist (file2read, get_data=False, get_header=True)

        # prepare row of filename and header values
        row = []
        for key in keys:
            if key in header:
                row.append(header[key])
            else:
                row.append(' ')

        # append to rows
        rows.append(row)


    # create table from rows
    names = []
    for i_key, key in enumerate(keys):
        names.append(key)

    if len(rows) == 0:
        # rows without entries: create empty table
        table = Table(names=names)
    else:
        table = Table(rows=rows, names=names)

    # order by DATE-OBS
    index_sort = np.argsort(table['DATE-OBS'])
    table = table[index_sort]



    # write table to ASCII file in tmp folder
    tmp_path = get_par(set_bb.tmp_dir,tel)
    # just in case it does not exist yet, create it
    make_dir (tmp_path)
    obslog_tmp = '{}/{}_{}_obslog.txt'.format(tmp_path, tel, date_eve)
    log.info ('saving header info to {}'.format(obslog_tmp))


    # if table is empty, no files were processed and reduced folder
    # needs to be created before writing the empty table
    if len(rows)==0:
        red_dir = '{}/{}'.format(red_path, date_dir)
        make_dir (red_dir)
        ascii.write (table, obslog_tmp, overwrite=True)
    else:
        # write the filled table
        ascii.write (table, obslog_tmp, format='fixed_width_two_line',
                     delimiter_pad=' ', position_char=' ',
                     formats=formats, overwrite=True)



    # for MeerLICHT and BlackGEM, save the weather page as a screen
    # shot in tmp folder, and add it as attachment to the mail
    if tel=='ML1':
        png_tmp = '{}/{}_SAAOweather.png'.format(tmp_path, date_eve)
        webpage = 'https://suthweather.saao.ac.za'
        width = 1500
        height = 2150
    else:
        png_tmp = '{}/{}_LaSilla_meteo.png'.format(tmp_path, date_eve)
        #webpage = 'https://www.ls.eso.org/lasilla/dimm/meteomonitor.html'
        #webpage = 'https://archive.eso.org/asm/ambient-server?site=lasilla'
        # date in format e.g. 28+Oct+2024
        date_tmp = '{}+{}+{}'.format(date_eve[6:8],
                                     calendar.month_abbr[int(date[4:6])],
                                     date[0:4])
        webpage = ('https://archive.eso.org/asm/ambient-server?'
                   'night={}&site=lasilla'.format(date_tmp))
        # the following line is only returning a white stripe, perhaps
        # only the very top of the webpage?
        #webpage = ('https://www.eso.org/asm/ui/publicLog?name=LaSilla&startDate='
        #           '{}'.format(date_eve))
        width = 1500
        height = 1150


    # define png destination
    png_dest = '{}/{}/{}'.format(red_path, date_dir, png_tmp.split('/')[-1])


    png_present = False
    if weather_screenshot:
        try:

            log.info ('saving screenshot of {} to {}'.format(webpage, png_tmp))
            cmd = ['wkhtmltoimage', '--quiet', '--quality', '80',
                   '--crop-w', str(width), '--crop-h', str(height),
                   webpage, png_tmp]
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            #log.info('stdout: {}'.format(result.stdout.decode('UTF-8')))
            #log.info('stderr: {}'.format(result.stderr.decode('UTF-8')))

        except Exception as e:
            log.exception ('exception occurred while making screenshot of '
                           'weather page {}: {}'.format(webpage, e))
            png_tmp = None
    else:
        # check if screenshot already exists
        if isfile(png_dest):
            copy_file (png_dest, png_tmp, move=False)
            png_present = True
        else:
            # do not include screenshot in email
            png_tmp = None


    # additional info that could be added to body of email
    # - any raw files that were not reduced?
    # - list any observing gaps, in fractional UT hours
    # - using above gaps, list fraction of night that telescope was observing
    # - list average exposure overhead in seconds

    body  = '{}: summary of {} observations:\n'.format(tel,
                                                       date_dir.replace('/','-'))
    body += '----------------------------------------\n'

    body += ('# raw images:       {} ({} biases, {} darks, {} flats, {} objects)'
             '\n'.format(nraw, nbias_raw, ndark_raw, nflat_raw, nobject_raw))
    body += ('# reduced images:   {} ({} biases, {} darks, {} flats, {} objects)'
             '\n'.format(nred, nbias_red, ndark_red, nflat_red, nobject_red))


    # collect full-source, transient and sso catalog lists, using
    # [list_files] as little as possible
    all_cats_list = list_files('{}/{}'.format(full_path, tel), end_str='.fits')

    log.info ('collecting lists of full-source, transient and sso catalogs, and '
              'counting how many of them are flagged red')
    cat_list = [c for c in all_cats_list if c.endswith('_red_cat.fits')]
    body += ('# full-source cats: {} ({} red-flagged)\n'.format(
        len(cat_list), count_redflags(cat_list)))

    trans_list = [c for c in all_cats_list if c.endswith('_red_trans.fits')]
    body += ('# transient cats:   {} ({} red-flagged)\n'.format(
        len(trans_list), count_redflags(trans_list, key='TQC-FLAG')))

    sso_list = [c for c in all_cats_list if c.endswith('_red_trans_sso.fits')]
    body += ('# SSO cats:         {} ({} empty)\n'.format(
        len(sso_list), count_redflags(sso_list, key='SDUMCAT')))
    body += '\n'


    # link to weather page
    body += 'Observing conditions: {}\n'.format(webpage)


    if email:
        # email the obslog (with the weather page for MeerLICHT as
        # attachment) to a list of interested people
        try:
            # subject
            recipients = get_par(set_bb.recipients,tel)
            subject = '{} night report {}'.format(tel, date_dir.replace('/','-'))
            sender = get_par(set_bb.sender,tel)
            reply_to = get_par(set_bb.reply_to,tel)
            smtp_server = get_par(set_bb.smtp_server,tel)
            port = get_par(set_bb.port,tel)
            use_SSL = get_par(set_bb.use_SSL,tel)

            log.info ('sending email with subject {} to {} using smtp server {} '
                      'on port {}'
                      .format(subject, recipients, smtp_server, port))

            send_email (recipients, subject, body,
                        attachments='{},{}'.format(obslog_tmp, png_tmp),
                        sender=sender, reply_to=reply_to,
                        smtp_server=smtp_server, port=port, use_SSL=use_SSL)

        except Exception as e:
            log.exception('exception occurred during sending of email: {}'
                          .format(e))


    # now that email is sent, move obslog and weather screenshot
    obslog_dest = '{}/{}/{}_{}_obslog.txt'.format(red_path, date_dir,
                                                  tel, date_eve)
    copy_file (obslog_tmp, obslog_dest, move=True)


    if png_tmp and not png_present:
        copy_file (png_tmp, png_dest, move=True)



    return


################################################################################

def count_redflags(catlist, key='QC-FLAG'):

    nredflags = 0

    for catname in catlist:

        fn_hdr = catname.replace('.fits', '_hdr.fits')
        if isfile (fn_hdr):
            file2read = fn_hdr
        else:
            file2read = catname


        # read file header
        header = read_hdulist (file2read, get_data=False, get_header=True)

        # in case of full-source or transient catalog, 'red' should be
        # in the QC-FLAG keyword, but for SSO cat, SDUMCAT being True
        # indicates a red flag; "==True" is required because if the
        # keyword is not a boolean, header[key] will always be True
        if key in header and ('red' in str(header[key]) or header[key]==True):
            nredflags += 1


    return nredflags


################################################################################

def send_email (recipients, subject, body,
                attachments=None,
                sender='Radboud GW Alert <scheduler@blackgem.org>',
                reply_to='p.vreeswijk@astro.ru.nl',
                smtp_server='smtp-relay.gmail.com',
                port=465, use_SSL=True):

    if use_SSL:
        smtpObj = smtplib.SMTP_SSL(smtp_server, port)
    else:
        smtpObj = smtplib.SMTP(smtp_server, port)

    smtpObj.ehlo()
    send_from = sender
    send_to = recipients.split(',')
    msg = MIMEMultipart()
    msg['from'] = send_from
    msg['to'] = recipients
    msg['reply-to'] = reply_to
    msg['date'] = formatdate(localtime=True)
    msg['subject'] = subject

    if body is None:
        text = ''
    elif isfile(body):
        with open(body, 'r') as f:
            text = f.read()
    else:
        text = body

    msg.attach( MIMEText(text) )

    if attachments is not None:
        att_list = attachments.split(',')
        for attachment in att_list:
            if isfile(attachment):
                part = MIMEBase('application', "octet-stream")
                part.set_payload( open(attachment,"rb").read() )
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment; filename={}'
                                .format(attachment.split('/')[-1]))
                msg.attach(part)

    smtpObj.sendmail(send_from, send_to, msg.as_string())
    smtpObj.close()


################################################################################

def get_flatstats (data, header, data_mask, tel=None):


    if get_par(set_zogy.timing,tel):
        t = time.time()

    # mask of valid pixels
    mask_use = (data_mask == 0)


    # add some header keywords with the statistics
    sec_temp = get_par(set_bb.flat_norm_sec,tel)
    value_temp = '[{}:{},{}:{}]'.format(
        sec_temp[0].start+1, sec_temp[0].stop+1,
        sec_temp[1].start+1, sec_temp[1].stop+1)
    header['STATSEC'] = (
        value_temp, 'pre-defined statistics section [y1:y2,x1:x2]')


    # statistics on STATSEC
    mask_use_temp = mask_use[sec_temp]
    median_sec = np.nanmedian(data[sec_temp][mask_use_temp])
    std_sec = np.nanstd(data[sec_temp][mask_use_temp])
    # using masked array (slow!)
    #median_sec = np.ma.median(data_masked[sec_temp])
    #std_sec = np.ma.std(data_masked[sec_temp])

    header['MEDSEC'] = (median_sec, '[e-] median flat over STATSEC')
    header['STDSEC'] = (std_sec, '[e-] sigma (STD) flat over STATSEC')
    header['RSTDSEC'] = (std_sec/median_sec, 'relative sigma (STD) flat '
                         'over STATSEC')


    # full image statistics
    index_stat = get_rand_indices(data.shape)
    mask_use_temp = mask_use[index_stat]
    median = np.nanmedian(data[index_stat][mask_use_temp])
    std = np.nanstd(data[index_stat][mask_use_temp])
    # masked array (slow!)
    #median = np.ma.median(data_masked[index_stat])
    #std = np.ma.std(data_masked[index_stat])

    header['FLATMED'] = (median, '[e-] median flat')
    header['FLATSTD'] = (std, '[e-] sigma (STD) flat')
    header['FLATRSTD'] = (std/median, 'relative sigma (STD) flat')

    # add the channel median level to the flatfield header
    chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red = (
        define_sections(np.shape(data), tel=tel))
    nchans = np.shape(data_sec)[0]

    for i_chan in range(nchans):

        median_temp = np.nanmedian(data[data_sec_red[i_chan]])
        header['FLATM{}'.format(i_chan+1)] = (
            median_temp,
            '[e-] channel {} median flat (bias-subtracted)'.format(i_chan+1))

        std_temp = np.nanstd(data[data_sec_red[i_chan]])
        header['FLATS{}'.format(i_chan+1)] = (
            std_temp,
            '[e-] channel {} sigma (STD) flat'.format(i_chan+1))

        header['FLATRS{}'.format(i_chan+1)] = (
            std_temp/median_temp,
            'channel {} relative sigma (STD) flat'.format(i_chan+1))



    # split image in 8x8 subimages and calculate a few additional
    # statistics; size of subimages is taken to be the same as the
    # ZOGY subimages
    ysize, xsize = data.shape
    subsize = get_par(set_zogy.subimage_size,tel)
    nsubs_side = int(ysize/subsize)

    # create masked array and reshape it
    data_masked_reshaped = np.ma.masked_array(data, mask=~mask_use).reshape(
        nsubs_side,subsize,-1,subsize).swapaxes(1,2).reshape(
            nsubs_side,nsubs_side,-1)

    # get statistics
    index_stat = get_rand_indices((data_masked_reshaped.shape[2],), fraction=0.1)
    mini_median = np.ma.median(data_masked_reshaped[:,:,index_stat], axis=2)
    mini_median_reshaped = mini_median.reshape(nsubs_side, nsubs_side, 1)

    # to avoid adding possible stars to the STD determination,
    # calculate the STD for each subimage only for the values below
    # the median, and do this calculation with respect to the median,
    # i.e.:  STD**2 = sum((data[<median] - median)**2) / (N-1)
    mask_clip = (data_masked_reshaped > mini_median_reshaped)
    # update mask of masked array
    data_masked_reshaped.mask |= mask_clip

    # standard deviation
    #mini_std = np.ma.std(data_masked_reshaped, axis=2)
    # do not bother to work on fraction of pixels (index_stat) as half
    # of the pixels have already been clipped
    mini_std = np.sqrt(
        (np.ma.sum((data_masked_reshaped - mini_median_reshaped)**2, axis=2) /
         (np.ma.count(data_masked_reshaped, axis=2) - 1)))


    # avoid using outer rim of subimages in RDIF-MAX and RSTD-MAX
    # statistics; this mask discards those subimages
    mask_cntr = ndimage.binary_erosion(np.ones(mini_median.shape, dtype=bool))

    # statistic used by Danielle, or the maximum relative difference
    # between the boxes' medians
    minimum = np.amin(mini_median[mask_cntr])
    maximum = np.amax(mini_median[mask_cntr])
    danstat = np.abs((maximum - minimum) / (maximum + minimum))
    if not np.isfinite(danstat):
        danstat = 'None'

    header['NSUBSTOT'] = (mask_cntr.size, 'number of subimages available for statistics')
    header['NSUBS'] = (np.sum(mask_cntr), 'number of subimages used for statistics')
    header['RDIF-MAX'] = (danstat, '(max(subs)-min(subs)) / (max(subs)+min(subs))')

    mask_nonzero = (mini_median[mask_cntr] != 0)
    if np.sum(mask_nonzero) != 0:
        rstd_max = np.amax(mini_std[mask_cntr][mask_nonzero] /
                           np.abs(mini_median[mask_cntr][mask_nonzero]))
    else:
        rstd_max = 'None'

    header['RSTD-MAX'] = (rstd_max, 'max. relative sigma (STD) of subimages')


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_flatstats')

    return


################################################################################

def close_log (log, logfile):

    handlers = log.handlers[:]
    for handler in handlers:
        if logfile in str(handler):
            log.info('removing handler {} from log'.format(handler))
            log.removeHandler(handler)
            handler.close()


    # remove the last handler, which is assumed to be the filehandler
    # added inside blackbox_reduce
    #log.removeHandler(log.handlers[-1])

    return


################################################################################

def make_dir (path, empty=False):

    """Function to make directory. If [empty] is True and the
       directory already exists, it will first be removed. In case of
       google cloud version, don't do anything - there are no
       actual directories in a bucket."""


    # check if google_cloud is set
    if not path[0:5] == 'gs://':

        # if already exists but needs to be empty, remove it first
        if isdir(path) and empty:
            shutil.rmtree(path, ignore_errors=True)

        # do not check if directory exists, just try to make it; changed this
        # after racing condition occurred on the ilifu Slurm cluster when
        # reducing flatfields, where different tasks need to make the same
        # directory
        os.makedirs(path, exist_ok=True)


    return


################################################################################

def clean_tmp (tmp_path, keep_tmp):

    """ Function that removes the tmp folder corresponding to the
        reduced image / reference image if [set_bb.keep_tmp] not True.
    """

    # check if folder exists
    if isdir(tmp_path):

        # delete [tmp_path] folder if [set_bb.keep_tmp] not True
        if not keep_tmp:
            #log.info ('removing temporary folder: {}'.format(tmp_path))
            shutil.rmtree(tmp_path, ignore_errors=True)

        else:
            # otherwise fpack its fits images
            #list_2pack = glob.glob('{}/*.fits'.format(tmp_path))
            list_2pack = list_files(tmp_path, end_str='.fits')

            for filename in list_2pack:
                fpack (filename)

    else:
        log.warning ('tmp folder {} does not exist'.format(tmp_path))


    return


################################################################################

def copy_files2keep (src_base, dest_base, ext2keep, move=True, run_fpack=True):

    """Function to copy/move files with base name [src_base] and
    extensions [ext2keep] to files with base name [dest_base] with the
    same extensions. The base names should include the full path.
    """

    # make copy of [ext2keep] to avoid modifying the input parameter
    ext2keep_copy = np.copy(ext2keep)

    # select unique entries in input [ext2keep]
    ext2keep_uniq = list(set(ext2keep_copy))
    log.info ('extensions to copy: {}'.format(ext2keep_uniq))

    # list of all files starting with [src_base]
    #src_files = glob.glob('{}*'.format(src_base))
    src_files = list_files(src_base)


    # loop this list
    for src_file in src_files:
        # determine file string following [src_base]
        src_ext = src_file.split(src_base)[-1]
        # check if this matches entry in [ext2keep_uniq]
        for ext in ext2keep_uniq:
            if ext in src_ext:
                dest_file = '{}{}'.format(dest_base, src_ext)
                # if so, and the source and destination names are not
                # identical, go ahead and copy
                if src_file != dest_file:


                    # if the data of an unpacked src_file image is the
                    # same as the data of the fpacked dest_file image,
                    # then only modify the destination file header
                    # with that of the src_file to avoid unnecessary
                    # funpacking and fpacking
                    skip = False
                    dest_file_fz = '{}.fz'.format(dest_file)
                    if (src_file.split('.')[-1] == 'fits'
                        and '_ldac.fits' not in src_file
                        and isfile(dest_file_fz)
                        # in Google cloud, do not execute the block
                        # below: not possible to only update the
                        # header of a file in a bucket
                        and dest_file[0:5] != 'gs://'):


                        # read src_file data and header
                        data_src, header_src = read_hdulist(src_file,
                                                            get_header=True)

                        # check if src_file is an image
                        if int(header_src['NAXIS'])==2:

                            # read dest_file data
                            data_dest = read_hdulist(dest_file_fz)

                            # compare data arrays; this comparison is
                            # done before fpacking src_file, so data
                            # arrays should be very close if
                            # destination file was already fully
                            # reduced; still, allow for absolute
                            # difference of 1e- between the data sets
                            if np.allclose(data_src, data_dest, atol=1):

                                # skip copying/moving below
                                skip = True

                                log.info ('existing image {} contains same data '
                                          'as {}; skipping copy/move'
                                          .format(dest_file_fz, src_file))

                                exts_keephead = [
                                    '_red.fits', '_red_limmag.fits', '_D.fits',
                                    '_Scorr.fits', '_trans_limmag.fits',
                                    '_Fpsf.fits']

                                for ext_tmp in exts_keephead:

                                    if ext_tmp in dest_file:
                                        # for various images, update
                                        # header of already existing
                                        # destination file with that
                                        # of src_file; header file
                                        # does not need to be updated
                                        # - already done if properly
                                        # processed by zogy
                                        log.info ('updating fits header of {}'
                                                  .format(dest_file_fz))
                                        update_imcathead (dest_file_fz, header_src)

                            else:
                                log.info ('data of existing image {} is '
                                          'significantly different from that of '
                                          '{}'.format(dest_file_fz, src_file))


                    if run_fpack:
                        # fpack src_file if needed
                        src_file = fpack (src_file)

                        # add '.fz' extension to [dest_file] in case
                        # [src_file] was fpacked (not all files are
                        # fpacked)
                        if '.fz' in src_file and '.fz' not in dest_file:
                            dest_file = '{}.fz'.format(dest_file)

                        # create a jpg image of [src_file] now that
                        # files are being copied from tmp to red or
                        # ref and not vice versa, and file is reduced,
                        # D or Scorr image
                        if ('_red.fits' in src_file or '_D.fits' in src_file or
                            '_Scorr.fits' in src_file):
                            src_file_jpg = create_jpg (src_file)
                            # copy/move it over to the destination folder
                            #dest_folder = os.path.dirname(dest_file)
                            #copy_file (src_file_jpg, dest_folder+'/', move=move)
                            dest_file_jpg = '{}.jpg'.format(
                                dest_file.split('.fits')[0])
                            copy_file (src_file_jpg, dest_file_jpg, move=move)



                    # remove the potentially existing f/unpacked
                    # counterparts of [dest_file] already present in
                    # the destination folder for some reason, to avoid
                    # that both the unpacked and packed file will be
                    # present in the destination folder
                    if '.fz' in dest_file:
                        file_2remove = dest_file.split('.fz')[0]
                    else:
                        file_2remove = '{}.fz'.format(dest_file)

                    if isfile(file_2remove):
                        #log.info('removing existing {}'.format(file_2remove))
                        #os.remove(file_2remove)
                        remove_files ([file_2remove], verbose=True)


                    # move or copy file if it does not need to be skipped
                    if not skip:
                        if 'log' in src_file:
                            # copy logfile as it is still being used
                            copy_file (src_file, dest_file, move=False)
                        else:
                            copy_file (src_file, dest_file, move=move)



    return


################################################################################

def run_asta (data, header, data_mask, header_mask, tmp_path):

    if get_par(set_zogy.timing,tel):
        t = time.time()


    # bin input data if nbin > 1
    nbin = get_par(set_bb.sat_bin,tel)
    if nbin == 1:
        data_binned = data
    else:
        data_binned = data.reshape(np.shape(data)[0] // nbin, nbin,
                                   np.shape(data)[1] // nbin, nbin).sum(3).sum(1)


    #write data to tmp file; not needed anymore since Fiore changed
    # input to ASTA function from fits image to array
    #fits_tmp = ('{}/{}'.format(
    #    tmp_path, tmp_path.split('/')[-1].replace('_red', '_red_asta.fits')))
    #fits.writeto(fits_tmp, data, header, overwrite=True)


    asta_model = get_par(set_bb.asta_model,tel)
    processor = ASTA(asta_model)
    #mask_sat, __, __ = processor.process_image(fits_tmp)
    mask_sat_binned, results_df, __, __ = processor.process_image(
        data_binned, header, area_threshold=int(3000/nbin**2),
        min_size=int(500/nbin**2))

    #unbin mask
    if nbin == 1:
        mask_sat = mask_sat_binned
    else:
        mask_sat = np.kron(
            mask_sat_binned, np.ones((nbin, nbin))).astype(np.uint8)


    # add pixels affected by satellite trails to [data_mask]
    data_mask[mask_sat==1] |= get_par(set_zogy.mask_value['satellite trail'],
                                      tel)
    #nsatpixels = np.sum(mask_sat)


    # determining number of trails; 2 pixels are considered from the
    # same trail also if they are only connected diagonally
    if False:
        struct = np.ones((3,3), dtype=bool)
        __, nsats = ndimage.label(mask_sat, structure=struct)
    else:
        # alternatively, just count number of rows in results_df
        nsats = len(results_df)


    header['NSATS'] = (nsats, 'number of satellite trails identified')
    header_mask['NSATS'] = (nsats, 'number of satellite trails identified')

    log.info('number of satellite trails identified: {}'.format(nsats))


    # remove file(s) if not keeping intermediate/temporary files
    #if not get_par(set_bb.keep_tmp,tel):
    #    remove_files ([fits_tmp])


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in run_asta')


    return data_mask


################################################################################

def sat_detect (data, header, data_mask, header_mask, tmp_path, nbin=2):

    # could also try skimage.transform.probabilistic_hough_line()

    if get_par(set_zogy.timing,tel):
        t = time.time()

    #bin data
    binned_data = data.reshape(np.shape(data)[0] // nbin, nbin,
                               np.shape(data)[1] // nbin, nbin).sum(3).sum(1)
    satellite_fitting = False

    for j in range(1):
        #write binned data to tmp file
        fits_binned_mask = ('{}/{}'.format(
            tmp_path, tmp_path.split('/')[-1].replace('_red',
                                                      '_binned_satmask.fits')))
        fits.writeto(fits_binned_mask, binned_data, overwrite=True)
        #detect satellite trails
        try:
            results, errors = detsat(fits_binned_mask, chips=[0],
                                     n_processes=1, buf=40, sigma=3,
                                     h_thresh=0.2, plot=False, verbose=False)
        except Exception as e:
            log.exception('exception was raised during [detsat]: {}'.format(e))
            # raise exception
            raise RuntimeError ('problem with running detsat module')
        else:
            # also raise exception if detsat module returns errors
            if len(errors) != 0:
                log.error('detsat errors: {}'.format(errors))
                raise RuntimeError ('problem with running detsat module')

        #create satellite trail if found
        trail_coords = results[(fits_binned_mask,0)]
        #continue if satellite trail found
        if len(trail_coords) > 0:
            trail_segment = trail_coords[0]
            try:
                #create satellite trail mask
                mask_binned = make_mask(fits_binned_mask, 0, trail_segment,
                                        sublen=5, pad=0, sigma=5).astype('uint8')
            except ValueError:
                #if error occurs, add comment
                log.exception ('satellite trail found but could not be '
                               'fitted for file {} and is not included in '
                               'the mask'.format(tmp_path.split('/')[-1]))
                break

            satellite_fitting = True
            binned_data[mask_binned == 1] = np.median(binned_data)
            fits_old_mask = '{}/old_mask.fits'.format(tmp_path)
            if isfile(fits_old_mask):
                old_mask = read_hdulist(fits_old_mask)
                mask_binned = old_mask+mask_binned
            fits.writeto(fits_old_mask, mask_binned, overwrite=True)
        else:
            break

    if satellite_fitting == True:
        #unbin mask
        mask_sat = np.kron(
            mask_binned, np.ones((nbin, nbin))).astype(np.uint8)
        # add pixels affected by satellite trails to [data_mask]
        data_mask[mask_sat==1] |= get_par(set_zogy.mask_value['satellite trail'],
                                          tel)
        # determining number of trails; 2 pixels are considered from the
        # same trail also if they are only connected diagonally
        struct = np.ones((3,3), dtype=bool)
        __, nsats = ndimage.label(mask_sat, structure=struct)
        nsatpixels = np.sum(mask_sat)
    else:
        nsats = 0
        nsatpixels = 0

    header['NSATS'] = (nsats, 'number of satellite trails identified')
    header_mask['NSATS'] = (nsats, 'number of satellite trails identified')

    log.info('number of satellite trails identified: {}'.format(nsats))


    # remove file(s) if not keeping intermediate/temporary files
    if not get_par(set_bb.keep_tmp,tel):
        remove_files ([fits_binned_mask])
        if 'fits_old_mask' in locals():
            remove_files ([fits_old_mask])


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in sat_detect')

    return data_mask


################################################################################

def cosmics_corr (data, header, data_mask, header_mask):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    mem_use (label='in cosmics_corr at start')


    # set satlevel to infinite, as input [data_mask] already contains
    # saturated and saturated-connected pixels that will not be considered
    # in the cosmic-ray detection; in fact all masked pixels are excluded
    #satlevel_electrons = (get_par(set_bb.satlevel,tel) *
    #                      np.mean(get_par(set_bb.gain,tel)) - header['BIASMEAN'])
    satlevel_electrons = np.inf


    # determine reduced data sections
    __, __, __, __, data_sec_red = define_sections(np.shape(data), tel=tel)


    if False:

        # when using 1.0.9+ version of astroscrappy:

        # create readnoise image to use
        data_rdnoise2 = np.zeros_like (data)


        # loop channels
        nchans = np.shape(data_sec_red)[0]
        for i_chan in range(nchans):

            # channel section
            sec_tmp = data_sec_red[i_chan]

            rdn_str = 'RDN{}'.format(i_chan+1)
            if rdn_str not in header:
                log.error ('keyword {} expected but not present in header'
                           .format(rdn_str))
            else:
                data_rdnoise2[sec_tmp] = header[rdn_str]**2


        # add Poisson noise
        data_var = data_rdnoise2 + data

        mask_cr, data = astroscrappy.detect_cosmics(
            data, inmask=(data_mask!=0), invar=data_var,
            sigclip=get_par(set_bb.sigclip,tel),
            sigfrac=get_par(set_bb.sigfrac,tel),
            objlim=get_par(set_bb.objlim,tel),
            niter=get_par(set_bb.niter,tel),
            satlevel=satlevel_electrons,
            cleantype='medmask',
            #fsmode='convolve', psfmodel='gauss', psffwhm=4.5, psfsize=7,
            sepmed=get_par(set_bb.sepmed,tel))


    else:

        # when using 1.0.8 version of astroscrappy:

        readnoise = header['RDNOISE']

        mask_cr, data = astroscrappy.detect_cosmics(
            data, inmask=(data_mask!=0),
            sigclip=get_par(set_bb.sigclip,tel),
            sigfrac=get_par(set_bb.sigfrac,tel),
            objlim=get_par(set_bb.objlim,tel),
            niter=get_par(set_bb.niter,tel),
            readnoise=readnoise, gain=1.0,
            satlevel=satlevel_electrons,
            cleantype='medmask',
            sepmed=get_par(set_bb.sepmed,tel))



    mem_use (label='in cosmics_corr just after astroscrappy')

    # from astroscrappy 'manual': To reproduce the most similar
    # behavior to the original LA Cosmic (written in IRAF), set inmask
    # = None, satlevel = np.inf, sepmed=False, cleantype='medmask',
    # and fsmode='median'.
    #mask_cr, data = astroscrappy.detect_cosmics(
    #    data, inmask=None, sigclip=get_par(set_bb.sigclip,tel),
    #    sigfrac=get_par(set_bb.sigfrac,tel), objlim=get_par(set_bb.objlim,tel),
    #    niter=get_par(set_bb.niter,tel),
    #    readnoise=header['RDNOISE'], satlevel=np.inf)

    # add pixels affected by cosmic rays to [data_mask]
    data_mask[mask_cr==1] |= get_par(set_zogy.mask_value['cosmic ray'],tel)


    # determining number of cosmics; 2 pixels are considered from the
    # same cosmic also if they are only connected diagonally
    struct = np.ones((3,3), dtype=bool)
    __, ncosmics = ndimage.label(mask_cr, structure=struct)
    ncosmics_persec = ncosmics / float(header['EXPTIME'])
    header['NCOSMICS'] = (ncosmics_persec, '[/s] number of cosmic rays identified')
    # also add this to header of mask image
    header_mask['NCOSMICS'] = (ncosmics_persec, '[/s] number of cosmic rays identified')

    log.info('number of cosmic rays identified: {}'.format(ncosmics))


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in cosmics_corr')


    return data, data_mask


################################################################################

def mask_init (data, header, filt, imgtype):

    """Function to create initial mask from the bad pixel mask (defining
       the bad and edge pixels), and pixels that are saturated and
       pixels connected to saturated pixels.

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    fits_bpm = (get_par(set_bb.bad_pixel_mask,tel)
                .replace('bpm', 'bpm_{}'.format(filt)))

    bpm_present, fits_bpm = already_exists (fits_bpm, get_filename=True)
    if bpm_present:
        # if it exists, read it
        data_mask = read_hdulist(fits_bpm, dtype='uint8')
        log.info ('using bad pixel mask {}'.format(fits_bpm))
    else:
        # if not, create uint8 array of zeros with same shape as
        # [data]
        log.warning ('bad pixel mask {} does not exist'.format(fits_bpm))
        data_mask = np.zeros(np.shape(data), dtype='uint8')


    # create initial mask header
    header_mask = fits.Header()


    if imgtype == 'object':

        # mask of pixels with non-finite values in [data]
        mask_infnan = ~np.isfinite(data)
        # replace those pixel values with zeros
        data[mask_infnan] = 0
        # and add them to [data_mask] with same value defined for 'bad' pixels
        # unless that pixel was already masked
        mask_value = get_par(set_zogy.mask_value,tel)
        data_mask[(mask_infnan) & (data_mask==0)] |= mask_value['bad']


        if False:

            # old block to identify saturated pixels, where
            # set_bb.satlevel was an image-average number; saturation
            # level (ADU) is taken from blackbox settings file, which
            # needs to be multiplied by the gain and have the mean
            # biaslevel subtracted
            satlevel_electrons = (np.array(get_par(set_bb.satlevel,tel)) *
                                  np.min(get_par(set_bb.gain,tel))
                                  - header['BIASMEAN'])
            mask_sat = (data >= satlevel_electrons)


            header_mask['SATURATE'] = (satlevel_electrons, '[e-] adopted '
                                       'saturation threshold')
            header['SATURATE'] = (satlevel_electrons, '[e-] adopted '
                                  'saturation threshold')



        # to use starting from August 2024: channel-specific
        # saturation levels

        # reduced data channel sections
        __, __, __, __, data_sec_red = define_sections(np.shape(data),
                                                       tel=tel)
        # number of channels
        nchans = np.shape(data_sec_red)[0]


        # determine saturated pixels for each channel separately
        biaslevel_chans = np.array([header['BIASM{}'.format(i_chan+1)]
                                    for i_chan in range(nchans)])

        # array of satlevels in e- for different channels
        satlevel_chans = (np.array(get_par(set_bb.satlevel,tel)) *
                          np.array(get_par(set_bb.gain,tel))
                          - biaslevel_chans)


        # add mean saturation level to both header and header_mask;
        # keep name SATURATE for the mean saturation level as that is
        # also used in buildref
        satlevel_mean = np.mean(satlevel_chans)
        header_mask['SATURATE'] = (satlevel_mean, '[e-] mean saturation '
                                   'threshold')
        header['SATURATE'] = (satlevel_mean, '[e-] mean saturation threshold')


        # initialize full-image mask of saturated pixels, needed
        # also for further down below to identify
        # saturation-connected pixels
        mask_sat = np.zeros_like(data, dtype=bool)


        # loop channels
        for i_chan in range(nchans):

            # channel saturation level
            satlevel_chan = satlevel_chans[i_chan]


            # add channel saturation level to both header and
            # header_mask
            key = 'SATLEV{}'.format(i_chan+1)
            descr = '[e-] channel {} saturation threshold'.format(i_chan+1)
            header[key] = (satlevel_chan, descr)
            header_mask[key] = (satlevel_chan, descr)


            # based on saturated pixels in current channel, define
            # pixels in other (victim) channels that are most affected
            # by high pixel values in this (source) channel due to
            # crosstalk

            # mask of saturated pixels in current channel
            chan_sec = data_sec_red[i_chan]
            mask_sat_chan = (data[chan_sec] >= satlevel_chan)


            # add to full-image saturation mask
            mask_sat[chan_sec] = mask_sat_chan


            # row of this source channel; if row of source and victim
            # channel are different, mask_sat_chan needs to be flipped
            # in y
            row_source = i_chan // 8


            # flipped version of mask_sat_chan to be used in loop below
            mask_sat_chan_flip = np.flipud(mask_sat_chan)


            # loop victim channels
            for i_victim in range(nchans):

                if i_victim != i_chan:

                    # victim channel image section
                    chan_sec_victim = data_sec_red[i_victim]


                    # row of victim channel
                    row_victim = i_victim // 8

                    if row_source == row_victim:
                        mask_use = mask_sat_chan
                    else:
                        mask_use = mask_sat_chan_flip


                    # add crosstalk pixels to the full-image mask
                    data_mask[chan_sec_victim][mask_use] |= \
                        mask_value['crosstalk']





        # add them to the mask of edge and bad pixels
        data_mask[mask_sat] |= mask_value['saturated']


        # determining number of saturated objects; 2 saturated pixels are
        # considered from the same object also if they are only connected
        # diagonally
        struct = np.ones((3,3), dtype=bool)
        __, nobj_sat = ndimage.label(mask_sat, structure=struct)


        # add number of saturated objects to headers
        header_mask['NOBJ-SAT'] = (nobj_sat, 'number of saturated objects')
        header['NOBJ-SAT'] = (nobj_sat, 'number of saturated objects')
        # rest of the mask header entries are added in one go using
        # function [mask_header] once all the reduction steps have
        # finished


        # identify pixels connected to saturated pixels
        struct = np.ones((3,3), dtype=bool)
        mask_satcon = ndimage.binary_dilation(mask_sat, structure=struct,
                                              iterations=1)
        # add them to the mask
        mask_satcon2add = (mask_satcon & ~mask_sat)
        data_mask[mask_satcon2add] |= mask_value['saturated-connected']


        # fill potential holes using function [fill_sat_holes]
        fill_sat_holes (data_mask, mask_value)




    #fits.writeto ('test_data.fits', data, overwrite=True)
    #fits.writeto ('test_mask2.fits', data_mask, overwrite=True)
    #raise SystemExit


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in mask_init')

    return data_mask.astype('uint8'), header_mask


################################################################################

def fill_sat_holes (data_mask, mask_value):

    """fill_holes and binary_close saturated pixels in data_mask"""

    value_sat = mask_value['saturated']
    value_satcon = mask_value['saturated-connected']
    mask_satcon = ((data_mask & value_sat == value_sat) |
                   (data_mask & value_satcon == value_satcon))
    struct = np.ones((3,3), dtype=bool)
    mask_satcon = ndimage.binary_closing(mask_satcon, structure=struct)
    mask_satcon = ndimage.binary_fill_holes(mask_satcon, structure=struct)
    mask_satcon2add = (mask_satcon & (data_mask==0))
    data_mask[mask_satcon2add] = value_satcon


################################################################################

def mask_header(data_mask, header_mask):

    """Function to add info from all reduction steps to mask header"""

    mask = {}
    text = {'bad': 'BP', 'edge': 'EP', 'saturated': 'SP',
            'saturated-connected': 'SCP', 'satellite trail': 'STP',
            'cosmic ray': 'CRP'}

    for mask_type in text.keys():
        value = get_par(set_zogy.mask_value[mask_type],tel)
        mask[mask_type] = (data_mask & value == value)
        header_mask['M-{}'.format(text[mask_type])] = (
            True, '{} pixels included in mask?'.format(mask_type))
        header_mask['M-{}VAL'.format(text[mask_type])] = (
            value, 'value added to mask for {} pixels'.format(mask_type))
        header_mask['M-{}NUM'.format(text[mask_type])] = (
            np.sum(mask[mask_type]), 'number of {} pixels'.format(mask_type))

    return


################################################################################

def master_prep (fits_master, data_shape, create_master, pick_alt=True,
                 tel=None, proc_mode=None):

    """function to create a master calibration file [fits_master] with
       shape [data_shape] if it does not already exist and does not
       contain a red flag, or to pick an existing one nearby in time.
       If [create_master] is False, or there are too few individual
       calibration files (<3) available, or the master created has
       been flagged red, a nearby master frame is searched for.
       In addition to that, [pick_alt] needs to be True

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()


    # infer path, imgtype, evening date and filter from input [fits_master]
    path, filename = os.path.split(fits_master)


    # previously, master frames were named [imgtype]_[date_eve]..
    # instead of [tel]_[imgtype]_[date_eve]..
    #imgtype, date_eve = filename.split('.fits')[0].split('_')[0:2]
    # the following selection handles both cases
    imgtype, date_eve = (filename.split('.fits')[0].split('{}_'.format(tel))[-1]
                         .split('_')[0:2])


    # for flat, also extract filter
    if imgtype == 'flat':
        filt = filename.split('.fits')[0].split('_')[-1]
    else:
        filt = None


    # check if already present (if fpacked, fits_master below will
    # point to fpacked file)
    master_present, fits_master = already_exists (fits_master, get_filename=True)


    if master_present:
        log.info ('master {} {} exists'.format(imgtype, fits_master))


    # check if master bias/flat does not contain any red flags:
    master_ok = True
    if master_present:
        if qc_flagged (fits_master):
            master_ok = False
            log.warning ('existing master {} {} contains a red flag'
                         .format(imgtype, fits_master))


    if not (master_present and master_ok):

        # in night mode only, sleep for 60s to make sure individual
        # biases and/or flats have been reduced and written to disk;
        # this is only used in the old "chopper" night mode, where the
        # master files were created on-the-fly triggered by the first
        # science frame of the night. At ilifu/Slurm or in the google
        # cloud, the master frames are prepared in advance before the
        # night starts
        google_cloud = (fits_master[0:5] == 'gs://')
        if proc_mode == 'night' and not google_cloud:
            log.warning ('waiting for 60s for all individual calibration frames '
                         'to have been reduced before continuing with '
                         '[master_prep]')
            time.sleep(60)


        # prepare master image from files in [path] +/- the specified
        # time window
        nwindow = int(get_par(set_bb.cal_window,tel)[imgtype])


        # for both ilifu/Slurm and google cloud, the individual
        # calibration files are in red_dir
        red_dir = get_par(set_bb.red_dir,tel)


        # collect individual calibration files in [file_list]
        file_list = []
        for n_day in range(-nwindow, nwindow+1):
            # determine mjd at noon (local or UTC, does not matter) of
            # date_eve +- n_day
            mjd_noon = date2mjd('{}'.format(date_eve), time_str='12:00') + n_day
            # corresponding path
            date_tmp = (Time(mjd_noon, format='mjd').isot.split('T')[0]
                        .replace('-','/'))
            path_tmp = '{}/{}/{}/{}_20'.format(red_dir, date_tmp, imgtype, tel)

            # additional search string, which will select particular
            # filter for flats
            if imgtype=='flat':
                search_str = '{}.fits'.format(filt)
            else:
                search_str = '.fits'

            # collect files
            file_list.append(list_files(path_tmp, search_str=search_str))


        # clean up lists in [file_list] and sort
        file_list = sorted([f for sublist in file_list for f in sublist])
        nfiles = len(file_list)


        if create_master:

            # do not consider image with header QC-FLAG set to red,
            # and also avoid using MeerLICHT evening flats due to dome
            # vignetting for period from July 2019 until February
            # 2020; moreover, mjd_obs is read from header to be able
            # to sort the calibration files in time futher below

            # execute this block only if [create_master] is switched
            # on, otherwise the line with read_hdulist below leads to
            # an exception when running both bias and flatfield
            # reductions on Slurm with multiple tasks/processes:
            # Header missing END card. [blackbox_reduce, line 1323]
            mjd_obs = np.zeros(nfiles)
            mask_keep = np.ones(nfiles, dtype=bool)
            for i_file, filename in enumerate(file_list):

                log.info ('reading header of {}'.format(filename))
                # check!!! - the following line leads to an exception when
                # running both bias and flatfield reductions on Slurm with
                # multiple tasks/processes:
                # Header missing END card. [blackbox_reduce, line 1323]
                header_tmp = read_hdulist (filename, get_data=False,
                                           get_header=True)
                if 'QC-FLAG' in header_tmp and header_tmp['QC-FLAG'] == 'red':
                    mask_keep[i_file] = False

                # record MJD-OBS in array
                if 'MJD-OBS' in header_tmp:
                    mjd_obs[i_file] = header_tmp['MJD-OBS']

                # for period from July 2019 until February 2020, avoid
                # using MeerLICHT evening flats due to dome vignetting
                mjd_avoid = Time(['2019-07-01T12:00:00', '2020-03-01T12:00:00'],
                                 format='isot').mjd
                if (tel=='ML1' and mjd_obs[i_file] % 1 > 0.5 and
                    mjd_obs[i_file] > mjd_avoid[0] and
                    mjd_obs[i_file] < mjd_avoid[1]):

                    mask_keep[i_file] = False


                if imgtype == 'flat':
                    # BlackGEM evening flats show some sort of reflection,
                    # which is absent in the morning flats, so only use
                    # the latter through the setting parameter
                    # flat_reject_eve
                    flat_reject_eve = get_par(set_bb.flat_reject_eve,tel)
                    if (flat_reject_eve and
                        (mjd_obs[i_file] % 1 > 0.5 or
                         # for BlackGEM in Chilean summer, flats can
                         # be taken just after midnight UT
                         mjd_obs[i_file] % 1 < 0.1)):

                        log.warn ('rejecting evening flat {}'.format(filename))
                        mask_keep[i_file] = False



            file_list = np.array(file_list)[mask_keep]
            mjd_obs = mjd_obs[mask_keep]
            nfiles = len(file_list)



        # look for a nearby master instead if the master bias/flat
        # present contains a red flag, or if there are too few
        # individual frames to make a master, or the input
        # [create_master] is switched off
        if nfiles < 5 or not master_ok or not create_master:

            if imgtype == 'flat':
                msg = 'flat in filter {}'.format(filt)
            else:
                msg = imgtype

            # if input [pick_alt] is True, look for a nearby master
            # flat, otherwise just return None
            if pick_alt or not create_master:
                fits_master_near = get_nearest_master(date_eve, imgtype,
                                                      fits_master, filt=filt)
            else:
                if master_ok:
                    log.warning ('too few good frames available to produce '
                                 'master {} for evening date {} +/- window '
                                 'of {} days'.format(msg, date_eve, nwindow))
                return None


            if fits_master_near is not None:

                # if master bias subtraction switch is off, the master
                # bias is still prepared; only show message below in
                # case switch is on, otherwise it is confusing
                if ((imgtype=='bias' and get_par(set_bb.subtract_mbias,tel))
                    or imgtype=='flat'):

                    log.warning ('using {} as master for evening date {}'
                                 .format(fits_master_near, date_eve))

                # previously we created a symbolic link so future
                # files would automatically use this as the master
                # file, but as this symbolic link is confusing, let's
                # not do that; searching for nearby master frame takes
                # a negligible amount of time
                # os.symlink(fits_master_near, fits_master)
                fits_master = fits_master_near

            else:
                if ((imgtype=='bias' and get_par(set_bb.subtract_mbias,tel))
                    or imgtype=='flat'):

                    log.error('no alternative master {} found'.format(msg))

                return None

        else:


            # should number of biases/darks/flats exceeds ncal_max,
            # select the ones closest in time to midnight of the
            # evening date
            nmax = int(get_par(set_bb.ncal_max,tel)[imgtype])

            # difference between observed MJD and midnight of the
            # evening date
            mjd_midnight = date2mjd('{}'.format(date_eve), time_str='23:59')
            mjd_obs_delta = mjd_obs - mjd_midnight
            # sort the observed delta MJDs of the files
            index_sort = np.argsort (np.abs(mjd_obs_delta))
            # select nmax
            file_list = file_list[index_sort][0:nmax]
            # update mjd_obs_delta for use further below
            mjd_obs_delta = mjd_obs_delta[index_sort][0:nmax]
            nfiles_orig = nfiles
            nfiles = len(file_list)


            # if nearest flat taken in the past is not within 12 hours
            # of midnight of the evening date, do not bother to make a
            # new master as it would be similar to (or worse than)
            # yesterday's master; make an exception if flats from
            # future nights are included
            all_past = np.all(mjd_obs_delta < 0)
            if np.amin(np.abs(mjd_obs_delta)) > 0.5 and all_past:
                log.warning ('all {} selected calibration files closest in time '
                             'to midnight of {} are from before this date and '
                             'taken longer than 12 hours ago; no point in '
                             'making master {}'
                             .format(nmax, date_eve, fits_master))
                return None


            # create the master frame
            if imgtype=='flat':
                msg = 'flat in filter {}'.format(filt)
            else:
                msg = imgtype

            log.info ('making {} master {} for night {} from the following '
                      'files:\n{}'.format(tel, msg, date_eve, file_list))

            if nfiles_orig > nmax:
                log.warning ('number of available {} frames ({}) exceeds the '
                             'maximum specified ({}); using the frames closest '
                             'in time to midnight of the evening date ({})'
                             .format(imgtype, len(index_sort), nmax, date_eve))

            if imgtype=='bias' and not get_par(set_bb.subtract_mbias,tel):
                log.warning ('this master bias will not be applied to the input '
                             'image as [subtract_mbias] is set to False)')


            # assuming that individual flats/biases have the same
            # shape as the input data
            ysize, xsize = data_shape
            master_cube = np.zeros((nfiles, ysize, xsize), dtype='float32')


            # initialize master header
            header_master = fits.Header()


            # fill the cube; ra_flats and dec_flats are used to check
            # offsets between flats
            ra_flats = []
            dec_flats = []
            for i_file, filename in enumerate(file_list):

                master_cube[i_file], header_tmp = read_hdulist(filename,
                                                               get_header=True)

                if imgtype=='flat':

                    # infer median over normalization section, either
                    # from header of individual flat, or recalculate
                    if 'MEDSEC' in header_tmp:
                        median = header_tmp['MEDSEC']
                    else:
                        index_flat_norm = get_par(set_bb.flat_norm_sec,tel)
                        median = np.median(master_cube[i_file][index_flat_norm])


                    # divide by median over the region [set_bb.flat_norm_sec]
                    log.info ('flat name: {}, median: {:.1f} e-'
                              .format(filename, median))

                    if median != 0:
                        master_cube[i_file] /= median

                    # collect RA and DEC to check for dithering
                    if 'RA' in header_tmp and 'DEC' in header_tmp:
                        ra_flats.append(header_tmp['RA'])
                        dec_flats.append(header_tmp['DEC'])


                # copy some header keyword values from first file
                if i_file==0:
                    for key in ['IMAGETYP', 'DATE-OBS', 'FILTER', 'RA', 'DEC',
                                'XBINNING', 'YBINNING', 'MJD-OBS', 'AIRMASS',
                                'ORIGIN', 'TELESCOP', 'PYTHON-V', 'BB-V']:
                        if key in header_tmp:
                            header_master[key] = (header_tmp[key],
                                                  header_tmp.comments[key])


                if imgtype=='flat':
                    comment = 'name reduced flat'
                else:
                    comment = 'name gain/os-corrected {} frame'.format(imgtype)

                # add name reduced calibration file to master header
                header_master['{}{}'.format(imgtype.upper(), i_file+1)] = (
                    filename.split('/')[-1].split('.fits')[0],
                    '{} {}'.format(comment, i_file+1))

                # add original name of calibration file to master header
                if 'ORIGFILE' in header_tmp.keys():
                    header_master['{}OR{}'.format(imgtype.upper(), i_file+1)] = (
                        header_tmp['ORIGFILE'], 'name original {} {}'
                        .format(imgtype, i_file+1))

                # also copy a few header keyword values from the last file
                if i_file==nfiles-1:
                    for key in ['DATE-END', 'MJD-END']:
                        if key in header_tmp:
                            header_master[key] = (header_tmp[key],
                                                  header_tmp.comments[key])


            # determine the median
            master_median = np.median(master_cube, axis=0)


            # add number of files combined
            header_master['N{}'.format(imgtype.upper())] = (
                nfiles, 'number of {} frames combined'.format(imgtype.lower()))


            # add time window used
            header_master['{}-WIN'.format(imgtype.upper())] = (
                nwindow, '[days] input time window to include {} frames'
                .format(imgtype.lower()))


            # add some more header keywords to the master flat
            if imgtype=='flat':

                sec_tmp = get_par(set_bb.flat_norm_sec,tel)
                value_tmp = '[{}:{},{}:{}]'.format(
                    sec_tmp[0].start+1, sec_tmp[0].stop+1,
                    sec_tmp[1].start+1, sec_tmp[1].stop+1)
                header_master['STATSEC'] = (
                    value_tmp, 'pre-defined statistics section [y1:y2,x1:x2]')


                header_master['MFMEDSEC'] = (
                    np.median(master_median[sec_tmp]),
                    'median master flat over STATSEC')


                header_master['MFSTDSEC'] = (
                    np.std(master_median[sec_tmp]),
                    'sigma (STD) master flat over STATSEC')


                # "full" image statistics
                index_stat = get_rand_indices(master_median.shape)
                __, median_master, std_master = sigma_clipped_stats(
                    master_median[index_stat], mask_value=0)
                header_master['MFMED'] = (median_master, 'median master flat')
                header_master['MFSTD'] = (std_master, 'sigma (STD) master flat')


                # check if flats were dithered; calculate offset in
                # arcsec of each flat with respect to the previous one
                ra_flats = np.array(ra_flats)
                dec_flats = np.array(dec_flats)
                noffset = 0
                offset_mean = 0
                if len(ra_flats) > 0 and len(dec_flats) > 0:
                    offset = 3600. * haversine (ra_flats, dec_flats,
                                                np.roll(ra_flats,1),
                                                np.roll(dec_flats,1))
                    # count how many were offset by at least 5"
                    mask_off = (offset >= 5)
                    noffset = np.sum(mask_off)
                    if noffset > 0:
                        offset_mean = np.mean(offset[mask_off])


                header_master['N-OFFSET'] = (noffset, 'number of flats with '
                                             'offsets > 5 arcsec')
                header_master['OFF-MEAN'] = (offset_mean,
                                             '[arcsec] mean dithering offset')

                if float(noffset)/nfiles >= 0.66:
                    flat_dithered = True
                else:
                    flat_dithered = False

                header_master['FLATDITH'] = (flat_dithered,
                                             'majority of flats were dithered')


                # set edge and non-positive pixels to 1; edge pixels
                # are identified by reading in bad pixel mask as
                # master preparation is not necessariliy linked to the
                # mask of an object image, e.g. in function
                # [masters_left]
                fits_bpm = (get_par(set_bb.bad_pixel_mask,tel)
                            .replace('bpm', 'bpm_{}'.format(filt)))
                bpm_present, fits_bpm = already_exists (fits_bpm,
                                                        get_filename=True)

                if bpm_present:
                    # if mask exists, read it
                    data_mask = read_hdulist(fits_bpm)
                    mask_replace = ((data_mask==get_par(
                        set_zogy.mask_value['edge'],tel)) | (master_median<=0))
                    master_median[mask_replace] = 1


                # now that master flat is produced, calculate - but do
                # not apply - the different channels' normalization
                # factors such that the resulting image would appear
                # smooth without any jumps in levels between the
                # different channels
                __, __, __, __, data_sec_red = define_sections(data_shape,
                                                               tel=tel)
                nchans = np.shape(data_sec_red)[0]
                med_chan_cntr = np.zeros(nchans)
                std_chan_cntr = np.zeros(nchans)


                # copy of master_median
                master_median_corr = np.copy(master_median)


                # first match the channels vertically, by using the
                # statistics of the regions at the top of the bottom
                # channels and bottom of the top channels
                nrows = 200
                for i_chan in range(nchans):
                    data_chan = master_median_corr[data_sec_red[i_chan]]
                    if i_chan < 8:
                        med_chan_cntr[i_chan] = np.median(data_chan[-nrows:,:])
                    else:
                        med_chan_cntr[i_chan] = np.median(data_chan[0:nrows,:])

                    # correct master image channel
                    master_median_corr[data_sec_red[i_chan]] /= med_chan_cntr[i_chan]


                # channel correction factor applied so far
                factor_chan = 1./med_chan_cntr


                # now match channels horizontally
                ysize, xsize = data_shape
                ny = get_par(set_bb.ny,tel)
                nx = get_par(set_bb.nx,tel)
                dy = ysize // ny
                dx = xsize // nx


                nrows = 2000
                ncols = 200
                for i in range(1,nx):
                    # index of lower left pixel of upper right channel
                    # of the 4 being considered
                    y_index = dy
                    x_index = i*dx

                    # statistics of right side of previous channel pair
                    data_stat1 = master_median_corr[y_index-nrows:y_index+nrows,
                                                    x_index-ncols:x_index]

                    # statistics of left side of current channel pair
                    data_stat2 = master_median_corr[y_index-nrows:y_index+nrows,
                                                    x_index:x_index+ncols]
                    ratio = np.median(data_stat1)/np.nanmedian(data_stat2)

                    # correct relevant channels
                    master_median_corr[data_sec_red[i]] *= ratio
                    master_median_corr[data_sec_red[i+nx]] *= ratio

                    # update correction factor
                    factor_chan[i] *= ratio
                    factor_chan[i+nx] *= ratio


                if False:
                    # normalize corrected master to [flat_norm_sec] section
                    sec_tmp = get_par(set_bb.flat_norm_sec,tel)
                    ratio_norm = np.median(master_median_corr[sec_tmp])
                    master_median_corr /= ratio_norm
                    factor_chan /= ratio_norm


                # normalize correction factors to an average of unity
                factor_chan /= np.mean(factor_chan)


                # add factor_chan values to header
                for i_chan in range(nchans):
                    header_master['GAINCF{}'.format(i_chan+1)] = (
                        factor_chan[i_chan], 'channel {} gain correction factor'
                        .format(i_chan+1))


            elif imgtype=='bias':

                # add some header keywords to the master bias
                index_stat = get_rand_indices(master_median.shape)
                mean_master, __, std_master = sigma_clipped_stats(
                    master_median[index_stat], mask_value=0)
                header_master['MBMEAN'] = (mean_master, '[e-] mean master bias')
                header_master['MBRDN'] = (std_master, '[e-] sigma (STD) master '
                                          'bias')

                # including the means and standard deviations of the master
                # bias in the separate channels
                __, __, __, __, data_sec_red = define_sections(data_shape,
                                                               tel=tel)
                nchans = np.shape(data_sec_red)[0]
                mean_chan = np.zeros(nchans)
                std_chan = np.zeros(nchans)

                for i_chan in range(nchans):
                    data_chan = master_median[data_sec_red[i_chan]]
                    index_stat = get_rand_indices(data_chan.shape)
                    mean_chan[i_chan], __, std_chan[i_chan] = sigma_clipped_stats(
                        data_chan[index_stat], mask_value=0)

                for i_chan in range(nchans):
                    header_master['MBIASM{}'.format(i_chan+1)] = (
                        mean_chan[i_chan], '[e-] channel {} mean master bias'
                        .format(i_chan+1))

                for i_chan in range(nchans):
                    header_master['MBRDN{}'.format(i_chan+1)] = (
                        std_chan[i_chan], '[e-] channel {} sigma (STD) master '
                        'bias'.format(i_chan+1))


            elif imgtype=='dark':

                # add some header keywords to the master dark
                index_stat = get_rand_indices(master_median.shape)
                mean_master, __, std_master = sigma_clipped_stats(
                    master_median[index_stat], mask_value=0)
                header_master['MDMEAN'] = (mean_master, '[e-] mean master dark')
                header_master['MDRDN'] = (std_master, '[e-] sigma (STD) master '
                                          'dark')

                # including the means and standard deviations of the master
                # dark in the separate channels
                __, __, __, __, data_sec_red = define_sections(data_shape,
                                                               tel=tel)
                nchans = np.shape(data_sec_red)[0]
                mean_chan = np.zeros(nchans)
                std_chan = np.zeros(nchans)

                for i_chan in range(nchans):
                    data_chan = master_median[data_sec_red[i_chan]]
                    index_stat = get_rand_indices(data_chan.shape)
                    mean_chan[i_chan], __, std_chan[i_chan] = sigma_clipped_stats(
                        data_chan[index_stat], mask_value=0)

                for i_chan in range(nchans):
                    header_master['MDARKM{}'.format(i_chan+1)] = (
                        mean_chan[i_chan], '[e-] channel {} mean master dark'
                        .format(i_chan+1))

                for i_chan in range(nchans):
                    header_master['MDRDN{}'.format(i_chan+1)] = (
                        std_chan[i_chan], '[e-] channel {} sigma (STD) master '
                        'dark'.format(i_chan+1))


            # call [run_qc_check] to update master header with any QC flags
            run_qc_check (header_master, tel)

            # write fits
            fits_master = write_fits (
                fits_master, master_median.astype('float32'), header_master,
                master=True, tel=tel)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in master_prep')


    return fits_master


################################################################################

def delta_one_month (date_eve, dmonth):

    # make sure there are no characters in [date_eve]
    date_eve = ''.join(e for e in date_eve if e.isdigit())

    if dmonth == 0:
        # mjd of date_eve
        mjd_noon = date2mjd('{}'.format(date_eve), time_str='12:00')

    elif dmonth == -1:
        # mjd of last day of last month
        day = int(date_eve[6:8])
        mjd_noon = date2mjd('{}'.format(date_eve), time_str='12:00') - (day+1)

    elif dmonth == 1:
        # mjd of first day of next month
        year = int(date_eve[0:4])
        month = int(date_eve[4:6])
        day = 1
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

        date_eve_new = '{}{:02}{:02}'.format(year, month, day)
        mjd_noon = date2mjd('{}'.format(date_eve_new), time_str='12:00')

    else:
        log.error ('maximum [dmonth] in [delta_one_month] is 1')


    # return corresponding path without the day: yyyy/mm/
    return (Time(mjd_noon, format='mjd').isot.split('T')[0]
            .replace('-','/'))[0:8]


################################################################################

def get_nearest_master (date_eve, imgtype, fits_master, filt=None):

    log.info ('running get_nearest_master')


    if get_par(set_zogy.timing,tel):
        t = time.time()


    # first look for yesterday's master to avoid searching for nearby
    # masters in the Google Cloud
    date_eve_dash = '{}-{}-{}'.format(date_eve[0:4], date_eve[4:6],
                                      date_eve[6:8])
    date_eve_slash = date_eve_dash.replace('-','/')
    date_eve_yest_dash = (Time(date_eve_dash)-1*u.day).isot.split('T')[0]
    date_eve_yest_slash = date_eve_yest_dash.replace('-','/')
    date_eve_yest = date_eve_yest_dash.replace('-','')

    fits_master_yest = (fits_master.replace(date_eve, date_eve_yest)
                        .replace(date_eve_slash, date_eve_yest_slash))
    master_present, fits_master = already_exists (fits_master_yest,
                                                  get_filename=True)


    if master_present and not qc_flagged (fits_master):

        # if it exists and it is not red-flagged, return yesterday's
        # master
        par2return = fits_master

    else:

        # otherwise look for nearest master over the current, past and
        # next month; base path to directory with master frames
        master_dir = get_par(set_bb.master_dir,tel)


        # go through search window containing the previous, current
        # and next month
        file_list = []
        nwindow = 1
        for n_month in range(-nwindow, nwindow+1):

            # use function [delta_one_month] to extract path yyyy/mm/
            date_tmp = delta_one_month (date_eve, n_month)

            # create path to be used in [list_files]:
            # [master_dir]/yyyy/mm
            path_tmp = '{}/{}'.format(master_dir, date_tmp)

            # start string, [tel]_[imgtype]_, e.g. 'ML1_bias_'
            start_str = '{}_{}_'.format(tel, imgtype)

            # end_str of filename, where filter can be specified for flats
            if imgtype=='flat':
                end_str = '{}.fits.fz'.format(filt)
            else:
                end_str = '.fits.fz'

            # collect files
            file_list.append(list_files(path_tmp, start_str=start_str,
                                        end_str=end_str, recursive=True))



        # clean up lists in [file_list] and sort
        file_list = sorted([f for sublist in file_list for f in sublist])
        nfiles = len(file_list)


        # find file that is closest in time to [date_eve]
        if nfiles > 0:

            # list of evening dates corresponding to file_list
            list_date_eve = [''.join(f.split('/')[-5:-2]) for f in file_list]

            # mjds
            mjds = np.array([date2mjd(date) for date in list_date_eve])

            # these mjds corresponding to the very start of the day
            # (midnight) but in the comparison this offset cancels out
            delta_mjds = abs(mjds - date2mjd(date_eve))
            idx_sort = np.argsort(delta_mjds)

            # loop these starting with nearest one, and select
            # nearest one that is not red-flagged
            for i_near in idx_sort:
                # check if it is red-flagged
                file_near = file_list[i_near]
                if not qc_flagged (file_near):
                    par2return = file_near
                    break
            else:
                # if none of the flats are ok, return None
                par2return = None

        else:
            # if no files found, return None
            par2return = None



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in get_nearest_master')


    return par2return


################################################################################

def qc_flagged (fits_name, flag='red'):

    # check if header of [fits_name] contains red flag
    header = read_hdulist (fits_name, get_data=False, get_header=True)

    if ('QC-FLAG' in header and header['QC-FLAG']==flag):
        return True
    else:
        return False


################################################################################

def date2mjd (date_str, time_str=None, get_jd=False):

    """convert [date_str] and [time_str] to MJD or JD with possible
    formats: yyyymmdd or yyyy-mm-dd for [date_str] and hhmmss[.s] or
    hh:mm[:ss.s] for [time_str]

    """

    if '-' not in date_str:
        date_str = '{}-{}-{}'.format(date_str[0:4],
                                     date_str[4:6],
                                     date_str[6:8])

    if time_str is not None:
        if ':' not in time_str:
            time_str = '{}:{}:{}'.format(time_str[0:2],
                                         time_str[2:4],
                                         time_str[4:])
        date_str = '{} {}'.format(date_str, time_str)


    if get_jd:
        return Time(date_str).jd
    else:
        return Time(date_str).mjd


################################################################################

def check_header1 (header, filename):

    header_ok = True

    # check that all crucial keywords are present in the header; N.B.:
    # [sort_files] function near top of BlackBOX already requires the
    # IMAGETYP keyword so this need not really be checked here

    # crucial keywords for any image type
    for key in ['IMAGETYP', 'DATE-OBS', 'FILTER']:
        if key not in header:
            log.error ('crucial keyword {} not present in header; '
                       'not processing {}'.format(key, filename))
            header_ok = False
            # return immediately in this case as keyword 'IMAGETYP' is
            # used below which may not exist
            return header_ok


    # define imgtype
    imgtype = header['IMAGETYP'].lower()


    # for early ML data, header keyword FIELD_ID instead of OBJECT was
    # used for the field identification
    if 'FIELD_ID' in header:
        obj = header['FIELD_ID']
    elif 'OBJECT' in header:
        obj = header['OBJECT']
    else:
        if imgtype=='object':
            # if neither FIELD_ID nor OBJECT present in header of an
            # object image, then also bail out
            log.error ('FIELD_ID or OBJECT keyword not present in '
                       'header; not processing {}'.format(filename))
            header_ok = False
            # return right away as otherwise [obj] not defined, which
            # is used below
            return header_ok


    if imgtype=='object':

        # check if OBJECT keyword value contains digits only
        try:
            int(obj)
        except Exception as e:
            log.exception ('keyword OBJECT (or FIELD_ID if present) does '
                           'not contain digits only; not processing {}'
                           .format(filename))
            header_ok = False

        else:
            # check if OBJECT keyword is in the right range 1-19999
            if int(obj)==0 or int(obj)>=20000:
                log.error ('OBJECT (or FIELD_ID) not in range 1-19999; '
                           'not processing {}'.format(filename))
                header_ok = False


        # remaining important keywords; for biases, darks and flats, these
        # keywords are not strictly necessary (although for flats they are
        # used to check if they were dithered; if RA and DEC not present,
        # any potential dithering will not be detected)
        for keys in ['EXPTIME', 'RA', 'DEC']:
            if key not in header:
                log.error ('crucial keyword {} not present in header; '
                           'not processing {}'.format(key, filename))
                header_ok = False


    # check if filename contains 'test'
    if False:
        if 'test' in filename.lower():
            log.warning ('filename contains string \'test\'; '
                         'not processing {}'.format(filename))
            header_ok = False


    return header_ok


################################################################################

def radec_offset (header, filename):

    # determine the offset between the RA-CNTR and DEC-CNTR (inferred
    # in [zogy]) and the expected RA and DEC from the definition of
    # ML/BG field IDs, and add the offset to the header. The header
    # value can be used in the QC check.

    # ML/BG field definition contained in fits table with columns
    # 'field_id', 'ra_c', 'dec_c'; previously an ASCII file
    mlbg_fieldIDs = get_par(set_bb.mlbg_fieldIDs,tel)
    #table_grid = ascii.read(mlbg_fieldIDs, names=['ID', 'RA', 'DEC'],
    #                        data_start=0)
    table_grid = Table.read(mlbg_fieldIDs, memmap=True)


    if 'RA-CNTR' in header and 'DEC-CNTR' in header:

        ra_cntr = header['RA-CNTR']
        dec_cntr = header['DEC-CNTR']

        # find relevant object/field ID in field definition
        obj = header['OBJECT']
        mask_match = (table_grid['field_id'].astype(int) == int(obj))
        i_grid = np.nonzero(mask_match)[0][0]

        # calculate offset in degrees
        offset_deg = haversine(table_grid['ra_c'][i_grid],
                               table_grid['dec_c'][i_grid], ra_cntr, dec_cntr)

        offset_max = 60.
        if offset_deg > offset_max/60.:
            log.warning (
                'input header field ID, RA-CNTR and DEC-CNTR combination '
                'is inconsistent (>{}\') with definition of field IDs\n'
                'header field ID: {}, RA-CNTR: {:.4f}, DEC-CNTR: {:.4f}\n'
                'vs.    field ID: {}, RA     : {:.4f}, DEC     : {:.4f} '
                'in {} for {}'
                .format(offset_max, obj, ra_cntr, dec_cntr,
                        table_grid['field_id'][i_grid],
                        table_grid['ra_c'][i_grid],
                        table_grid['dec_c'][i_grid], mlbg_fieldIDs, filename))


        # calculate offsets in RA and DEC separately
        offset_ra = haversine(table_grid['ra_c'][i_grid], dec_cntr,
                              ra_cntr, dec_cntr)
        offset_dec = haversine(ra_cntr, table_grid['dec_c'][i_grid],
                               ra_cntr, dec_cntr)


    else:
        # set offsets to 'None'
        offset_deg = 'None'
        offset_ra =  'None'
        offset_dec = 'None'


    # add header keywords
    header['RADECOFF'] = (offset_deg, '[deg] offset RA,DEC-CNTR wrt ML/BG field '
                          'grid')
    header['RAOFF'] = (offset_ra, '[deg] offset RA-CNTR wrt ML/BG field grid RA')
    header['DECOFF'] = (offset_dec, '[deg] offset DEC-CNTR wrt ML/BG field grid '
                        'DEC')


    return


################################################################################

def check_header2 (header, filename):

    header_ok = True

    # check if the field ID and RA-REF, DEC-REF combination is
    # consistent with definition of ML/BG field IDs; threshold used:
    # 10 arc minutes
    offset_max = 10.

    mlbg_fieldIDs = get_par(set_bb.mlbg_fieldIDs,tel)
    table_grid = Table.read(mlbg_fieldIDs, memmap=True)
    imgtype = header['IMAGETYP'].lower()
    if imgtype=='object':
        obj = header['OBJECT']
        # use REF coords; do not use RA, DEC because they could be off
        if header['RA-REF'] != 'None' and header['DEC-REF'] != 'None':
            ra_deg = Angle(header['RA-REF'], unit=u.hour).degree
            dec_deg = Angle(header['DEC-REF'], unit=u.deg).degree


            # check if there is a match with the defined field IDs
            mask_match = (table_grid['field_id'].astype(int) == int(obj))
            if sum(mask_match) == 0:
                # observed field is not present in definition of field IDs
                header_ok = False
                log.error ('input header field ID not present in definition '
                           'of field IDs:\n{}\nheader field ID: {}, RA-REF: '
                           '{:.4f}, DEC-REF: {:.4f}\nnot processing {}'
                           .format(mlbg_fieldIDs, obj, ra_deg, dec_deg,
                                   filename))

            else:
                i_grid = np.nonzero(mask_match)[0][0]
                if haversine(table_grid['ra_c'][i_grid],
                             table_grid['dec_c'][i_grid],
                             ra_deg, dec_deg) > offset_max/60.:
                    log.error (
                        'input header field ID, RA-REF and DEC-REF combination '
                        'is inconsistent (>{}\') with definition of field IDs\n'
                        'header field ID: {}, RA-REF: {:.4f}, DEC-REF: {:.4f}\n'
                        'vs.    field ID: {}, RA    : {:.4f}, DEC    : {:.4f} '
                        'in {}\nnot processing {}'
                        .format(offset_max, obj, ra_deg, dec_deg,
                                table_grid['field_id'][i_grid],
                                table_grid['ra_c'][i_grid],
                                table_grid['dec_c'][i_grid], mlbg_fieldIDs,
                                filename))

                    header_ok = False


    # if binning is not 1x1, also skip processing
    if 'XBINNING' in header and 'YBINNING' in header:
        if int(header['XBINNING']) != 1 or int(header['YBINNING']) != 1:
            log.error ('BINNING not 1x1; not processing {}'.format(filename))

            header_ok = False


    return header_ok


################################################################################

def set_header(header, filename, silent=False):

    def edit_head (header, key, value=None, comments=None, dtype=None,
                   silent=silent):
        # update value
        if value is not None:
            if key in header:
                if header[key] != value and value != 'None':
                    header[key] = value
                    if not silent:
                        log.warning ('value of existing keyword {} updated from '
                                     '{} to {}'.format(key, header[key], value))
            else:
                header[key] = value
        # update comments
        if comments is not None:
            if key in header:
                header.comments[key] = comments
            else:
                if not silent:
                    log.warning ('keyword {} does not exist: comment is not '
                                 'updated'.format(key))
        # update dtype
        if dtype is not None:
            if key in header and header[key] != 'None':
                header[key] = dtype(header[key])
            else:
                if not silent:
                    log.warning ('dtype of keyword {} is not updated'
                                 .format(key))



    edit_head(header, 'NAXIS', comments='number of array dimensions')
    edit_head(header, 'NAXIS1', comments='length of array axis')
    edit_head(header, 'NAXIS2', comments='length of array axis')

    edit_head(header, 'BUNIT', value='e-',
              comments='Physical unit of array values')
    edit_head(header, 'BSCALE', comments='value = fits_value*BSCALE+BZERO')
    edit_head(header, 'BZERO', comments='value = fits_value*BSCALE+BZERO')
    #edit_head(header, 'CCD-AMP', value='', comments='Amplifier mode of the CCD camera')
    #edit_head(header, 'CCD-SET', value='', comments='CCD settings file')

    edit_head(header, 'CCD-TEMP', value='None',
              comments='[C] Current CCD temperature')

    if 'XBINNING' in header:
        edit_head(header, 'XBINNING', comments='[pix] Binning factor X axis')
    else:
        xsize = int(header['NAXIS1'])
        nx = get_par(set_bb.nx,tel)
        dx = get_par(set_bb.xsize_chan,tel)
        xbinning = int(np.ceil(float(nx*dx)/xsize))
        edit_head(header, 'XBINNING', value=xbinning,
                  comments='[pix] Binning factor X axis')

    if 'YBINNING' in header:
        edit_head(header, 'YBINNING', comments='[pix] Binning factor Y axis')
    else:
        ysize = int(header['NAXIS2'])
        ny = get_par(set_bb.ny,tel)
        dy = get_par(set_bb.ysize_chan,tel)
        ybinning = int(np.ceil(float(ny*dy)/ysize))
        edit_head(header, 'YBINNING', value=ybinning,
                  comments='[pix] Binning factor Y axis')


    edit_head(header, 'RADESYS', value='ICRS',
              comments='Coordinate reference frame')
    edit_head(header, 'EPOCH', value=get_par(set_zogy.cal_epoch,tel),
              comments='Coordinate reference epoch')


    edit_head(header, 'DOMEAZ', value='None', dtype=float,
              comments='[deg] Dome azimuth (N=0;E=90)')


    edit_head(header, 'FLIPSTAT', value='None',
              comments='Telescope side of the pier')
    edit_head(header, 'EXPTIME', comments='[s] Requested exposure time')

    if 'ISTRACKI' in header:
        # convert the string value to boolean
        #value = (header['ISTRACKI']=='True')
        value = str2bool(header['ISTRACKI'])
        edit_head(header, 'ISTRACKI', value=value,
                  comments='Telescope is tracking')


    # record original DATE-OBS and END-OBS in ACQSTART and ACQEND
    edit_head(header, 'ACQSTART', value=header['DATE-OBS'],
              comments='start of acquisition (server timing)')
    if 'END-OBS' in header:
        edit_head(header, 'ACQEND', value=header['END-OBS'],
                  comments='end of acquisition (server timing)')
    else:
        edit_head(header, 'ACQEND', value='None',
                  comments='end of acquisition (server timing)')


    # for some flatsfieds, IMAGETYP was erroneously set to Object;
    # update those
    imgtype = header['IMAGETYP'].lower()
    if 'flat' in filename.lower() and imgtype == 'object':
        imgtype = 'flat'
        edit_head(header, 'IMAGETYP', value=imgtype)


    # midexposure DATE-OBS is based on GPSSTART and GPSEND; if these
    # keywords are not present in the header, or if the image is a
    # bias or dark frame (which both should not contain these
    # keywords, and if they do, the keyword values are actually
    # identical to those of the image preceding the bias/dark), then
    # just adopt the original DATE-OBS (=ACQSTART) as the date of
    # observation
    if ('GPSSTART' in header and 'GPSEND' in header and
        header['GPSSTART'] != header['GPSEND'] and
        (imgtype == 'object' or imgtype == 'flat')):

        # replace DATE-OBS with (GPSSTART+GPSEND)/2
        gps_mjd = Time([header['GPSSTART'], header['GPSEND']], format='isot').mjd
        mjd_obs = np.sum(gps_mjd)/2
        date_obs_str = Time(mjd_obs, format='mjd').isot
        edit_head(header, 'DATE-OBS', value=date_obs_str,
                  comments='Midexp. date @img cntr:(GPSSTART+GPSEND)/2')

        # change from a string to Time instance, used below
        date_obs = Time(date_obs_str, format='isot')

        # also add keyword to check (GPSEND-GPSSTART) - EXPTIME
        gps_shut = (gps_mjd[1]-gps_mjd[0])*24*3600. - float(header['EXPTIME'])
        edit_head(header, 'GPS-SHUT', value=gps_shut,
                  comments='[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME')

    else:
        # assume date_obs to be the original header's DATE-OBS
        # (=ACQSTART) plus exptime/2 plus 1s which appears to be a
        # reasonable estimate for BlackGEM
        exptime = float(header['EXPTIME'])
        date_obs = Time(header['DATE-OBS'], format='isot')
        # add exptime/2 plus 1s, except for bias frames
        if imgtype != 'bias':
            date_obs += (exptime/2+1)*u.s

        # update header DATE-OBS
        date_obs_str = date_obs.isot
        edit_head(header, 'DATE-OBS', value=date_obs_str,
                  comments='Midexp. date @img cntr:(ACQSTART+EXPTIME/2+1s)')

        # also define mjd_obs, which will be added to the header below
        mjd_obs = date_obs.mjd



    edit_head(header, 'GPSSTART', value='None',
              comments='GPS timing start of opening shutter')
    edit_head(header, 'GPSEND', value='None',
              comments='GPS timing end of opening shutter')

    if imgtype == 'object' or imgtype == 'flat':
        edit_head(header, 'GPS-SHUT', value='None',
                  comments='[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME')

    edit_head(header, 'MJD-OBS', value=mjd_obs,
              comments='[d] MJD (using DATE-OBS)')

    # in degrees:
    lon_temp = get_par(set_zogy.obs_lon,tel)
    lst = date_obs.sidereal_time('apparent', longitude=lon_temp)
    lst_deg = lst.deg
    # in hh:mm:ss.sss
    lst_str = lst.to_string(sep=':', precision=3)
    edit_head(header, 'LST', value=lst_str,
              comments='apparent LST (using DATE-OBS)')

    utc = (mjd_obs-np.floor(mjd_obs)) * 3600. * 24
    edit_head(header, 'UTC', value=utc, comments='[s] UTC (using DATE-OBS)')
    edit_head(header, 'TIMESYS', value='UTC', comments='Time system used')


    # telescope latitude, longitude and height (used for AIRMASS and
    # SITELONG, SITELAT and ELEVATIO)
    lat = get_par(set_zogy.obs_lat,tel)
    lon = get_par(set_zogy.obs_lon,tel)
    height = get_par(set_zogy.obs_height,tel)


    # this keyword is added later on by Astrometry.net
    #edit_head(header, 'EQUINOX', value='None',
    #          comments='[yr] equatorial coordinates definition')


    if 'RA' in header and 'DEC' in header:

        # RA
        if ':' in str(header['RA']):
            # convert sexagesimal to decimal degrees
            ra_deg = Angle(header['RA'], unit=u.hour).degree
        else:
            # convert RA decimal hours to degrees
            ra_deg = float(header['RA']) * 15.

        # DEC
        if ':' in str(header['DEC']):
            # convert sexagesimal to decimal degrees
            dec_deg = Angle(header['DEC'], unit=u.deg).degree
        else:
            # for ra_icrs, dec_icrs and airmass determination below
            # float is needed as sometimes it is a string
            dec_deg = float(header['DEC'])

        # assuming RA,DEC are JNOW, convert them to J2000/ICRS
        equinox = Time(mjd_obs, format='mjd').jyear_str
        ra_icrs, dec_icrs = jnow2icrs (ra_deg, dec_deg, equinox)

        edit_head(header, 'RA', value=ra_icrs,
                  comments='[deg] Telescope right ascension (ICRS)')
        edit_head(header, 'DEC', value=dec_icrs,
                  comments='[deg] Telescope declination (ICRS)')


        # this keyword is added later on by Astrometry.net
        #edit_head(header, 'EQUINOX', value=2000.0,
        #          comments='[yr] Equatorial coordinates definition')


        # for ML1, the RA and DEC were incorrectly referring to the
        # subsequent image up to 9 Feb 2019 (except when put in by
        # hand with the sexagesimal notation, in which case keywords
        # RA-TEL and DEC-TEL are not present in the header); for these
        # images we replace the RA and DEC by the RA-REF and DEC-REF
        if tel=='ML1':
            tcorr_radec = Time('2019-02-09T00:00:00', format='isot').mjd
            if (mjd_obs < tcorr_radec and 'RA-REF' in header and
                'DEC-REF' in header):
                ra_icrs = Angle(header['RA-REF'], unit=u.hour).degree
                dec_icrs = Angle(header['DEC-REF'], unit=u.deg).degree

                # RA-REF and DEC-REF are assumed to be J2000/ICRS,
                # so no need to convert from JNOW
                edit_head(header, 'RA', value=ra_icrs,
                          comments='[deg] Telescope right ascension (=RA-REF)')
                edit_head(header, 'DEC', value=dec_icrs,
                          comments='[deg] Telescope declination (=DEC-REF)')

        # determine airmass
        airmass, alt, az = get_airmass(ra_icrs, dec_icrs, date_obs_str, lat, lon,
                                       height, get_altaz=True)
        edit_head(header, 'AIRMASS', value=float(airmass),
                  comments='Airmass (using RA, DEC, DATE-OBS)')

        # ALTITUDE and AZIMUTH not always present in raw header, so
        # add the values calculated using [get_airmass] above
        if 'ALTITUDE' in header:
            if not silent:
                log.info ('ALTITUDE in raw header: {:.2f}, value calculated '
                          'using [get_airmass]: {:.2f}; adopting the latter'
                          .format(header['ALTITUDE'], alt))

        edit_head(header, 'ALTITUDE', value=float(alt),
                  comments='[deg] Telescope altitude (using RA/DEC)')

        if 'AZIMUTH' in header:
            if not silent:
                log.info ('AZIMUTH in raw header: {:.2f}, value calculated '
                          'using [get_airmass]: {:.2f}; adopting the latter'
                          .format(header['AZIMUTH'], az))

        edit_head(header, 'AZIMUTH', value=float(az),
                  comments='[deg] Telescope azimuth (N=0;E=90, using RA/DEC)')



    edit_head(header, 'SITELAT',  value=lat, comments='[deg] Site latitude')
    edit_head(header, 'SITELONG', value=lon, comments='[deg] Site longitude')
    edit_head(header, 'ELEVATIO', value=height, comments='[m] Site elevation')


    # add some Moon parameters
    location = EarthLocation(lat=lat, lon=lon, height=height)
    coords_moon = get_body('moon', date_obs, location)
    moon_ra = coords_moon.ra.deg
    moon_dec = coords_moon.dec.deg

    if 'RA' in header and 'DEC' in header:
        coords = SkyCoord (ra_icrs, dec_icrs, unit='deg', frame='icrs')
        # separation between image RA/DEC and moon
        moon_sep = coords_moon.separation(coords, origin_mismatch='ignore').deg
        # position angle of the moon with respect to the image
        moon_pa = coords_moon.position_angle(coords).deg
        # above actually provides the position angle of image wrt the moon,
        # so switch it around
        moon_pa = (moon_pa - 180) % 360
        # N.B.: both separation and position_angle method are not
        # symmetric if coords_moon and coords are switched around,
        # because the moon is in the Geocentric frame, while coords
        # are in the ICRS frame (see
        # https://docs.astropy.org/en/latest/coordinates/common_errors.html)
    else:
        moon_sep = 'None'
        moon_pa = 'None'


    # alt-az
    moon_altaz = coords_moon.transform_to(AltAz(obstime=date_obs,
                                                location=location))
    # moon illumination
    # using astroplan
    #moon_ill = moon.moon_illumination (date_obs)
    # using ephem
    moon_ill = ephem.Moon(date_obs.datetime).moon_phase


    edit_head(header, 'MOON-RA', value=moon_ra,
              comments='[deg] Moon right ascension (GCRS)')
    edit_head(header, 'MOON-DEC', value=moon_dec,
              comments='[deg] Moon declination (GCRS)')
    edit_head(header, 'MOON-SEP', value=moon_sep,
              comments='[deg] Moon separation to telescope RA/DEC')
    edit_head(header, 'MOON-PA', value=moon_pa,
              comments='[deg] Moon pos. angle wrt telescope RA/DEC')
    edit_head(header, 'MOON-ALT', value=moon_altaz.alt.deg,
              comments='[deg] Moon altitude')
    edit_head(header, 'MOON-AZ', value=moon_altaz.az.deg,
              comments='[deg] Moon azimuth (N=0;E=90)')
    edit_head(header, 'MOON-ILL', value=moon_ill,
              comments='Moon illumination fraction')


    # add some Sun parameters
    coords_sun = get_body('sun', date_obs, location)
    sun_altaz = coords_sun.transform_to(AltAz(obstime=date_obs,
                                              location=location))

    edit_head(header, 'SUN-ALT', value=sun_altaz.alt.deg,
              comments='[deg] Sun altitude')
    edit_head(header, 'SUN-AZ', value=sun_altaz.az.deg,
              comments='[deg] Sun azimuth (N=0;E=90)')


    # update -REF and -TEL of RAs and DECs; if the -REFs do not exist
    # yet, create them with 'None' values - needed for the Database
    edit_head(header, 'RA-REF', value='None',
              comments='Requested right ascension')
    edit_head(header, 'DEC-REF', value='None', comments='Requested declination')

    # do not consider RA-TEL and DEC-TEL anymore for the reduced header
    if False:

        if 'RA-TEL' in header and 'DEC-TEL' in header:

            ra_tel_deg = float(header['RA-TEL'])
            dec_tel_deg = float(header['DEC-TEL'])

            # convert RA-TEL value from hours to degrees; assume that
            # until 15-03-2019 RA-TEL was in degrees, afterwards in hours,
            # although for many bias and other calibration frames it was
            # still in degrees after this date
            if tel=='ML1':
                tcorr = Time('2019-03-16T12:00:00', format='isot').mjd
                if mjd_obs > tcorr and ra_tel_deg < 24:
                    ra_tel_deg *= 15.

            # assuming RA-TEL,DEC-TEL are JNOW, convert them to J2000/ICRS
            equinox = Time(mjd_obs, format='mjd').jyear_str
            ra_tel_icrs, dec_tel_icrs = jnow2icrs (ra_tel_deg, dec_tel_deg,
                                                   equinox)

            edit_head(header, 'RA-TEL', value=ra_tel_icrs,
                      comments='[deg] Telescope right ascension (ICRS)')
            edit_head(header, 'DEC-TEL', value=dec_tel_icrs,
                      comments='[deg] Telescope declination (ICRS)')

        else:

            # if not available in raw header, add them with 'None' values
            edit_head(header, 'RA-TEL', value='None',
                      comments='[deg] Telescope right ascension')
            edit_head(header, 'DEC-TEL', value='None',
                      comments='[deg] Telescope declination')


    # now that RA/DEC are (potentially) corrected, determine local
    # hour angle; this keyword was in the raw image header for a while,
    # but seems to have disappeared during the 2nd half of March 2019
    if 'RA' in header:

        # first convert ra_icrs to ra_jnow
        equinox = Time(mjd_obs, format='mjd').jyear_str
        ra_jnow, dec_jnow = jnow2icrs (ra_icrs, dec_icrs, equinox,
                                       icrs2jnow=True)
        # local hour angle
        lha_deg = lst_deg - ra_jnow

        # PaulG noticed some lha_deg values are between -340 and -360
        # and between +340 and +360:
        if lha_deg < -180:
            lha_deg += 360
        elif lha_deg >= 180:
            lha_deg -= 360

        #edit_head(header, 'HA', value=lha_deg,
        #          comments='[deg] Local hour angle (=LST-RA_Jnow)')
        lha_hr = lha_deg / 15
        edit_head(header, 'HA', value=lha_hr,
                  comments='[hr] Local hour angle (=LST-RA_Jnow)')


    # Weather headers required for Database
    edit_head(header, 'CL-BASE',  value='None', dtype=float,
              comments='[m] Reinhardt cloud base altitude')
    edit_head(header, 'RH-MAST',  value='None', dtype=float,
              comments='Vaisala RH mast')
    edit_head(header, 'RH-DOME',  value='None', dtype=float,
              comments='CilSense2 RH dome')
    edit_head(header, 'RH-AIRCO', value='None', dtype=float,
              comments='CilSense3 RH server room airco')
    edit_head(header, 'RH-PIER',  value='None', dtype=float,
              comments='CilSense1 RH pier')
    edit_head(header, 'PRESSURE', value='None', dtype=float,
              comments='[hPa] Vaisala pressure mast')
    edit_head(header, 'T-PIER',   value='None', dtype=float,
              comments='[C] CilSense1 temperature pier')
    edit_head(header, 'T-DOME',   value='None', dtype=float,
              comments='[C] CilSense2 temperature dome')
    edit_head(header, 'T-ROOF',   value='None', dtype=float,
              comments='[C] Reinhardt temperature roof')
    edit_head(header, 'T-AIRCO',  value='None', dtype=float,
              comments='[C] CilSense3 temperature server room airco')
    edit_head(header, 'T-MAST',   value='None', dtype=float,
              comments='[C] Vaisala temperature mast')
    edit_head(header, 'T-STRUT',  value='None', dtype=float,
              comments='[C] Temperature carbon strut between M1 and M2')
    edit_head(header, 'T-CRING',  value='None', dtype=float,
              comments='[C] Temperature main carbon ring around M1')
    edit_head(header, 'T-SPIDER', value='None', dtype=float,
              comments='[C] Temperature carbon spider above M2')
    edit_head(header, 'T-FWN',    value='None', dtype=float,
              comments='[C] Temperature filter wheel housing North')
    edit_head(header, 'T-FWS',    value='None', dtype=float,
              comments='[C] Temperature filter wheel housing South')
    edit_head(header, 'T-M2HOLD', value='None', dtype=float,
              comments='[C] Temperature aluminium M2 holder')
    edit_head(header, 'T-GUICAM', value='None', dtype=float,
              comments='[C] Temperature guide camera')
    edit_head(header, 'T-M1',     value='None', dtype=float,
              comments='[C] Temperature backside M1')
    edit_head(header, 'T-CRYWIN', value='None', dtype=float,
              comments='[C] Temperature Cryostat window')
    edit_head(header, 'T-CRYGET', value='None', dtype=float,
              comments='[K] Temperature Cryostat getter')
    edit_head(header, 'T-CRYCP',  value='None', dtype=float,
              comments='[K] Temperature Cryostat cold plate')
    edit_head(header, 'PRES-CRY', value='None', dtype=float,
              comments='[bar] Cryostat vacuum pressure')
    edit_head(header, 'WINDAVE',  value='None', dtype=float,
              comments='[km/h] Vaisala wind speed mast')
    edit_head(header, 'WINDGUST', value='None', dtype=float,
              comments='[km/h] Vaisala wind gust mast')
    edit_head(header, 'WINDDIR',  value='None', dtype=float,
              comments='[deg] Vaisala wind direction mast')


    edit_head(header, 'FILTER', comments='Filter')
    if tel=='ML1':
        # for some 2017 data, 'VR' was used for 'q':
        if header['FILTER'] == 'VR':
            edit_head(header, 'FILTER', value='q')

        # for ML1: filter is incorrectly identified in the header for data
        # taken with Abot from 2017-11-19T00:00:00 until 2019-01-13T15:00:00.
        # Divided this time in a transition period (from 2017-11-19T00:00:00
        # to 2018-02-24T23:59:59) where some data was taken with Abot and some
        # was taken manually, and a period in which all data was taken with
        # Abot (from 2018-02-25T00:00:00 to 2019-01-13T15:00:00). Data that is
        # taken manually does not need to be corrected for filter. For the data
        # taken with Abot, this is the correct mapping,
        # correct filter=filt_corr[old filter], as determined by PaulG, Oliver
        # & Danielle (see also Redmine bug #281)
        filt_corr = {'u':'q',
                     'g':'r',
                     'q':'i',
                     'r':'g',
                     'i':'z',
                     'z':'u'}

        transition_mjd = Time(['2017-11-19T00:00:00', '2018-02-24T23:59:59'],
                              format='isot').mjd
        tcorr_mjd = Time(['2018-02-25T00:00:00', '2019-01-13T15:00:00'],
                         format='isot').mjd
        if mjd_obs >= transition_mjd[0] and mjd_obs <= transition_mjd[1]:
            if 'OBSERVER' in header and header['OBSERVER'].lower()=='abot':
                filt_old = header['FILTER']
                edit_head(header, 'FILTER', value=filt_corr[filt_old],
                          comments='Filter (corrected)')
        elif mjd_obs >= tcorr_mjd[0] and mjd_obs <= tcorr_mjd[1]:
            filt_old = header['FILTER']
            edit_head(header, 'FILTER', value=filt_corr[filt_old],
                      comments='Filter (corrected)')

    edit_head(header, 'CCD-ID',   value='None', dtype=str,
              comments='CCD camera ID')
    edit_head(header, 'CONTROLL', value='None', dtype=str,
              comments='CCD controller')
    edit_head(header, 'DETSPEED', value='None', dtype=int,
              comments='[kHz] Detector read speed')
    edit_head(header, 'CCD-NW', value=get_par(set_bb.nx,tel), dtype=int,
              comments='Number of channels in width')
    edit_head(header, 'CCD-NH', value=get_par(set_bb.ny,tel), dtype=int,
              comments='Number of channels in height')
    edit_head(header, 'INSTRUME', value='None', dtype=str,
              comments='Instrument name')
    edit_head(header, 'FOCUSPOS', value='None', dtype=int,
              comments='[micron] Focuser position')

    if tel=='ML1':
        origin = 'MeerLICHT-1,Sutherland'
        mpc_code = 'L66'
        telescop = 'MeerLICHT-1'
    if tel[0:2]=='BG':
        origin = 'BlackGEM, La Silla, ESO'
        mpc_code = 'X17'
        telescop = 'BlackGEM-{}'.format(tel[2:])

    edit_head(header, 'ORIGIN', value=origin, comments='Origin of data')
    edit_head(header, 'MPC-CODE', value=mpc_code, comments='MPC Observatory code')
    edit_head(header, 'TELESCOP', value=telescop, comments='Telescope ID')


    edit_head(header, 'IMAGETYP', dtype=str, comments='Image type')
    edit_head(header, 'OBJECT',   dtype=str,
              comments='Name of object observed (field ID)')

    if header['IMAGETYP'].lower()=='object':
        if 'FIELD_ID' in header:
            obj = header['FIELD_ID']
        else:
            obj = header['OBJECT']

        edit_head(header, 'OBJECT', value='{:0>5}'.format(obj),
                  comments='Name of object observed (field ID)')


    # do not add ARCFILE name for the moment
    #arcfile = '{}.{}'.format(tel, date_obs_str)
    #edit_head(header, 'ARCFILE', value=arcfile, comments='Archive filename')
    edit_head(header, 'ORIGFILE', value=filename.split('/')[-1].split('.fits')[0],
              comments='ABOT name')


    edit_head(header, 'OBSERVER', value='None', dtype=str,
              comments='Robotic observations software and PC ID')
    edit_head(header, 'ABOTVER',  value='None', dtype=str,
              comments='ABOT version')
    edit_head(header, 'PROGNAME', value='None', dtype=str,
              comments='Program name')
    edit_head(header, 'PROGID',   value='None', dtype=str,
              comments='Program ID')
    edit_head(header, 'GUIDERST', value='None', dtype=str,
              comments='Guider status')
    edit_head(header, 'GUIDERFQ', value='None', dtype=float,
              comments='[Hz] Guide loop frequency')
    edit_head(header, 'TRAKTIME', value='None', dtype=float,
              comments='[s] Autoguider exposure time during imaging')
    edit_head(header, 'ADCX',     value='None', dtype=float,
              comments='[mm] Position offset ADC lens in x')
    edit_head(header, 'ADCY',     value='None', dtype=float,
              comments='[mm] Position offset ADC lens in y')


    # remove the following keywords:
    keys_2remove = ['FILTWHID', 'FOC-ID', 'EXPOSURE', 'END-OBS', 'FOCUSMIT',
                    'FOCUSAMT', 'OWNERGNM', 'OWNERGID', 'OWNERID',
                    'AZ-REF', 'ALT-REF', 'CCDFULLH', 'CCDFULLW', 'RADECSYS',
                    'RA-TEL', 'DEC-TEL']

    if 'BG' in tel:
        # no dome orientation for BlackGEM, but just keep it with
        # value 'None'
        #keys_2remove.append('DOMEAZ')
        pass


    for key in keys_2remove:
        if key in header:
            header.remove(key, remove_all=True)
            if not silent:
                log.info ('removing keyword {}'.format(key))


    # put some order in the header
    keys_sort = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                 'BUNIT', 'BSCALE', 'BZERO',
                 'XBINNING', 'YBINNING',
                 'ALTITUDE', 'AZIMUTH', 'DOMEAZ', 'RADESYS', 'EPOCH',
                 #'RA', 'RA-REF', 'RA-TEL', 'DEC', 'DEC-REF', 'DEC-TEL',
                 'RA', 'RA-REF', 'DEC', 'DEC-REF',
                 'HA', 'FLIPSTAT', 'ISTRACKI',
                 'OBJECT', 'IMAGETYP', 'FILTER', 'EXPTIME',
                 'ACQSTART', 'ACQEND', 'GPSSTART', 'GPSEND', 'GPS-SHUT',
                 'DATE-OBS', 'MJD-OBS', 'LST', 'UTC', 'TIMESYS',
                 'SITELAT', 'SITELONG', 'ELEVATIO', 'AIRMASS',
                 'MOON-RA', 'MOON-DEC', 'MOON-SEP', 'MOON-PA',
                 'MOON-ALT', 'MOON-AZ', 'MOON-ILL', 'SUN-ALT', 'SUN-AZ',
                 'SET-TEMP', 'CCD-TEMP', 'CCD-ID', 'CONTROLL', 'DETSPEED',
                 'CCD-NW', 'CCD-NH', 'FOCUSPOS',
                 'ORIGIN', 'MPC-CODE', 'TELESCOP', 'INSTRUME',
                 'OBSERVER', 'ABOTVER', 'PROGNAME', 'PROGID', 'ORIGFILE',
                 'GUIDERST', 'GUIDERFQ', 'TRAKTIME', 'ADCX', 'ADCY',
                 'CL-BASE', 'RH-MAST', 'RH-DOME', 'RH-AIRCO', 'RH-PIER',
                 'PRESSURE', 'T-PIER', 'T-DOME', 'T-ROOF', 'T-AIRCO', 'T-MAST',
                 'T-STRUT', 'T-CRING', 'T-SPIDER', 'T-FWN', 'T-FWS', 'T-M2HOLD',
                 'T-GUICAM', 'T-M1', 'T-CRYWIN', 'T-CRYGET', 'T-CRYCP',
                 'PRES-CRY', 'WINDAVE', 'WINDGUST', 'WINDDIR']


    # create empty header
    header_sort = fits.Header()
    for nkey, key in enumerate(keys_sort):
        if key in header:
            # append key, value and comments to new header
            header_sort.append((key, header[key], header.comments[key]))
        else:
            if not silent:
                log.warning ('keyword {} not in header'.format(key))


    return header_sort


################################################################################

def jnow2icrs (ra_in, dec_in, equinox, icrs2jnow=False):

    """function to convert RA and DEC coordinates in decimal degrees to
       ICRS, or back using icrs2jnow=True

    N.B.: Steven added this conversion to the Hydra (on 7 March 2024
          for BG2 and BG4, and on 15 Nov 2024 for BG3), which is
          probably the correct way of transforming between mount
          coordinates and J2000:

    # --------------------------------------------------------------------------
    # SB added on 7 March 2024 - reporting in J2000 instead of Jnow
     # To do it more correctly, need to work in ICRS rather than J2000.
   # Current version also only takes precession into account
    current_epoch = time.localtime().tm_year+time.localtime().tm_yday/365.25
    pointing_J2000 = coords.EquatorialCoordinatesEquinox(self.pointing.ra,self.pointing.dec,epoch=current_epoch) # still in Jnow
    pointing_J2000.transformToEpoch(2000) # now in J2000
    ra_value = str(pointing_J2000.ra.h)
    dec_value = str(pointing_J2000.dec.d)
    # --------------------------------------------------------------------------


    """

    if icrs2jnow:
        coords = SkyCoord(ra_in*u.degree, dec_in*u.degree, frame='icrs')
        jnow = FK5(equinox=equinox)
        coords_out = coords.transform_to(jnow)

    else:
        coords_out = SkyCoord(ra_in*u.degree, dec_in*u.degree, frame='fk5',
                              equinox=equinox).icrs

    return coords_out.ra.value, coords_out.dec.value


################################################################################

def define_sections (data_shape, xbin=1, ybin=1, tel=None):

    """Function that defines and returns [chan_sec], [data_sec],
    [os_sec_hori], [os_sec_vert] and [data_sec_red], based on the
    number of channels in x and y and the sizes of the channel data
    sections defined in the blackbox settings file [set_blackbox], and
    the input shape (ysize, xsize) that define the total size of the
    raw image.

    """

    ysize, xsize = data_shape
    ny = get_par(set_bb.ny,tel)
    nx = get_par(set_bb.nx,tel)
    dy = ysize // ny
    dx = xsize // nx

    ysize_chan = get_par(set_bb.ysize_chan,tel) // ybin
    xsize_chan = get_par(set_bb.xsize_chan,tel) // xbin
    ysize_os = (ysize-ny*ysize_chan) // ny
    xsize_os = (xsize-nx*xsize_chan) // nx

    # the sections below are defined such that e.g. chan_sec[0] refers
    # to all pixels of the first channel, where the channel indices
    # are currently defined to be located on the CCD as follows:
    #
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]

    # channel section slices including overscan; shape=(16,2)
    chan_sec = tuple([(slice(y,y+dy), slice(x,x+dx))
                      for y in range(0,ysize,dy) for x in range(0,xsize,dx)])

    # channel data section slices; shape=(16,2)
    data_sec = tuple([(slice(y,y+ysize_chan), slice(x,x+xsize_chan))
                      for y in range(0,ysize,dy+ysize_os) for x in range(0,xsize,dx)])

    # channel vertical overscan section slices; shape=(16,2)
    # cut off [ncut] pixels to avoid including pixels on the edge of the
    # overscan that are contaminated with flux from the image
    # and also discard last column as can have high value
    ncut = 5
    ncut_vert = max(ncut // xbin, 1)
    os_sec_vert = tuple([(slice(y,y+dy), slice(x+xsize_chan+ncut_vert,x+dx-1))
                         for y in range(0,ysize,dy) for x in range(0,xsize,dx)])

    # channel horizontal overscan sections; shape=(16,2)
    # cut off [ncut] pixels to avoid including pixels on the edge of the
    # overscan that are contaminated with flux from the image
    ncut = 10
    ncut_hori = max(ncut // ybin, 1)
    ysize_os_cut = ysize_os - ncut_hori
    os_sec_hori = tuple([(slice(y,y+ysize_os_cut), slice(x,x+dx))
                         for y in range(dy-ysize_os_cut,dy+ysize_os_cut,ysize_os_cut)
                         for x in range(0,xsize,dx)])

    # channel reduced data section slices; shape=(16,2)
    data_sec_red = tuple([(slice(y,y+ysize_chan), slice(x,x+xsize_chan))
                          for y in range(0,ysize-ny*ysize_os,ysize_chan)
                          for x in range(0,xsize-nx*xsize_os,xsize_chan)])


    return chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red


################################################################################

def os_corr (data, header, imgtype, xbin=1, ybin=1, data_limit=2000, tel=None):

    """Function that corrects [data] for the overscan signal in the
       vertical and horizontal overscan strips. The definitions of the
       different data/overscan/channel sections are taken from
       [set_blackbox].  The function returns a data array that
       consists of the data sections only, i.e. without the overscan
       regions. The [header] is update in place.

       Updated in Jan 2025 to replace original function os_corr();
       main change is that instead of column-by-column subtraction of
       horizontal overscan section, a spline fit is performed over the
       range x=0:150 and a polynomial fit over the range
       x=150:1320. For BlackGEM, any column in the horizontal overscan
       section that has a saturated pixel in the data section will not
       be fit; this is to avoid oversubtracting columns affected by
       saturated stars leaking flux into the overscan section.
    """

    if get_par(set_zogy.timing,tel):
        t = time.time()


    # to be able to detect the interpolate.UnivariateSpline user
    # warning in a try-except statement
    warnings.filterwarnings('error')


    # infer channel, data and overscan sections
    chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red = (
        define_sections(np.shape(data), xbin=xbin, ybin=ybin, tel=tel))


    # number of data columns and rows in the channel (without overscans)
    ncols = get_par(set_bb.xsize_chan,tel) // xbin
    nrows = get_par(set_bb.ysize_chan,tel) // ybin

    # initialize output data array (without overscans)
    ny = get_par(set_bb.ny,tel)
    nx = get_par(set_bb.nx,tel)
    data_out = np.zeros((nrows*ny, ncols*nx), dtype='float32')

    # and arrays to calculate average means and stds over all channels
    nchans = np.shape(data_sec)[0]
    mean_vos = np.zeros(nchans)
    std_vos = np.zeros(nchans)
    median_hos = np.zeros(nchans)

    vos_poldeg = get_par(set_bb.voscan_poldeg,tel)
    nrows_chan = np.shape(data[chan_sec[0]])[0]
    y_vos = np.arange(nrows_chan)

    nrows_overlap = nrows_chan - nrows


    # channel saturation levels
    satlevel = np.array(get_par(set_bb.satlevel,tel))
    gain = np.array(get_par(set_bb.gain,tel))
    satlevel_electrons = satlevel * gain


    for i_chan in range(nchans):


        # -----------------
        # vertical overscan
        # -----------------

        # first subtract a low-order polynomial fit to the clipped
        # mean (not median, because input pixels are still integers!)
        # of the vertical overcan section from the entire channel

        # determine clipped mean for each row
        data_vos = data[os_sec_vert[i_chan]]
        mean_vos_col, __, __ = sigma_clipped_stats(
            data_vos, axis=1, mask_value=0, cenfunc='mean')


        # fit low order polynomial to vertical overscan, avoiding
        # outlying values due to e.g. very bright star that bleeds
        # into the vertical overscan sections at the bottom of the
        # image for channels 1-8, and at the top for channels 9-16
        try:
            polyfit_ok = True
            nsigma = 5
            mean, median, stddev = sigma_clipped_stats(
                mean_vos_col, sigma=nsigma, cenfunc='mean')
            if stddev==0:
                mask_fit = np.ones(nrows_chan, dtype=bool)
            else:
                mask_fit = (np.abs(mean_vos_col-mean)/stddev <= nsigma)

            # do not fit overlap of vertical and horizontal overscans,
            # at least for channel 9 of BG2, this leads to trouble;
            # N.B.: overscan sections of channels 9-16 are mirrored in
            # y with respect to channels 1-8
            if i_chan < 8:
                mask_fit[nrows:] = False
            else:
                mask_fit[:nrows_overlap] = False

            # perform fit
            p = np.polyfit(y_vos[mask_fit], mean_vos_col[mask_fit], vos_poldeg)


        except Exception as e:
            polyfit_ok = False
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during polynomial fit to '
                          'channel {} vertical overscan'.format(i_chan))


        # add fit coefficients to image header
        for nc in range(len(p)):
            p_reverse = p[::-1]
            if np.isfinite(p_reverse[nc]):
                value = p_reverse[nc]
            else:
                value = 'None'
            header['BIAS{}A{}'.format(i_chan+1, nc)] = (
                value, '[e-] channel {} vert. overscan A{} polyfit coeff'
                .format(i_chan+1, nc))

        # fit values
        fit_vos_col = np.polyval(p, y_vos)
        if not np.all(np.isfinite(fit_vos_col)):
            polyfit_ok = False

        header['VFITOK{}'.format(i_chan+1)] = (
            polyfit_ok, 'channel {} vert. overscan polyfit finite?'
            .format(i_chan+1))


        # if polynomial fit is reliable, subtract this off the entire
        # channel; otherwise subtract the nanmedian of the vos row
        # means
        if polyfit_ok:
            mean_vos[i_chan] = np.mean(fit_vos_col)
            data[chan_sec[i_chan]] -= fit_vos_col.reshape(nrows_chan,1)
        else:
            mean_vos[i_chan] = np.nanmedian(mean_vos_col)
            data[chan_sec[i_chan]] -= mean_vos[i_chan]


        # mean levels of vertical and horizontal overscans do not
        # always correspond, e.g. for channel 9 of BG2 there is clear
        # difference of ~5-10e-; correct for this level offset by
        # determining the sigma-clipped mean value of the
        # vos-subtracted data on the right side of the horizontal
        # overscan
        dlevel, __, __ = sigma_clipped_stats (
            data[os_sec_hori[i_chan]][:,ncols-300:ncols], cenfunc='mean')
        #log.info ('dlevel: {}'.format(dlevel))
        data[os_sec_hori[i_chan]] -= dlevel


        # determine std of overscan subtracted vos:
        __, __, std_vos[i_chan] = sigma_clipped_stats(
            data[os_sec_vert[i_chan]], mask_value=0, cenfunc='mean')



        # -------------------
        # horizontal overscan
        # -------------------

        # determine the running clipped mean of the overscan using all
        # values for [ncols] columns
        data_hos = data[os_sec_hori[i_chan]][:,:ncols]


        if tel=='ML1':

            # replace very high values (due to bright objects on edge
            # of channel) with function [inter_pix] in zogy.py
            mask_hos = (data_hos > data_limit)


            # if it concerns a single column on its own that is covering
            # at least half the overscan height, unmask that column
            mask_x = (np.sum(mask_hos, axis=0) > 0.5 * mask_hos.shape[0])


            # opening with [True,True] will remove single True value from
            # mask_x but will leave multiple adjacent True values be
            mask_x_open = ndimage.binary_opening(mask_x, structure=np.ones(2))


            # identify any single True values that were removed
            mask_x_restore = np.logical_xor (mask_x, mask_x_open)


            # reset mask_hos for that/those column(s)
            mask_hos[:,mask_x_restore] = False


            # add couple of pixels connected to this mask
            mask_hos = ndimage.binary_dilation(mask_hos,
                                               structure=np.ones((3,3)).astype('bool'),
                                               iterations=2)

        else:

            # for BlackGEM identify columns where one/a few or more
            # pixels in data section is at or above the saturation
            # level in the vicinity of the overscan section; those
            # columns tend to leak flux into the overscan section;
            # make a distinction between lightly and heavily saturated
            # stars
            mask_hos = np.zeros_like(data_hos, dtype=bool)
            ypix_lim = {'BG2':(2640,5280), 'BG3':(1320,2640), 'BG4':(1320,2640)}
            if i_chan >= 8:
                ypix_range1 = range(0, ypix_lim[tel][0])
                ypix_range2 = range(0, ypix_lim[tel][1])
            else:
                ypix_range1 = range(nrows-ypix_lim[tel][0], nrows)
                ypix_range2 = range(nrows-ypix_lim[tel][1], nrows)


            # define row pixels that were affected by nearby saturated
            # stars
            mask_sat_row = (np.sum(data[data_sec[i_chan]][ypix_range1,:] >=
                                   0.9*satlevel_electrons[i_chan], axis=0) >= 3)
            # for heavily saturated stars
            mask_sat_row |= (np.sum(data[data_sec[i_chan]][ypix_range2,:] >=
                                    0.9*satlevel_electrons[i_chan], axis=0) >= 10)

            # update mask_hos
            mask_hos[:] |= mask_sat_row



        # determine clipped mean for each column, first defining
        # masked array
        data_hos_ma = np.ma.masked_array(data_hos, mask=mask_hos)
        # sigma_clip
        nsigma = 2.5
        data_hos_ma = sigma_clip (data_hos_ma, axis=0, cenfunc='mean',
                                  sigma=nsigma)
        # number of rows in each column, used below
        # to estimate the error in the mean
        nvalues_col = np.sum(~data_hos_ma.mask, axis=0)
        # clipped mean and std
        mean_hos = np.nanmean(data_hos_ma, axis=0)
        std_hos = np.nanstd(data_hos_ma, axis=0, ddof=1)
        # mask of columns with more than 1 valid value after
        # clipping
        mask_valid = (nvalues_col > 1)


        # spline fit errors
        xcol = np.arange(len(mean_hos)) + 1
        err_hos = np.zeros_like(mean_hos)
        err_hos[mask_valid] = (std_hos[mask_valid] /
                               np.sqrt(nvalues_col[mask_valid]))

        # weights
        weights = np.zeros_like(err_hos)
        mask_nonzero = (err_hos != 0)
        weights[mask_nonzero] = 1/err_hos[mask_nonzero]
        # do not fit spline for first three columns if all of them are
        # valid
        if np.all(mask_valid[0:3]):
            weights[0:3] = 0


        # index up to which spline is used, and afterwards the
        # polynomial fit
        idx_switch = 150
        overlap = 30

        # perform spline fit
        idx_fit = np.arange(idx_switch + overlap)
        npoints = np.sum(mask_valid[idx_fit] & mask_nonzero[idx_fit])

        # this make_splrep() is giving occasional issues, despite this
        # method being recommended over the UnivariateSpline method
        # below; for the moment use the latter
        #
        #fit = interpolate.make_splrep(xcol[idx_fit][mask_valid[idx_fit]],
        #                              mean_hos[idx_fit][mask_valid[idx_fit]],
        #                              w=weights[idx_fit][mask_valid[idx_fit]],
        #                              k=2, s=npoints)
        try:
            m = mask_valid

            # to avoid spline fitting single high points, median
            # smooth the points to be fit
            mean_hos_2fit = mean_hos[idx_fit][m[idx_fit]]
            dpix = 1
            nfit = len(mean_hos_2fit)
            mean_hos_2fit[3:] = [np.ma.median(mean_hos_2fit[max(k-dpix,3):
                                                            min(k+dpix+1,nfit)])
                                 for k in range(3,nfit)]

            splfit = interpolate.UnivariateSpline(xcol[idx_fit][m[idx_fit]],
                                                  mean_hos_2fit,
                                                  #mean_hos[idx_fit][m[idx_fit]],
                                                  w=weights[idx_fit][m[idx_fit]],
                                                  k=2, s=npoints)
        except UserWarning as uw:
            log.warning ('problem with fitting spline to channel {} overscan'
                         '; trying again with k=3 and 50% higher smoothing '
                         'parameter s; warning: {})'.format(i_chan+1, uw))
            splfit = interpolate.UnivariateSpline(xcol[idx_fit][m[idx_fit]],
                                                  mean_hos_2fit,
                                                  #mean_hos[idx_fit][m[idx_fit]],
                                                  w=weights[idx_fit][m[idx_fit]],
                                                  k=3, s=1.5*npoints)



        # lowish order polynomial fit for data beyond idx_switch
        mask_valid_poly = mask_valid.copy()
        mask_valid_poly[0:idx_switch-overlap] = False

        # clean from high/low values
        nsigma = 5
        mean_hos_poly = mean_hos[mask_valid_poly]
        mean, median, stddev = sigma_clipped_stats(
            mean_hos_poly, sigma=nsigma, cenfunc='mean')
        if stddev==0:
            mask_fit_tmp = np.ones(len(mean_hos_poly), dtype=bool)
        else:
            mask_fit_tmp = (np.abs(mean_hos_poly-mean)/stddev <= nsigma)

        mask_valid_poly[mask_valid_poly] = mask_fit_tmp


        if not (tel=='BG2' and i_chan==8):

            for it in range(3):
                p = np.polyfit(xcol[mask_valid_poly],
                               mean_hos[mask_valid_poly], 7)
                fit_poly = np.polyval(p, xcol)
                # reject data
                #log.info ('it: {}, np.sum(mask_valid_poly): {}'
                #          .format(it, np.sum(mask_valid_poly)))
                mask_valid_poly &= (np.abs(fit_poly - mean_hos) <= 3*err_hos)


            # overscan array to be subtracted
            oscan = fit_poly

        else:

            # for channel 9 of BG2, need to split polynomial fit into
            # two pieces, separated at column x=654
            idx_split = 654

            mask_fit = mask_valid_poly.copy()
            mask_fit[idx_split:] = False
            for it in range(3):
                p = np.polyfit(xcol[mask_fit], mean_hos[mask_fit], 5)
                fit1_poly = np.polyval(p, xcol)
                # reject data
                #log.info ('it: {}, np.sum(mask_fit) 1: {}'
                #          .format(it, np.sum(mask_fit)))
                mask_fit &= (np.abs(fit1_poly - mean_hos) <= 3*err_hos)


            mask_fit = mask_valid_poly.copy()
            mask_fit[:idx_split] = False
            for it in range(3):
                p = np.polyfit(xcol[mask_fit], mean_hos[mask_fit], 5)
                fit2_poly = np.polyval(p, xcol)
                # reject data
                #log.info ('it: {}, np.sum(mask_fit) 2: {}'
                #          .format(it, np.sum(mask_fit)))
                mask_fit &= (np.abs(fit2_poly - mean_hos) <= 3*err_hos)


            # overscan array to be subtracted
            oscan = fit1_poly
            oscan[idx_split:] = fit2_poly[idx_split:]




        # replace columns up to idx_switch with spline fit
        oscan[0:idx_switch] = splfit(xcol[0:idx_switch])
        # for first couple of columns, adopt mean if valid
        oscan[0:3][mask_valid[0:3]] = mean_hos[0:3][mask_valid[0:3]]


        # CHECK!!! - test: use original subtraction of
        # column-by-column for "spline" columns with no/few saturated
        # pixels, defined in mask_sat_row already determined above;
        # make sure there are sufficient values in the overscan column
        # through mask_valid
        mask_usemean = ~mask_sat_row & mask_valid
        # only for column up to idx_switch
        mask_usemean[idx_switch:] = False
        #oscan_alt = oscan.copy()
        oscan[mask_usemean] = mean_hos[mask_usemean]


        if False:

            #if tel=='BG2' and i_chan==8:
            #    for i in range(idx_split-10,idx_split+10):
            #        log.info ('i: {}, oscan[i]: {}, oscan_alt[i]: {}'
            #                  .format(i, oscan[i], oscan_alt[i]))


            plt.errorbar (xcol, mean_hos, yerr=err_hos, color='k',
                          linestyle="None", capsize=2)
            plt.plot (xcol, mean_hos, 'k.', label='mean_hos')
            plt.plot (xcol, oscan_alt, '-', color='purple', label='overscan alt')
            plt.plot (xcol, oscan, 'g-', label='overscan used')
            plt.plot (xcol[:idx_switch], splfit(xcol[:idx_switch]), 'b-',
                      label='spline fit')
            plt.ylim (max(-50, np.amin(mean_hos)), abs(1.2*np.amax(oscan+10)))
            plt.title('channel: {}'.format(i_chan+1))
            plt.legend()
            plt.savefig('hos_chan{:02}.pdf'.format(i_chan+1))
            plt.xlim (0,idx_switch+30)
            plt.savefig('hos_chan{:02}_zoom.pdf'.format(i_chan+1))
            plt.close()




        # finally, subtract horizontal overscan from data section
        data[data_sec[i_chan]] -= oscan

        # place into [data_out]
        data_out[data_sec_red[i_chan]] = data[data_sec[i_chan]]



    # add headers outside above loop to make header more readable
    for i_chan in range(nchans):
        header['BIASM{}'.format(i_chan+1)] = (
            mean_vos[i_chan], '[e-] channel {} mean vertical overscan'
            .format(i_chan+1))

    for i_chan in range(nchans):
        header['RDN{}'.format(i_chan+1)] = (
            std_vos[i_chan], '[e-] channel {} sigma (STD) vertical overscan'
            .format(i_chan+1))


    # write the average of both the means and standard deviations
    # determined for each channel to the header
    header['BIASMEAN'] = (np.nanmean(mean_vos), '[e-] average all channel means '
                          'vert. overscan')
    header['RDNOISE'] = (np.nanmean(std_vos), '[e-] average all channel sigmas '
                         'vert. overscan')


    # reset warnings
    warnings.resetwarnings()


    return data_out


################################################################################

def os_corr_orig (data, header, imgtype, xbin=1, ybin=1, data_limit=2000,
                  tel=None):

    """Function that corrects [data] for the overscan signal in the
       vertical and horizontal overscan strips. The definitions of the
       different data/overscan/channel sections are taken from
       [set_blackbox].  The function returns a data array that
       consists of the data sections only, i.e. without the overscan
       regions. The [header] is update in place.

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red = (
        define_sections(np.shape(data), xbin=xbin, ybin=ybin, tel=tel))

    # use median box filter with width [dcol] to decrease the noise
    # level in the overscan column's clipped mean for the horizontal
    # overscan when it has a limited amount of pixels
    nrows_hos = np.shape(data[os_sec_hori[0]])[0]
    if nrows_hos <= int(100/ybin):
        # after testing, 15-21 seem decent widths to use
        dcol = int(np.ceil(15./ybin))
    else:
        # otherwise, determine it per column
        dcol = 1

    dcol_half = int(dcol/2.)+1

    # number of data columns and rows in the channel (without overscans)
    ncols = get_par(set_bb.xsize_chan,tel) // xbin
    nrows = get_par(set_bb.ysize_chan,tel) // ybin

    # initialize output data array (without overscans)
    ny = get_par(set_bb.ny,tel)
    nx = get_par(set_bb.nx,tel)
    data_out = np.zeros((nrows*ny, ncols*nx), dtype='float32')

    # and arrays to calculate average means and stds over all channels
    nchans = np.shape(data_sec)[0]
    mean_vos = np.zeros(nchans)
    std_vos = np.zeros(nchans)
    median_vos = np.zeros(nchans)
    median_hos = np.zeros(nchans)

    vos_poldeg = get_par(set_bb.voscan_poldeg,tel)
    nrows_chan = np.shape(data[chan_sec[0]])[0]
    y_vos = np.arange(nrows_chan)


    for i_chan in range(nchans):

        # -----------------
        # vertical overscan
        # -----------------

        # first subtract a low-order polynomial fit to the clipped
        # mean (not median, because input pixels are still integers!)
        # of the vertical overcan section from the entire channel

        # determine clipped mean for each row
        data_vos = data[os_sec_vert[i_chan]]
        mean_vos_col, __, __ = sigma_clipped_stats(data_vos, axis=1, mask_value=0)

        # fit low order polynomial to vertical overscan, avoiding
        # outlying values due to e.g. very bright star that bleeds
        # into the vertical overscan sections at the bottom of the
        # image for channels 1-8, and at the top for channels 9-16
        try:
            polyfit_ok = True
            nsigma = 5
            mean, median, stddev = sigma_clipped_stats(mean_vos_col, sigma=nsigma)
            if stddev==0:
                mask_fit = np.ones(nrows_chan, dtype=bool)
            else:
                mask_fit = (np.abs(mean_vos_col-mean)/stddev <= nsigma)

            p = np.polyfit(y_vos[mask_fit], mean_vos_col[mask_fit], vos_poldeg)

        except Exception as e:
            polyfit_ok = False
            #log.exception(traceback.format_exc())
            log.exception('exception was raised during polynomial fit to '
                          'channel {} vertical overscan'.format(i_chan))


        # add fit coefficients to image header
        for nc in range(len(p)):
            p_reverse = p[::-1]
            if np.isfinite(p_reverse[nc]):
                value = p_reverse[nc]
            else:
                value = 'None'
            header['BIAS{}A{}'.format(i_chan+1, nc)] = (
                value, '[e-] channel {} vert. overscan A{} polyfit coeff'
                .format(i_chan+1, nc))

        # fit values
        fit_vos_col = np.polyval(p, y_vos)
        if not np.all(np.isfinite(fit_vos_col)):
            polyfit_ok = False

        header['VFITOK{}'.format(i_chan+1)] = (
            polyfit_ok, 'channel {} vert. overscan polyfit finite?'
            .format(i_chan+1))

        # if polynomial fit is reliable, subtract this off the entire
        # channel; otherwise subtract the nanmedian of the vos row
        # means
        if polyfit_ok:
            mean_vos[i_chan] = np.mean(fit_vos_col)
            data[chan_sec[i_chan]] -= fit_vos_col.reshape(nrows_chan,1)
        else:
            mean_vos[i_chan] = np.nanmedian(mean_vos_col)
            data[chan_sec[i_chan]] -= mean_vos[i_chan]

        if False:
            plt.plot(y_vos, mean_vos_col, color='black')
            plt.plot(y_vos, fit_vos_col, color='red')
            plt.ylim([mean-nsigma*stddev, mean+nsigma*stddev])
            plt.savefig('test_poly_chan{}.pdf'.format(i_chan+1))
            plt.close()

        data_vos = data[os_sec_vert[i_chan]]
        # determine std of overscan subtracted vos:
        __, median_vos[i_chan], std_vos[i_chan] = sigma_clipped_stats(
            data_vos, mask_value=0)


        # -------------------
        # horizontal overscan
        # -------------------

        # determine the running clipped mean of the overscan using all
        # values across [dcol] columns, for [ncols] columns
        data_hos = data[os_sec_hori[i_chan]]

        # replace very high values (due to bright objects on edge of
        # channel) with function [inter_pix] in zogy.py
        mask_hos = (data_hos > data_limit)

        # if it concerns a single column on its own that is covering
        # at least half the overscan height, unmask that column
        mask_x = (np.sum(mask_hos, axis=0) > 0.5 * mask_hos.shape[0])

        # opening with [True,True] will remove single True value from
        # mask_x but will leave multiple adjacent True values be
        mask_x_open = ndimage.binary_opening(mask_x, structure=np.ones(2))

        # identify any single True values that were removed
        mask_x_restore = np.logical_xor (mask_x, mask_x_open)

        # reset mask_hos for that/those column(s)
        mask_hos[:,mask_x_restore] = False

        # add couple of pixels connected to this mask
        mask_hos = ndimage.binary_dilation(mask_hos,
                                           structure=np.ones((3,3)).astype('bool'),
                                           iterations=2)

        # interpolate spline over these pixels
        if imgtype == 'object':
            data_hos = inter_pix (data_hos, std_vos[i_chan], mask_hos,
                                  interp_func='poly', order=2,
                                  dpix=50, fit_neg_values=True)

        # determine clipped mean for each column
        mean_hos, __, __ = sigma_clipped_stats(data_hos, axis=0,
                                               mask_value=0)
        if dcol > 1:
            oscan = [np.nanmedian(mean_hos[max(k-dcol_half,0):
                                           min(k+dcol_half+1,ncols)])
                     for k in range(ncols)]
            # do not use the running mean for the first column(s)
            oscan[0:dcol_half] = mean_hos[0:dcol_half]
        else:
            oscan = mean_hos[0:ncols]


        # subtract horizontal overscan from data section
        data[data_sec[i_chan]] -= oscan

        # for channel 9 of BG2, add difference between clipped median
        # of vertical and horizontal overscans, where for both the
        # vertical overscan has been subtracted
        #
        # CHECK!!!
        if False and tel=='BG2' and i_chan==8:
            __, median_hos[i_chan], __ = sigma_clipped_stats(data_hos,
                                                             mask_value=0)
            data[data_sec[i_chan]] += (median_vos[i_chan] - median_hos[i_chan])


        # place into [data_out]
        data_out[data_sec_red[i_chan]] = data[data_sec[i_chan]]



    # add headers outside above loop to make header more readable
    for i_chan in range(nchans):
        header['BIASM{}'.format(i_chan+1)] = (
            mean_vos[i_chan], '[e-] channel {} mean vertical overscan'
            .format(i_chan+1))

    for i_chan in range(nchans):
        header['RDN{}'.format(i_chan+1)] = (
            std_vos[i_chan], '[e-] channel {} sigma (STD) vertical overscan'
            .format(i_chan+1))


    # write the average of both the means and standard deviations
    # determined for each channel to the header
    header['BIASMEAN'] = (np.nanmean(mean_vos), '[e-] average all channel means '
                          'vert. overscan')
    header['RDNOISE'] = (np.nanmean(std_vos), '[e-] average all channel sigmas '
                         'vert. overscan')


    # if the image is a flatfield, add some header keywords with the
    # statistics of [data_out]; this part has been moved to when
    # get_flatstats is called for flat
    if False and imgtype == 'flat':
        sec_temp = get_par(set_bb.flat_norm_sec,tel)
        value_temp = '[{}:{},{}:{}]'.format(
            sec_temp[0].start+1, sec_temp[0].stop+1,
            sec_temp[1].start+1, sec_temp[1].stop+1)
        header['STATSEC'] = (
            value_temp, 'pre-defined statistics section [y1:y2,x1:x2]')

        header['MEDSEC'] = (
            np.median(data_out[sec_temp]),
            '[e-] median flat over STATSEC')

        header['STDSEC'] = (
            np.std(data_out[sec_temp]),
            '[e-] sigma (STD) flat over STATSEC')

        # full image statistics
        index_stat = get_rand_indices(data_out.shape)
        __, median, std = sigma_clipped_stats(data_out[index_stat], mask_value=0)
        header['FLATMED'] = (median, '[e-] median flat')
        header['FLATSTD'] = (std, '[e-] sigma (STD) flat')


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in os_corr')

    return data_out


################################################################################

def xtalk_corr (data, crosstalk_file, data_mask=None):

    if get_par(set_zogy.timing,tel):
        t = time.time()


    if isfile(crosstalk_file):
        log.info ('crosstalk file: {}'.format(crosstalk_file))


    # should data_mask not be provided, create a zero mask
    if data_mask is None:
        data_mask = np.zeros(data.shape, dtype=bool)


    # read file with corrections
    #table = Table.read(crosstalk_file, format='ascii',
    #                   names=['victim', 'source', 'correction'])
    # new crosstalk files have columns defined on 1st line
    table = Table.read(crosstalk_file, format='ascii')
    # file contains channel numbers; convert to indices
    victim = table['victim'].value - 1
    source = table['source'].value - 1
    correction = table['correction'].value


    # channel image sections
    chan_sec, __, __, __, __ = define_sections(np.shape(data), tel=tel)
    # number of channels
    nchans = np.shape(chan_sec)[0]


    # prepare masks for source and victium channels separately
    mask_value = get_par(set_zogy.mask_value,tel)
    val_bad = mask_value['bad']
    val_cosmic = mask_value['cosmic ray']
    val_edge = mask_value['edge']

    # use positive fluxes and pixels not affected by bad pixels or
    # cosmics in source channel
    mask_source = ((data > 0) &
                   (data_mask & val_bad == 0) &
                   (data_mask & val_cosmic == 0))


    # avoid pixels that land in edge region in victim channel
    mask_victim = (data_mask & val_edge == 0)


    # use matrix multiplication with np.matmul; it is much faster
    # (~3s) than looping through the input file and correcting for
    # each source-victim combination separately (~25s) as done before


    # create 16x16 matrix with correction coefficients, with source
    # indices along axis=0 and victim indices along axis=1; could have
    # saved the coefficients as an array in a numpy file, but this
    # conversion is very quick and the ASCII file can be easily viewed
    coeffs = np.zeros((nchans,nchans))
    for k in range(len(table)):
        coeffs[source[k], victim[k]] = correction[k]



    # data_stack with shape (5280,1320,16) so it can be used in
    # np.matmul, taking into account mask_source
    data_stack = np.stack([data[chan_sec[i]] * mask_source[chan_sec[i]]
                           for i in range(nchans)], axis=2)
    # same but with channels flipped in y
    data_stack_flip = np.stack([np.flipud(data[chan_sec[i]] *
                                          mask_source[chan_sec[i]])
                                for i in range(nchans)], axis=2)


    # initialize data_corr_chans with shape (16,5280,1320), which will
    # contain the corrections to be subtracted from the input data
    ysize_chan = get_par(set_bb.ysize_chan,tel)
    xsize_chan = get_par(set_bb.xsize_chan,tel)
    data_corr_chans = np.zeros((nchans, ysize_chan, xsize_chan))


    # calculate matmul for the 4 quadrants of the 16x16 matrix
    # separately, because of flipping of data in the non-diagonal
    # quadrants
    s1 = slice(0,8)
    s2 = slice(8,16)

    # source and victim slices corresponding to quandrants
    # [q=0 q=2]
    # [q=1 q=3]
    sls = [s1, s2, s1, s2]
    slv = [s1, s1, s2, s2]

    # loop quandrants
    for q in range(4):

        # use (flipped) data stack for (non-) diagonal quandrants
        if q in [0,3]:
            data_use = data_stack
        else:
            data_use = data_stack_flip

        # apply matmul; the swapaxes commands are to arrange the axes
        # and shape from the output of np.matmul to the expected ones
        data_corr_chans[slv[q]] += (np.matmul(data_use[:,:,sls[q]],
                                              coeffs[sls[q],slv[q]])
                                    .swapaxes(0,2).swapaxes(1,2))


    # use data_corr_chans to correct original data, taking care of
    # mask_victim
    for i in range(nchans):
        data[chan_sec[i]] -= data_corr_chans[i] * mask_victim[chan_sec[i]]



    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in xtalk_corr')


    return


################################################################################

def xtalk_corr_old (data, crosstalk_file, data_mask=None):

    if get_par(set_zogy.timing,tel):
        t = time.time()


    if isfile(crosstalk_file):
        log.info ('crosstalk file: {}'.format(crosstalk_file))


    # should data_mask not be provided, create a zero mask
    if data_mask is None:
        data_mask = np.zeros(data.shape, dtype=bool)


    # read file with corrections
    #table = Table.read(crosstalk_file, format='ascii',
    #                   names=['victim', 'source', 'correction'])
    # new crosstalk files have columns defined on 1st line
    table = Table.read(crosstalk_file, format='ascii')
    victim = table['victim'].value
    source = table['source'].value
    corr = table['correction'].value


    # convert to indices
    victim -= 1
    source -= 1


    # channel image sections
    chan_sec, __, __, __, __ = define_sections(np.shape(data), tel=tel)
    # number of channels
    nchans = np.shape(chan_sec)[0]


    # prepare data with channels flipped in y, to be used inside loop
    # below in case source and victim channels are in different rows
    data_flip = np.copy(data)
    for i in range(nchans):
        data_flip[chan_sec[i]] = np.flipud(data[chan_sec[i]])


    # prepare masks for source and victium channels separately, to be
    # used inside loop below
    mask_value = get_par(set_zogy.mask_value,tel)
    val_bad = mask_value['bad']
    val_edge = mask_value['edge']

    # avoid pixels that land in edge region in victim channel
    mask_victim = ((data_mask & val_edge != val_edge))

    # use positive fluxes and pixels not affected by bad pixels in
    # source channel
    mask_source = ((data > 0) & (data_mask & val_bad != val_bad))

    # also prepare flipped mask_source
    mask_source_flip = np.copy(mask_source)
    for i in range(nchans):
        mask_source_flip[chan_sec[i]] = np.flipud(mask_source[chan_sec[i]])


    # loop arrays in file and correct the channels accordingly
    for k in range(len(victim)):

        # abbreviate source and victim channel sections
        css = chan_sec[source[k]]
        csv = chan_sec[victim[k]]


        # define source channel data; if source and victim channels
        # are not in the same row, use data where the channels are
        # flipped
        row_source = source[k] // 8
        row_victim = victim[k] // 8
        if row_source != row_victim:
            data_source_chan = data_flip[css]
            mask_source_chan = mask_source_flip[css]
        else:
            data_source_chan = data[css]
            mask_source_chan = mask_source[css]


        # mask to use is combination of source and victim masks
        mask_use = mask_source_chan & mask_victim[csv]


        # correct the victim channel; would be better to first
        # determine the delta correction instead of applying it
        # directly to the data, as the corrected data will be used in
        # the correction of other channels, but that effect is of the
        # order of (10^-4)**2, so insignificant
        data[csv][mask_use] -= data_source_chan[mask_use] * corr[k]



    # keep this info for the moment:

    # alternatively, an attempt to do it through matrix
    # multiplication, which should be much faster, but the loop is
    # only taking 1-2 seconds anyway.

    # build nchans x nchans correction matrix, such that when
    # matrix-multiplying: data[chan_sec] with the correction matrix,
    # the required crosstalk correction to data[chan_sec] is
    # immediately obtained
    #corr_matrix_old = np.zeros((nchans,nchans))
    #for k in range(len(victim)):
    #    corr_matrix_old[int(source[k]-1), int(victim[k]-1)] = correction[k]

    # since channels were defined differently, shuffle them around
    #corr_matrix = np.copy(corr_matrix_old)
    #top_left = tuple([slice(0,nchans/2), slice(0,nchans/2)])
    #bottom_right = tuple([slice(nchans/2,nchans), slice(nchans/2,nchans)])
    #corr_matrix[top_left] = corr_matrix_old[bottom_right]
    #corr_matrix[bottom_right] = corr_matrix_old[top_left]

    #shape_temp = np.shape(chan_sec[0]) + (nchans,)
    #data_chan_row = np.zeros(shape_temp)
    #data[chan_sec] -= np.matmul(data[chan_sec], corr_matrix)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in xtalk_corr')

    return data



################################################################################

def nonlin_corr(data, nonlin_corr_file):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    # read file with list of splinefit objects
    with open(nonlin_corr_file, 'rb') as f:
        fit_splines = pickle.load(f)

    # spline fit was determined from counts instead of electrons, so
    # need gain and correct channel for channel; could also perform
    # this correction before the gain correction, but then
    # overscan/bias correction should be done before the gain
    # correction as well
    gain = get_par(set_bb.gain,tel)

    # determine reduced data sections
    __, __, __, __, data_sec_red = define_sections(np.shape(data), tel=tel)

    # loop channels
    nchans = np.shape(data_sec_red)[0]
    for i_chan in range(nchans):

        # spline determines fractional correction:
        #   splinefit = (data - linear fit) / linear fit
        # so to correct data to linear fit:
        #   linear fit = data / (splinefit + 1)

        # temporary array with channel data in counts
        data_counts = data[data_sec_red[i_chan]]/gain[i_chan]

        # do not correct for data above 50,000 (+bias level)
        frac_corr = np.ones(data_counts.shape)
        mask_corr = (data_counts <= 50000)
        frac_corr[mask_corr] = fit_splines[i_chan](data_counts[mask_corr])

        # correct input data in electrons
        data[data_sec_red[i_chan]] /= (frac_corr + 1)


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in nonlin_corr')

    return data


################################################################################

def gain_corr(data, header, tel=None):

    """Returns [data] corrected for the [gain] defined in [set_bb.gain]
       for the different channels

    """

    if get_par(set_zogy.timing,tel):
        t = time.time()

    # gain values
    gain = get_par(set_bb.gain,tel)

    # channel image sections
    chan_sec, __, __, __, __ = define_sections(np.shape(data), tel=tel)
    nchans = np.shape(chan_sec)[0]

    for i_chan in range(nchans):
        data[chan_sec[i_chan]] *= gain[i_chan]
        header['GAIN{}'.format(i_chan+1)] = (gain[i_chan], '[e-/ADU] gain applied to '
                                             'channel {}'.format(i_chan+1))

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='in gain_corr')


    # check if different channels in [set_bb.gain] correspond to the
    # correct channels; currently indices of gain correspond to the
    # channels as follows:
    #
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]


################################################################################

def get_path (date, dir_type):

    # define path

    # date can be any of yyyy/mm/dd, yyyy.mm.dd, yyyymmdd,
    # yyyy-mm-dd or yyyy-mm-ddThh:mm:ss.s; if the latter is
    # provided, make sure to set [date_dir] to the date of the
    # evening before UT midnight
    #
    if 'T' in date:

        # these two lines replace the block below

        # get offset with UTC
        #UTC_offset = (datetime.now().replace(
        #    tzinfo=gettz(get_par(set_zogy.obs_timezone,tel)))
        #              .utcoffset().total_seconds()/3600)
        #date = (Time(int(Time(date).jd+UTC_offset/24), format='jd')
        #        .strftime('%Y%m%d'))

        # rounds date to microseconds as more digits cannot be
        # defined in the format (next line)
        #date = Time(date, format='isot').isot
        #date_format = '%Y-%m-%dT%H:%M:%S.%f'
        #high_noon = 'T12:00:00.0'

        if '.' in date:
            # rounds date to microseconds as more digits cannot be
            # defined in the format (next line)
            date = str(Time(date, format='isot'))
            date_format = '%Y-%m-%dT%H:%M:%S.%f'
            high_noon = 'T12:00:00.0'
        else:
            date_format = '%Y-%m-%dT%H:%M:%S'
            high_noon = 'T12:00:00'

        date_ut = datetime.strptime(date, date_format).replace(
            tzinfo=gettz('UTC'))
        date_noon = date.split('T')[0]+high_noon
        date_local_noon = datetime.strptime(date_noon,date_format).replace(
            tzinfo=gettz(get_par(set_zogy.obs_timezone,tel)))
        if date_ut < date_local_noon:
            # subtract day from date_only
            date = (date_ut - timedelta(1)).strftime('%Y-%m-%d')
        else:
            date = date_ut.strftime('%Y-%m-%d')


    # this [date_eve] in format yyyymmdd is also returned
    date_eve = ''.join(e for e in date if e.isdigit())
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])


    if dir_type == 'read':
        root_dir = get_par(set_bb.raw_dir,tel)
    elif dir_type == 'write':
        root_dir = get_par(set_bb.red_dir,tel)
    else:
        log.error ('[dir_type] not one of "read" or "write"')

    path = '{}/{}'.format(root_dir, date_dir)


    # remove double forward slashes, but not the ones associated to the
    # google cloud bucket name
    path_tmp = path.split('gs://')[-1]
    if '//' in path_tmp:
        log.info ('replacing double slash in path name: {}'.format(path))

        # replace double slash with single one
        path_tmp = path_tmp.replace('//','/')
        # put back start of bucket name, which can be an empty string
        path = '{}{}'.format(path.split(path_tmp)[0], path_tmp)



    return path, date_eve


################################################################################

def get_date_time (header):
    '''Returns image observation date and time in the correct format.

    :param header: primary header
    :type header: header
    :returns: str -- '(date), (time)'
    '''
    date_obs = header['DATE-OBS'] #load date from header
    date_obs_split = re.split('-|:|T|\.', date_obs) #split date into date and time
    return "".join(date_obs_split[0:3]), "".join(date_obs_split[3:6])


################################################################################

def sort_files(read_path, search_str, recursive=False):

    """Function to sort raw files by type. Lists all files in
       read_path and to sorts files into bias, flat and science images
       using the IMAGETYP header keyword. Compatible with Google Cloud
       buckets.

    """

    #glob all raw files and sort
    #if recursive:
    #    all_files = sorted(glob.glob('{}/**/{}'.format(read_path, search_str),
    #                                 recursive=recursive))
    #else:
    #    all_files = sorted(glob.glob('{}/{}'.format(read_path, search_str)))


    # dealing with google cloud bucket?
    google_cloud = (read_path[0:5] == 'gs://')

    if not google_cloud or types is None or 'object' in types:
        all_files = sorted(list_files(read_path, search_str, recursive=recursive))
    else:
        # reading a bucket fits file header is rather slow in the
        # Google cloud, so if only biases, darks, flats or object
        # images are needed through input parameter imgtypes (= global
        # parameter types), then only select those files
        tmp_files = []
        if 'bias' in types:
            tmp_files.append(list_files(read_path, 'Bias', recursive=recursive))
        if 'flat' in types:
            tmp_files.append(list_files(read_path, 'flat', recursive=recursive))
        if 'dark' in types:
            tmp_files.append(list_files(read_path, 'Dark', recursive=recursive))

        # for object files, all files are selected (first part of this
        # if statement), mainly because the raw image name does not
        # indicate the image type for object files

        # clean up lists within list
        all_files = sorted([f for sublist in tmp_files for f in sublist])


    biases = [] #list of biases
    darks = [] #list of darks
    flats = [] #list of flats
    objects = [] # list of science images
    others = [] # list of other images

    for i, filename in enumerate(all_files): #loop through raw files

        header = read_hdulist(filename, get_data=False, get_header=True)

        if 'IMAGETYP' not in header:
            log.info ('keyword IMAGETYP not present in header of image; '
                      'not processing {}'.format(filename))
            # add this file to [others] list, which will not be reduced
            others.append(filename)

        else:

            imgtype = header['IMAGETYP'].lower() #get image type

            if 'bias' in imgtype: #add bias files to bias list
                biases.append(filename)
            elif 'dark' in imgtype: #add dark files to dark list
                darks.append(filename)
            elif 'flat' in imgtype: #add flat files to flat list
                flats.append(filename)
            elif 'object' in imgtype: #add science files to science list
                objects.append(filename)
            else:
                # none of the above, add to others list
                others.append(filename)

    return biases, darks, flats, objects, others


################################################################################

def write_fits (fits_out, data, header, overwrite=True, run_fpack=True,
                run_create_jpg=True, master=False, tel=None):


    mem_use (label='in write_fits at start')


    # add time stamp of file creation to header
    header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')


    # dealing with google cloud bucket?
    google_cloud = (fits_out[0:5] == 'gs://')


    # normal file system
    if not google_cloud:

        # make dir for output file if it doesn't exist yet
        make_dir (os.path.dirname(fits_out))

        # write fits directly to the output [fits_out]
        fits.writeto(fits_out, data, header, overwrite=overwrite)

        # fpack
        if run_fpack:
            fits_out = fpack (fits_out)

        # create jpg
        if run_create_jpg:
            file_jpg = create_jpg (fits_out)


    else:
        # google cloud bucket

        # write data to fits_tmp in tmp_path folder, which is the tmp
        # dir with subfolder the name of the image without the fits
        # extension
        if not master:
            tmp_path = '{}/{}'.format(get_par(set_bb.tmp_dir,tel),
                                      fits_out.split('/')[-1].replace('.fits',''))
        else:
            # for master frames, just write the fits file without the
            # extra subfolder, otherwise would also need to remove
            # that folder while there is no logfile kept for master
            # frames anyway
            tmp_path = get_par(set_bb.tmp_dir,tel)
            # just in case it does not exist yet
            make_dir (tmp_path)


        # write the tmp fits file
        fits_tmp = '{}/{}'.format(tmp_path, fits_out.split('/')[-1])
        log.info ('writing tmp fits file {}'.format(fits_tmp))
        fits.writeto(fits_tmp, data, header, overwrite=overwrite)


        # fpack
        if run_fpack:
            fits_tmp = fpack (fits_tmp)
            # add '.fz' to fits_out, which is returned by function
            if '.fz' not in fits_out:
                fits_out = '{}.fz'.format(fits_out)


        # create jpg
        if run_create_jpg:
            file_jpg = create_jpg (fits_tmp)


        # move fits_tmp to [dest_folder] in bucket
        dest_folder = os.path.dirname(fits_out)
        copy_file (fits_tmp, dest_folder+'/', move=True)


        # move the corresponding jpg file
        if run_create_jpg:
            copy_file (file_jpg, dest_folder+'/', move=True)


        # move the log file
        if not master:
            log_tmp = '{}.log'.format(fits_tmp.split('.fits')[0])
            if isfile (log_tmp):
                copy_file (log_tmp, dest_folder+'/', move=True)


            # try removing the tmp folder, but apparently there is
            # still an application using the logfile, so that the NFS
            # creates a .nsf... file in the tmp_path, i.e. it is not
            # empty. But even shutil.rmtree does not work.
            # N.B.: this works fine if /tmp is used as temporary folder,
            # but files are not visible, even while they are
            # being created
            try:
                os.rmdir(tmp_path)
            except Exception as e:
                log.warning ('could not remove tmp_path {}: {}'
                             .format(tmp_path, e))


    mem_use (label='in write_fits at end')

    return fits_out


################################################################################

def copy_flist (filelist, dest, move=False, verbose=True):

    # assume all files are either google cloud or not, so sufficient
    # to only check the 1st file
    google_cloud = (filelist[0][0:5]=='gs://' or dest[0:5]=='gs://')

    # make sure files exist
    filelist = [f for f in filelist if isfile(f)]
    if len(filelist)==0:
        log.warning ('no existing file(s) to copy or move')
        return


    if verbose:
        if move:
            label = 'moving'
        else:
            label = 'copying'

        for src_file in filelist:
            log.info('{} {} to {}'.format(label, src_file, dest))




    if not google_cloud:
        for src_file in filelist:

            if not move:
                shutil.copy2(src_file, dest)
            else:

                # input [dest] can be a file or directory, construct two possible
                # output filenames to check for below; if dest is a file, fn1 will
                # be the destination file, otherwise fn2 will be the destination
                # file
                fn1 = dest
                fn2 = os.path.join(dest, src_file.split('/')[-1])

                # if destination file already exists, remove it
                for fn in [fn1, fn2]:
                    if os.path.isfile(fn):
                        log.info ('{} already exists; removing it'.format(fn))
                        os.remove(fn)

                # move
                shutil.move(src_file, dest)


    else:

        # this could be done in python, but much easier with gsutil
        # from the shell
        if move:
            cp_cmd = 'mv'
        else:
            cp_cmd = 'cp'


        # sometimes this fails with a GCP credentials issue, so try a
        # couple of times if destination file is not created
        for i in range(3):

            try:
                # gsutil command (not actively supported anymore)
                cmd = ['gsutil', '-m', '-q', cp_cmd, '-I', dest]
                result = subprocess.run(cmd, input='\n'.join(filelist)
                                        .encode('utf-8'))
            except:
                log.warning ('command {} did not succeed; trying again'
                             .format(cmd))
                if i==2:
                    msg = 'command {} failed three times in a row'.format(cmd)
                    log.exception (msg)
                    raise RuntimeError (msg)

            else:
                break



    return


################################################################################

def copy_file (src_file, dest, move=False, verbose=True):

    """function to copy or move a file [src_file] to [dest], which may
       be a file or folder; [src_file] and/or [dest] may be part of
       the usual filesystem or in a google cloud bucket; in the latter
       case the argument(s) should start with gs://[bucket_name]

    """

    # check if src_file exists
    if not isfile(src_file):
        log.error ('{} not found'.format(src_file))
        return


    if verbose:
        if move:
            label = 'moving'
        else:
            label = 'copying'

        log.info('{} {} to {}'.format(label, src_file, dest))


    # input [dest] can be a file or directory, construct two possible
    # output filenames to check for below; if dest is a file, fn1 will
    # be the destination file, otherwise fn2 will be the destination
    # file
    fn1 = dest
    fn2 = os.path.join(dest, src_file.split('/')[-1])


    # if not dealing with google cloud buckets, use shutil.copy2 or shutil.move
    if not (src_file[0:5] == 'gs://' or dest[0:5] == 'gs://'):

        if not move:
            shutil.copy2(src_file, dest)
        else:

            # if destination file already exists, remove it
            for fn in [fn1, fn2]:
                if os.path.isfile(fn):
                    log.info ('{} already exists; removing it'.format(fn))
                    os.remove(fn)

            # move
            shutil.move(src_file, dest)

    else:

        # this could be done in python, but much easier with gsutil
        # from the shell
        if move:
            cp_cmd = 'mv'
        else:
            cp_cmd = 'cp'


        # sometimes this fails with a GCP credentials issue, so try a
        # couple of times if destination file is not created
        for i in range(3):

            # gsutil command (not actively supported anymore)
            cmd = ['gsutil', '-q', cp_cmd, src_file, dest]
            # gcloud storage alternative
            #cmd = ['gcloud', 'storage', cp_cmd, src_file, dest]
            result = subprocess.run(cmd)

            if isfile(fn1) or isfile(fn2):
                # file was moved/copied ok
                break
            else:
                if i < 2:
                    log.warning ('command {} did not succeed; trying again'
                                 .format(cmd))
                else:
                    msg = 'command {} failed three times in a row'.format(cmd)
                    log.exception (msg)
                    raise RuntimeError (msg)


    return


################################################################################

def add_headkeys (path_full, fits_headers, search_str='', end_str='',
                  tel=None, nproc=1):

    # read [fits_headers]
    log.info ('reading fits table {}'.format(fits_headers))
    table_headers = Table.read(fits_headers, memmap=True)


    # determine its column names and dtypes
    colnames = table_headers.colnames
    dtypes = [str(table_headers.dtype[n]) for n in colnames]
    log.info ('number of columns: {}'.format(len(colnames)))


    if isfile (path_full):

        log.info ('{} recognized as existing file'.format(path_full))

        # if input path_full is actually a file, read it, assuming it
        # is an ASCII file without any header, with the filenames to
        # process listed in the 1st column
        table_fn = Table.read(path_full, format='ascii', names=['filename'],
                              data_start=0)
        filenames = sorted(table_fn['filename'])

        if tel is not None:
            # select filenames relevant for this telescope
            filenames = [fn for fn in filenames if tel in fn]


    else:

        raise SystemExit

        # use [list_files] to determine which filenames to process
        filenames = sorted(list_files('{}'.format(path_full), search_str=search_str,
                                      end_str=end_str, recursive=True))



    log.info ('number of filenames for which to add headers: {}'
              .format(len(filenames)))


    if len(filenames) > 0:

        # in case of only 1 processor:
        if nproc == 1:
            rows = []
            for filename in filenames:
                row = get_head_row (filename, colnames)
                rows.append(row)

        else:
            # for multiple processors, use pool_func and function
            # [get_row] to multi-process list of basenames
            rows = pool_func (get_head_row, filenames, colnames, nproc=nproc)


        # convert rows to table
        table = Table(rows=rows, names=colnames, masked=True, dtype=dtypes)

        # add table to input table
        table_headers = vstack([table_headers, table])

        # unique entries, sorted by FILENAME; this works on the entire
        # table, which is slower than sorting just the table being
        # added (as was done before), but it allows an entire night to
        # be added again when some files needed to be re-processed
        table_headers = unique(table_headers, keys='FILENAME', keep='last')

        # overwrite fits_headers
        if 'gs://' in fits_headers:
            # in google cloud table.write to a file in bucket is not
            # working, so save table to temporary file in the current
            # working directory and then copy to bucket
            with tempfile.NamedTemporaryFile(delete=True, suffix='.fits') as f:
                table_headers.write (f.name, overwrite=True)
                copy_file (f.name, fits_headers)
                log.info ('done copying')

        else:
            table_headers.write(fits_headers, overwrite=True)

    else:

        log.warning ('no files with path/folder {} (and its subfolders) with '
                     'search_str {} and end_str {} for which to add header keys'
                     .format(path_full, search_str, end_str))


    return


################################################################################

def get_head_row (filename, colnames):

    log.info ('processing {}'.format(filename))

    # read filename header
    header = read_hdulist(filename, get_data=False, get_header=True)

    # loop columns to add
    row = []
    for i, colname in enumerate(colnames):
        if colname in header:
            row += [header[colname]]
        elif colname.lower() == 'filename':
            # add filename with full path
            row += [filename]
            # add filename part starting with yyyy/mm/dd/..
            #row += [filename.split('/red/')[-1]]
        else:
            row += [np.ma.masked]

        if row[i] == 'None' or row[i] == '':
            row[i] = np.ma.masked

    return row


################################################################################

def unzip(imgname, put_lock=True, timeout=None):

    """Unzip a gzipped of fpacked file.
       Same [subpipe] function STAP_unzip.
    """

    if put_lock:
        lock.acquire()

    if '.gz' in imgname:
        log.info ('gunzipping {}'.format(imgname))
        subprocess.run(['gunzip',imgname])
        imgname = imgname.replace('.gz','')

    elif '.fz' in imgname:
        log.info ('funpacking {}'.format(imgname))
        subprocess.run(['funpack', '-D', imgname])
        imgname = imgname.replace('.fz','')

    if put_lock:
        lock.release()

    return imgname


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

    # multiprocessing method; see https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # and also https://pythonspeed.com/articles/python-multiprocessing/
    #set_start_method('spawn')
    #set_start_method('fork') # this is the current default
    #set_start_method('forkserver')


    parser = argparse.ArgumentParser(description='User parameters')

    parser.add_argument('--telescope', type=str, default='ML1',
                        help='Telescope name (ML1, BG2, BG3 or BG4); '
                        'default=\'ML1\'')

    parser.add_argument('--mode', type=str, default='day',
                        help='Day or night mode of pipeline; default=\'day\'')

    parser.add_argument('--date', type=str, default=None,
                        help='Date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd '
                        'or yyyy.mm.dd); default=None')

    parser.add_argument('--read_path', type=str, default=None,
                        help='Full path to the input raw data directory; if not '
                        'defined it is determined from [set_blackbox.raw_dir], '
                        '[telescope] and [date]; default=None')

    parser.add_argument('--recursive', type=str2bool, default=False,
                        help='Recursively include subdirectories for input '
                        'files; default=False')

    parser.add_argument('--imgtypes', type=str, default=None,
                        help='Only consider this(these) image type(s); '
                        'default=None')

    parser.add_argument('--filters', type=str, default=None,
                        help='Only consider this(these) filter(s); default=None')

    parser.add_argument('--image', type=str, default=None, help='Only process '
                        'this particular image (requires full path); '
                        'default=None')

    parser.add_argument('--image_list', type=str, default=None,
                        help='Process images listed in ASCII file with this '
                        'name; default=None')

    parser.add_argument('--img_reduce', type=str, default=None,
                        help='Perform basic image reduction part; default=None')

    parser.add_argument('--cat_extract', type=str, default=None,
                        help='Perform catalog extraction and calibration part; '
                        'default=None')

    parser.add_argument('--trans_extract', type=str, default=None,
                        help='Perform transient extraction part; default=None')

    parser.add_argument('--force_reproc_new', type=str, default=None,
                        help='Force reprocessing of new image; default=None')

    parser.add_argument('--master_date', type=str, default=None,
                        help='Create master file of type(s) [imgtypes] and '
                        'filter(s) [filters] for this(these) date(s) (e.g. 2019 '
                        'or 2019/10 or 2019-10-14; can also be an ascii file '
                        'with the date(s) in the 1st column and optionally the '
                        'filter(s) in the 2nd column); default=None')

    parser.add_argument('--name_genlog', type=str, default=None,
                        help='Name of general log file to save; if path is not '
                        'provided, it will be saved in the telescope\'s log '
                        'directory; default of None will create logfile with name '
                        '[tel]_[date]_[time].log with date/time at start of '
                        'running blackbox')

    parser.add_argument('--keep_tmp', default=None,
                        help='keep temporary directories')


    args = parser.parse_args()
    run_blackbox (telescope=args.telescope, mode=args.mode, date=args.date,
                  read_path=args.read_path, recursive=args.recursive,
                  imgtypes=args.imgtypes, filters=args.filters, image=args.image,
                  image_list=args.image_list, master_date=args.master_date,
                  img_reduce=args.img_reduce, cat_extract=args.cat_extract,
                  trans_extract=args.trans_extract,
                  force_reproc_new=args.force_reproc_new,
                  name_genlog=args.name_genlog, keep_tmp=args.keep_tmp)
