
import os
import sys
import gc
import pickle

import set_zogy
import set_blackbox as set_bb

# setting number of threads through environment variable (used by
# e.g. astroscrappy) needs to be done before numpy is imported in
# [zogy]
# this needs to be done before numpy is imported in [zogy]
os.environ['OMP_NUM_THREADS'] = str(set_bb.nthread)

from zogy import *

import re   # Regular expression operations
import glob # Unix style pathname pattern expansion 
from multiprocessing import Pool, Manager, Lock, Queue, Array
import datetime as dt 
from dateutil.tz import gettz
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle
from astropy.time import Time
from astropy import units as u
import astroscrappy
from acstools.satdet import detsat, make_mask, update_dq
import shutil
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#from slackclient import SlackClient as sc
import ephem
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from qc import qc_check, run_qc_check
import platform

import aplpy

# due to regular problems with downloading default IERS file (needed
# to compute UTC-UT1 corrections for e.g. sidereal time computation),
# Steven created a mirror of this file in a google storage bucket
from astropy.utils import iers
iers.conf.iers_auto_url = (
    'https://storage.googleapis.com/blackbox-auxdata/timing/finals2000A.all')
iers.conf.iers_auto_url_mirror = 'http://maia.usno.navy.mil/ser7/finals2000A.all'

#from pympler import tracker
#import tracemalloc
#tracemalloc.start()

# commands to force the downloading of above IERS bulletin file in
# case a recent one (younger than 30 days) is not present in the cache
tnow = Time.now()
tnow.ut1  


__version__ = '0.9.2'
keywords_version = '0.9.2'

#def init(l):
#    global lock
#    lock = l


################################################################################

def run_blackbox (telescope=None, mode=None, date=None, read_path=None,
                  recursive=None, imgtypes=None, filters=None, image=None, 
                  image_list=None, master_date=None, slack=None):
    
    """Function that processes MeerLICHT or BlackGEM images, performs
    basic image reduction tasks such as overscan subtraction,
    flat-fielding and cosmic-ray rejection, and the feeds the image to
    zogy.py along with a reference image of the corresponding ML/BG
    field ID, to detect transients present in the image. If no
    reference image is present, the image that is being processed is
    defined to be the reference image.

    To do (for both blackbox.py and zogy.py):
    -----------------------------------------

    (3) map out and, if possible, decrease the memory consumption of
        blackbox and zogy - needs to be done on chopper or mlcontrol
        machine at Radboud, as laptop has too little memory

    (4) in night mode, issue that only images already present are
        processed

    (5) improve processing speed of background subtraction and other
        parts in the code where for-loops can be avoided

      --> why is updated function get_back so much slower on mlcontrol
          machine vs. macbook?

    (10) check if SExtractor background is manual or global and 
         has any influence on detections

    (12) go through logs, look for errors and exceptions and fix them:
      
      --> moffat fit to objects near the edge
      --> avoid dividing by zero in e.g. optimal flux functions

    (15) if too many transient regions are found, leave function        
         [get_trans] immediately as it takes very long to do the fits to
         1000s of transients. Probably easier to check number of 
         transients just before [get_trans] and flag red if more than
         e.g. 1000.

    (16) speed up psfex by providing a limited random sample of good
         PSF stars, e.g. 1000, rather than all; needs to be random
         and not the brightest

    (17) moffat fit to transients in D provides negative chi2s?

    (19) make log input in functions optional through log=None, so that
         they can be easily used by modules outside of blackbox/zogy.
         Maybe do the same for other parameters such as telescope.

    (20) optimal vs. large aperture magnitudes shows discrepant values
         at the bright end; why?

    (22) limit PSFEx LDAC catalog to [psfex_nstars] random stars
         obeying S/N constraints, such that ldac catalog has reasonable
         size - see (16)

    (23) header of output catalog is much more complete than header of
         reduced image; need to update the latter

         related: qc-flags in reduced images not consistent with those
         of catalog files, i.e. reduced images can have more severe flag
         than catalog file; why? - is because qc-flags in catalog headers
         were not updated after zogy run, so they still had the qc-flags
         from at the start of zogy.

    (24) airmass of combined reference image is off (and therefore
         also the zeropoint) because zogy is calculating it using DATE-OBS
         and RA,DEC, but DATE-OBS is an average time for the combined
         reference; maybe AIRMASS keyword can be updated with average
         airmass? Or .. since flux was scaled to airmass 1 (true?),
         airmass keywords can be updated to that value?

    (25) does nproc decrease when running pool_func a 2nd time?

    (27) number of initials SExtractor detections (S-NOBJ) not the same
         as final number of objects in output catalog (5 sigma); maybe
         add keyword that indicates the latter number

    (29) need way to avoid running SExtractor on edges of images; for 
         MeerLICHT, edges are zeroed in BlackBOX

    (30) if mask is provided, but saturation is not included (e.g. in case
         only a bad pixel mask is provided), the saturated and 
         saturated-connected pixels should be added to the mask
    
    (31) when processing WHT/ACAM image, important WCS calibration
         header keywords such as CRVAL?, CRPIX?, CD_??, are not
         overwritten by the new WCS solution. Why not? Maybe easiest
         solution is to delete these header keywords when new WCS is
         needed.

    (32) add sensor header keywords to reduced data, even if they don't 
         exist yet in the raw image header, such as keywords from Cryostat 
         and CompactRIO.

    (33) check out new source-extractor; new capabilities or
         signficiant improvements? Different parameters (cf. J2000 ->
         ICRS)? On mac os and macports, it is not possible anymore to
         install the old version 2.19.5, as it is considered obsolete; how
         about Ubuntu?

    (36) check if background STD image produced by source-extractor is
         reasonably close to the improved background STD image made in
         [get_back]; if not, then improved background determination in
         run_sextractor still needs to be done even if the background
         was already subtracted from the new or ref image.

    (37) add switch, or use the existing "redo" switch of zogy, to
         force redoing the zogy part even if the reduced image already
         exists in the reduced folder. So that a reduced image can be
         altered and run through zogy again.

    (40) Paul mentonioned issue with astrometry for 47Tuc, especially
         in the u-band; seems that the A-DRASTD and A-DDESTD are a bit
         higher than allowed and many u-band images are flagged red.

    (41) go through headers of all images again and check if ranges
         set in set_qc are still appropriate.

    (43) determine which background box size to use

    (45) add/keep dome azimuth header keyword

    (46) filter cosmic rays not detected by astroscrappy from output
         catalog with condition object: FWHM_obj < f * FWHM_ima, where
         f needs to be determined (~0.1-0.5). Or using error estimate
         on image FWHM: FWHM_obj < FWHM_ima * nsigma * err_FWHM_ima
    
    (47) improve astroscrappy cosmic-ray rejection parameters; too
         many cosmics go undetected

    (48) add mode such that all input files are considered new images,
         i.e. no reference images are created, nor is the comparison
         between new and ref done. This is to speed up the processing
         of a large number of files to be used for the reference
         building routine. Once the reference images are built, the
         files will need to be processed again, but without any steps
         that can be skipped - see also item #37.

    (49) creating master flats can/should be multiprocessed

    (50) change jpg scaling to zscale

    (51) check what is done when no master flat is found

    Done:
    -----

    (1) add reference image building module

      --> Scorr image with new reference image has low scatter, but shows
          ringing features similar to those often seen in the D image; 
          try fixpix individual images; tried fixpix and indeed seems
          to have improved. Some saturated stars still present, which was
          due to bias level not being considered in saturation level, now
          it is

      --> need to update zogy.py with using the scatter in an image for the
          description of the noise, rather than the background level plus 
          the read noise squared; because the new reference images will have 
          zero background and also the read noise varies from channel to 
          channel, so scatter is probably more accurate. 

      --> probably best to merge imcombine_MLBG and build_ref_MLBG into one
          single module, that also has a settings file

      --> add possibility in zogy for input images to have zero background
          through header keyword BKG-SUB with boolean value. This needs   
          updating in function run_sextractor, running SExtractor with    
          BACK_TYPE MANUAL and BACK_VALUE 0.0 if BKG-SUB==T(rue) and      
          skipping the additional function get_back.                      
    
    (2) determine reason for stalls that Danielle encounters

        - seems to have gone away by going to python 3.7 (see also 9)

    (6) change output catalog column names from ALPHAWIN_J2000 and
        DELTAWIN_J2000 to simply RA and DEC - this has been
        implemented in [run_sextractor]; chosen not to add _ICRS to
        the names, because if zogy input image is already WCS
        corrected (in case it is not used by BlackBOX), it may not be
        in ICRS. Also, the RA/DEC catalog output columns in the
        transient catalog do not contain the ICRS extension either,
        such as RA_PEAK or RA_PSF_D.

    (7) change epoch in header from 2000 to 2015.5

    (8) filter out transients that correspond to a negative spot in the
        new or ref image

    (9) go to python 3.7 on chopper; update singularity image

    (11) replace clipped_stats with more robust sigma_clipped_stats in
         zogy.py and blackbox.py; N.B.: sigma_clipped_stats returns
         output with dtype float64 - convert to float32 if these turn
         into large arrays. 

         Curious bug: when bottleneck is installed, the mean and std
         returned by sigma_clipped_stats is that of bottleneck.nanmean
         and nanstd (see
         https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html)
         but that is incorrect for a large (size>=2**25) array with
         dtype float32 as input (see
         e.g. https://github.com/pandas-dev/pandas/issues/25307); no
         obvious reference to this on the bottleneck github page.
         Came across this when running sigma_clipped_stats on full
         MeerLICHT images (~2**27) and mean and std values being
         strangely off compared to e.g. np.nanmean. If mean or std of
         full image is needed, convert the data to float64. Out of
         precaution, updated all sigma_clipped_stats input to float64.


    (13) why is PSF to D so much slower when use_bkg_var=True?
         
         - often a fit with use_bkg_var=True reaches the maximum number of    
           iterations (10,000, taking 3s), probably due to the higher error   
           when the actual variance rather than sky + RON**2 is used. Whenever 
           the number of iterations is higher than about 100, the reduced chi2 
           is very large anyway, so reduced this number to 1000, which leads to 
           average execution time to be 0.1s/transient.

    (14) switch around PSF fit to D and Moffat fit to D; Moffat fit
         appears to be much faster, and will decrease the number of 
         transients to be fit in PSF fit to D; on 2nd thought, the 
         current chi2 threshold for the Moffat fit is very high, so
         few transients will be discarded by it at the moment. Also,
         Moffat fits are less robust than the PSF fit, so cannot use
         it reliably to discard transients.

    (18) replace string '+' concatenations with formats
    
    (21) force orientation in thumbnail images to be North up East left

    (26) determine image zeropoints per channel, and try fitting a
         polynomial surface to the zeropoint values across the frame.
         Or by averaging stars' zeropoints over boxes and then
         interpolating them, similar to how background map is
         determined. The zeropoints are determined in zogy, so if they
         are needed in BlackBOX, then need to save an ascii file with
         the zeropoint info that can be used by BlackBOX.

    (28) flatfields regularly show a gradient diagonally across the image,
         increasing from lower left to top right, and these flats are
         not flagged red yet at the moment. Should include a parameter
         in the header to check. Easiest option is to use the keyword
         already present in the header: FLATRSTD, which is higher than
         0.03-0.07 when things go wrong. Or the ratio of the median levels
         of the 1st and last channel: FLATM1 / FLATM16. Related to this
         is how to pick up condensation. Would need a finer grid (4x4
         and one central square) than the channels to be able to pick that 
         up using deviation from the average channel values. Rebin the
         image and use statistics to discard a particular flat? See
         ~/Python/BlackGEM/test_bin2D.py
         
         in the end, added a couple of statistics to the flat and 
         reduced image header: ST-MRDIF and ST-MRDEV with which all of
         the flats with gradients can be discarded and most of the
         condensation spots as well.

    (34) bkg and bkg_std images in tmp dir are float64; check
         precision in get_back so that these are not bigger than
         float32

    (35) if tmp folders are kept, fpack them as well

    (38) PaulG noticed that something went wrong in scheduling
         observations (ascii2schedule.py?) with dithering pattern at
         relatively large declination, which may indicate a cos(dec)
         is not taken into account properly.
         -> this distance is calculated with the haversine function:

         if haversine(table_ID['RA'][i_ID], table_ID['DEC'][i_ID],
                      table_obs['RA'][i_obs], table_obs['DEC'][i_obs]) > 10./60:

         which should be correct; would be helpful if Paul noted down
         the output error, as that lists the input table coordinates and 
         the field coordinates.

    (39) singularity container building on ML-controlroom2 machine:
         problem with installing packages fitsio and reproject (latter
         is not important) and also watchdog seems to install fine but
         cannot be imported inside python. Tried with different versions
         of python (3, 3.6, 3.7, 3.8), but none works ok. Also tried
         installing packages through apt-get, e.g.:
    
         apt-get -y install python3-numpy python3-astropy python3-matplotlib
         ...

         but then still problems, o.a. with matplotlib

         IDIA will soon switch from singularity version 2.6.1 to
         version 3.5.2 (see email from ILIFU on 26 March 2020) - see
         if that could help the above issues
    
         Solution: deleted required installation of fitsio in setup.py 
                   file; also switched to singularity 3.5.2 

    (42) add jpg of reduced new/ref images to output, in similar
         fashion to fpacking of images at the end of blackbox

    (44) keep BlackBOX en ZOGY in separate directories when installing

    """
    
    
    global tel, filts, types
    tel = telescope
    filts = filters
    types = imgtypes
    if imgtypes is not None:
        types = imgtypes.lower()

        
    if get_par(set_zogy.timing,tel):
        t_run_blackbox = time.time()
        
        
    # initialize logging
    ####################

    if not os.path.isdir(get_par(set_bb.log_dir,tel)):
        os.makedirs(get_par(set_bb.log_dir,tel))
    
    global q, genlogfile, logger
    q = Manager().Queue() #create queue for logging
    genlogfile = '{}/{}_{}.log'.format(get_par(set_bb.log_dir,tel), tel,
                                       Time.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_log (genlogfile)

    # leave slack logging for now
    #log_stream = StringIO() #create log stream for upload to slack
    #streamhandler_slack = logging.StreamHandler(log_stream) #add log stream to logger
    #streamhandler_slack.setFormatter(formatter) #add format to log stream
    #genlog.addHandler(streamhandler_slack) #link logger to log stream
    #logger = MyLogger(genlog,mode,log_stream,slack) #load logger handler


    q.put(logger.info('processing in {} mode'.format(mode)))
    q.put(logger.info('log file: {}'.format(genlogfile)))
    q.put(logger.info('number of processes: {}'.format(get_par(set_bb.nproc,tel))))
    q.put(logger.info('number of threads: {}'.format(get_par(set_bb.nthread,tel))))


    # create master bias and/or flat if [master_date] is specified
    if master_date is not None:
        if types is None:
            mtypes = 'biasflat'
        else:
            mtypes = types
        if 'bias' in mtypes:
            create_masters ('bias', mdate=master_date)
        if 'flat' in mtypes:
            if filters is None:
                mfilts = get_par(set_zogy.zp_default,tel).keys()
            else:
                mfilts = re.sub(',|-|\.|\/', '', filters)
            for filt in mfilts:
                create_masters ('flat', filt=filt, mdate=master_date)
        logging.shutdown()
        return
            
    
    # [read_path] is assumed to be the full path to the directory with
    # raw images to be processed; if not provided as input parameter,
    # it is defined using the input [date] with the function
    # [get_path]
    if read_path is None:
        if date is not None:
            read_path, __ = get_path(date, 'read')
            q.put(logger.info('processing files from directory: {}'.format(read_path)))
        elif image is not None:
            pass
        elif image_list is not None:
            pass
        else:
            # if [read_path], [date], [image] and [image_list] are all None, exit
            q.put(logger.critical('[read_path], [date], [image] and [image_list] '
                                  'all None'))
            raise (SystemExit)
    else:
        # if it is provided but does not exist, exit
        if not os.path.isdir(read_path):
            q.put(logger.critical('[read_path] directory provided does not exist:\n{}'
                                  .format(read_path)))
            raise (SystemExit)
        
    
    # create global lock instance that can be used in [blackbox_reduce] for
    # certain blocks/functions to be accessed by one process at a time
    global lock
    lock = Lock()

    
    # start queue that will contain entries containing the reference
    # image header OBJECT and FILTER values, so that duplicate
    # reference building for the same object and filter by different
    # threads can be avoided
    global ref_ID_filt
    ref_ID_filt = Queue()

    
    # following line shows how shared Array can be initialized
    #count = Array('i', [0, 0, 0], lock=True)

    
    # for both day and night mode, create list of all
    # files present in [read_path], in image type order:
    # bias, dark, flat, object and other
    if image is None and image_list is None:
        biases, darks, flats, objects, others = sort_files(read_path, '*fits*', 
                                                           recursive=recursive)
        lists = [biases, darks, flats, objects, others]
        filenames = [name for sublist in lists for name in sublist]
    else:
        if mode == 'night':
            q.put(logger.critical('[image] or [image_list] should not be defined '
                                  'in night mode'))
            raise (SystemExit)
        
        elif image is not None:
            # if input parameter [image] is defined, the filenames
            # to process will contain a single image
            filenames = [image]
        elif image_list is not None:
            # if input parameter [image_list] is defined, 
            # read the ascii files into filenames list
            f = open(image_list, 'r')
            filenames = [name.strip() for name in f]
            f.close()
            
            
    # split into 'day' or 'night' mode
    if mode == 'day':

        if len(filenames)==0:
            q.put(logger.warning('no files to reduce'))


        # see https://docs.python.org/3/library/tracemalloc.html
        #snapshot1 = tracemalloc.take_snapshot()
            
        if get_par(set_bb.nproc,tel)==1 or image is not None:
            
            # if only 1 process is requested, or [image] input
            # parameter is not None, run it witout multiprocessing;
            # this will allow images to be shown on the fly if
            # [set_zogy.display] is set to True. In multiprocessing
            # mode this is not allowed (at least not on a macbook).
            print ('running with single processor')
            for filename in filenames:
                result = blackbox_reduce(filename)

        else:
            # use [pool_func] to process list of files
            result = pool_func (try_blackbox_reduce, filenames)


        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
        #    print(stat)


    elif mode == 'night':

        # if in night mode, check if anything changes in input directory
        # and if there is a new file, feed it to [blackbox_reduce]

        # create queue for submitting jobs
        queue = Queue()
        # create pool with given number of processes and queue feeding
        # into action function
        pool = Pool(get_par(set_bb.nproc,tel), action, (queue,))

        # create and setup observer, but do not start just yet
        observer = Observer()
        observer.schedule(FileWatcher(queue), read_path, recursive=recursive)

        # add files that are already present in the read_path
        # directory to the night queue, to reduce these first
        for filename in filenames: 
            queue.put(filename)
            
        # determine time of next sunrise
        obs = ephem.Observer()
        obs.lat = str(get_par(set_zogy.obs_lat,tel))
        obs.lon = str(get_par(set_zogy.obs_lon,tel))
        sunrise = obs.next_rising(ephem.Sun())

        # start observer
        observer.start()

        # keep monitoring [read_path] directory as long as:
        while ephem.now()-sunrise < ephem.hour:
            time.sleep(1)

        # night has finished, but finish queue if not empty yet
        while not queue.empty:
            time.sleep(1)

        # all done!
        q.put(logger.info('stopping time reached, exiting night mode'))
        observer.stop() #stop observer
        observer.join() #join observer


    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t_run_blackbox, label='run_blackbox before fpacking', log=logger)
        

    # do not execute rest of [run_blackbox] if single image was
    # specified to be processed with [image]
    if image is None:

        # remaining master files to prepare
        # ---------------------------------
        # using function [create_masters], make master bias and flat for
        # directories where reduced images are available, but no master
        # was created yet; this is not done with pool of processes,
        # because it is not time-critical and takes 10GB of memory for
        # each set of 10 biases or flats
        q.put(logger.info('looking for and preparing missing master frames'))
        create_masters ('bias')
        for filt in get_par(set_zogy.zp_default,tel).keys():
            create_masters ('flat', filt=filt)
        

        # fpack remaining fits images
        # ---------------------------       
        q.put(logger.info('fpacking fits images'))
        # now that all files have been processed, fpack!
        # create list of files to fpack
        list_2pack = prep_packlist (date)
        #print ('list_2pack', list_2pack)
        # use [pool_func] to process the list
        result = pool_func (fpack, list_2pack)


        # create jpg images for all reduced frames
        # ----------------------------------------
        q.put(logger.info('creating jpg images'))
        # create list of files to jpg
        list_2jpg = prep_jpglist (date)
        # use [pool_func] to process the list
        result = pool_func (create_jpg, list_2jpg)


        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t_run_blackbox, label='run_blackbox after fpacking', 
                               log=logger)


    logging.shutdown()
    return


################################################################################

def create_masters (imtype, filt=None, mdate=None):
    # prepare list of all red/yyyy/mm/dd/bias and flat directories
    red_dir = get_par(set_bb.red_dir,tel)
    # if mdate is not specified, loop all available [imtype]
    # folders in the reduced path
    year, month, day = '*', '*', '*'
    if mdate is not None:
        mdate = ''.join(e for e in mdate if e.isdigit())
        # if [mdate] is specified, only loop [imtype] folders in
        # the reduced path of specific year, month and/or day
        if len(mdate) >= 4:
            year = mdate[0:4]
            if len(mdate) >= 6:
                month = mdate[4:6]
                if len(mdate) >= 8:
                    day = mdate[6:8]
                    
    list_dirs = glob.glob('{}/{}/{}/{}/{}'.format(red_dir, year, month, day, imtype))
    # data shape is needed as input for [master_prep]
    data_shape = (get_par(set_bb.ny,tel) * get_par(set_bb.ysize_chan,tel), 
                  get_par(set_bb.nx,tel) * get_par(set_bb.xsize_chan,tel))
    # loop directories
    for path in list_dirs:
        date_eve = ''.join(path.split('/')[-4:-1])
        if imtype=='bias':
            # name master bias
            fits_master = '{}/{}_{}.fits'.format(path, imtype, date_eve)
        elif imtype=='flat':
            # name master flat
            fits_master = '{}/{}_{}_{}.fits'.format(path, imtype, date_eve, filt)

        # master bias/flat already present?
        master_present = (os.path.isfile(fits_master) or
                          os.path.isfile('{}.fz'.format(fits_master)))

        # if not, prepare it
        if not master_present:
            __ = master_prep (data_shape, path, date_eve, imtype, filt=filt)
                

    return


################################################################################

def already_exists (filename, get_filename=False):
    
    file_list = [filename, '{}.fz'.format(filename), '{}.gz'.format(filename),
                 filename.replace('.fz',''), filename.replace('.gz','')]
    
    exists = False
    existing_file = filename
    
    for file_temp in file_list:
        if os.path.isfile(file_temp):
            exists = True
            existing_file = file_temp
            break

    if get_filename:
        return exists, existing_file
    else:
        return exists
    

################################################################################

def pool_func (func, filelist, *args):

    try:
        results = []
        pool = Pool(get_par(set_bb.nproc,tel))
        for filename in filelist:
            args_temp = [filename]
            for arg in args:
                args_temp.append(arg)
            results.append(pool.apply_async(func, args_temp))

        pool.close()
        pool.join()
        results = [r.get() for r in results]
        #q.put(logger.info('result from pool.apply_async: {}'.format(results)))
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [pool.apply_async({})]: {}'
                           .format(func, e)))
        raise SystemExit
        

################################################################################

def prep_packlist (date):
    
    list_2pack = []
    if date is not None:
        # add files in [read_path]
        read_path, __ = get_path(date, 'read')
        list_2pack.append(glob.glob('{}/*.fits'.format(read_path)))
        # add files in [write_path]
        write_path, __ = get_path(date, 'write')
        list_2pack.append(glob.glob('{}/*.fits'.format(write_path)))
        # add fits files in bias and flat directories
        list_2pack.append(glob.glob('{}/bias/*.fits'.format(write_path)))
        list_2pack.append(glob.glob('{}/flat/*.fits'.format(write_path)))
    else:
        # just add all fits files in [set_bb.raw_dir]/*/*/*/*.fits
        raw_dir = get_par(set_bb.raw_dir,tel)
        list_2pack.append(glob.glob('{}/*/*/*/*.fits'.format(raw_dir)))
        # just add all fits files in [telescope]/red/*/*/*/*.fits
        # (could do it more specifically by going through raw fits
        # files and finding out where their reduced images are)
        red_dir = get_par(set_bb.red_dir,tel)
        list_2pack.append(glob.glob('{}/*/*/*/*.fits'.format(red_dir)))
        # add fits files in bias and flat directories
        list_2pack.append(glob.glob('{}/*/*/*/bias/*.fits'.format(red_dir)))
        list_2pack.append(glob.glob('{}/*/*/*/flat/*.fits'.format(red_dir)))


    # ref files will be fpacked by the reference building module
    # add ref files
    #ref_dir = get_par(set_bb.ref_dir,tel)
    #list_2pack.append(glob.glob('{}/*/*.fits'.format(ref_dir)))

    
    # add tmp folders if they are kept
    if get_par(set_bb.keep_tmp,tel):
        tmp_dir = get_par(set_bb.tmp_dir,tel)
        list_2pack.append(glob.glob('{}/*/*.fits'.format(tmp_dir)))

        
    # flatten this list of files
    list_2pack = [item for sublist in list_2pack for item in sublist]
    # and get the unique items
    list_2pack = list(set(list_2pack))

    return list_2pack
    

################################################################################

def fpack (filename):

    """Fpack fits images; skip fits tables"""

    try:
    
        # fits check if extension is .fits and not an LDAC fits file
        if filename.split('.')[-1] == 'fits' and '_ldac.fits' not in filename:
            header = read_hdulist(filename, get_data=False, get_header=True,
                                  ext_name_indices=0)

            # check if it is an image
            if header['NAXIS']==2:
                # determine if integer or float image
                if header['BITPIX'] > 0:
                    cmd = ['fpack', '-D', '-Y', '-v', filename]
                else:
                    if 'Scorr' in filename or 'limmag' in filename:
                        quant = 1
                    else:
                        quant = 16
                    cmd = ['fpack', '-q', str(quant), '-D', '-Y', '-v', filename]
                subprocess.call(cmd)

    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised in fpacking of image {} {}'
                           .format(filename,e)))


################################################################################

def prep_jpglist (date):
    
    list_2jpg = []
    if date is not None:
        # add files in [write_path]
        write_path, __ = get_path(date, 'write')
        list_2jpg.append(glob.glob('{}/*_red.fits.fz'.format(write_path)))
        # add fits files in bias and flat directories
        list_2jpg.append(glob.glob('{}/bias/*.fits.fz'.format(write_path)))
        list_2jpg.append(glob.glob('{}/flat/*.fits.fz'.format(write_path)))
    else:
        # just add all fits files in [telescope]/red/*/*/*/*.fits
        # (could do it more specifically by going through raw fits
        # files and finding out where their reduced images are)
        red_dir = get_par(set_bb.red_dir,tel)
        list_2jpg.append(glob.glob('{}/*/*/*/*_red.fits.fz'.format(red_dir)))
        # add fits files in bias and flat directories
        list_2jpg.append(glob.glob('{}/*/*/*/bias/*.fits.fz'.format(red_dir)))
        list_2jpg.append(glob.glob('{}/*/*/*/flat/*.fits.fz'.format(red_dir)))

        
    # flatten this list of files
    list_2jpg = [item for sublist in list_2jpg for item in sublist]
    # and get the unique items
    list_2jpg = list(set(list_2jpg))

    return list_2jpg
    

################################################################################

def create_jpg (filename):

    """Create jpg image from fits"""

    try:
        
        image_jpg = '{}.jpg'.format(filename.split('.fits')[0])

        if os.path.isfile(image_jpg):
            
            q.put(logger.info('{} already exists; skipping {}'
                              .format(image_jpg, filename)))

        else:

            q.put(logger.info('saving {} to {}'.format(filename, image_jpg)))
              
            # read input image
            data, header = read_hdulist(filename, get_header=True)
        
            imgtype = header['imagetyp'].lower()
            file_str = image_jpg.split('/')[-1].split('.jpg')[0]
            if imgtype == 'object':
                title = ('file:{}   object:{}   filter:{}   exptime:{:.1f}s'
                         .format(file_str, header['object'],
                                 header['filter'], header['exptime']))
            else:
                title = ('file:{}   imgtype:{}   filter:{}'
                         .format(file_str, header['imagetyp'], header['filter']))

            pixelcoords = True
            if pixelcoords:
                f = aplpy.FITSFigure(data)
            else:
                f = aplpy.FITSFigure(filename)

            f.show_colorscale(cmap='gray', pmin=5, pmax=95)
            f.add_colorbar()
            f.set_title(title)
            #f.add_grid()
            #f.set_theme('pretty')
            f.save(image_jpg)
            f.close()
            

    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised in creating jpg of image {} {}'
                           .format(filename,e)))


################################################################################

class WrapException(Exception):
    """see https://bugs.python.org/issue13831Ups"""

    def __init__(self):
        exc_type, exc_value, exc_tb = sys.exc_info()
        self.exception = exc_value
        self.formatted = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    def __str__(self):
        return '{}\nOriginal traceback:\n{}'.format(Exception.__str__(self), self.formatted)
        

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
        blackbox_reduce (filename)
    except:
        raise WrapException()
        
    
################################################################################
    
def blackbox_reduce (filename):

    """Function that takes as input a single raw fits image and works to
       work through entire chain of reduction steps, from correcting
       for the gain and overscan to running ZOGY on the reduced image.

    """

    if get_par(set_zogy.timing,tel):
        t_blackbox_reduce = time.time()


    q.put(logger.info('processing {}'.format(filename)))


    # just read the header for the moment
    try:
        header = read_hdulist(filename, get_data=False, get_header=True)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised in read_hdulist at top of '
                           '[blackbox_reduce]: {}'.format(e)))
        q.put(logger.error('not processing {}'.format(filename)))    
        return

     
    # first header check using function [check_header1]
    header_ok = check_header1 (header, filename)
    if not header_ok:
        return
   
    
    # determine the raw data path
    raw_path, __ = get_path(header['DATE-OBS'], 'read')


    # move or copy the image over to [raw_path] if it does not already exist
    src = filename
    dest = '{}/{}'.format(raw_path, filename.split('/')[-1])
    if already_exists (dest):
        q.put(logger.warning('{} already exists; not copying/moving file'.format(dest)))
    else:
        make_dir (raw_path)
        # moving:
        #shutil.move(src, dest)
        # copying:
        shutil.copy2(src, dest)


    # and let [filename] refer to the image in [raw_path]
    filename = dest

    
    # check if all crucial keywwords are present in the header
    qc_flag = run_qc_check (header, tel, log=logger)
    if qc_flag=='no_object':
        q.put(logger.error('keyword OBJECT not in header of object image; '
                           ' skipping image'))
        return

    
    if qc_flag=='red':
        q.put(logger.error('red QC flag in image {}; returning without making '
                           ' dummy catalogs'.format(filename)))
        return
    

    # if 'IMAGETYP' keyword not one of those specified in input parameter
    # [imgtypes] or complete set: ['bias', 'dark', 'flat', 'object']
    if types is not None:
        imgtypes2process = types
    else:
        imgtypes2process = ['bias', 'dark', 'flat', 'object']
    # then also return
    imgtype = header['IMAGETYP'].lower()
    if imgtype not in imgtypes2process:
        q.put(logger.warning('image type ({}) not in [imgtypes] ({}); '
                             'not processing {}'
                             .format(imgtype, imgtypes2process, filename)))
        return
    
    
    # extend the header with some useful/required keywords
    try: 
        header = set_header(header, filename)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [set_header]: {}'.format(e)))
        q.put(logger.error('returning without making dummy catalogs'))
        return
    
    
    # 2nd header check using function [check_header2]
    header_ok = check_header2 (header, filename)
    if not header_ok:
        return
    
    
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

    # UT date (yyyymmdd) and time (hhmmss)
    utdate, uttime = get_date_time(header)

    # if output file already exists, do not bother to redo it
    path = {'bias': bias_path, 
            'dark': dark_path, 
            'flat': flat_path, 
            'object': write_path}
    filt = header['FILTER']

    # if exptime is not in the header or if it's 0 for a science
    # image, skip image
    if 'EXPTIME' in header:
        exptime = header['EXPTIME']
        if 'IMAGETYP' in header and (header['IMAGETYP'].lower()=='object'
            and int(exptime)==0):
            q.put(logger.error('science image with EXPTIME of zero; skipping image'))
            return
    else:
        q.put(logger.warning('keyword EXPTIME not in header; skipping image'))
        return


    # if [only_filt] is specified, skip image if not relevant
    if filts is not None:
        if filt not in filts and imgtype != 'bias' and imgtype != 'dark':
            q.put(logger.warning('image filter ({}) not in [only_filters] ({}); '
                                 'not processing {}'
                                 .format(filt, filts, filename)))
            return
            
    fits_out = '{}/{}_{}_{}.fits'.format(path[imgtype], tel, utdate, uttime)

    if imgtype == 'bias':
        make_dir (bias_path)

    elif imgtype == 'dark':
        make_dir (dark_path)
        
    elif imgtype == 'flat':
        make_dir (flat_path)
        fits_out = fits_out.replace('.fits', '_{}.fits'.format(filt))

    elif imgtype == 'object':
        # if 'FIELD_ID' keyword is present in the header, use it instead of OBJECT
        # as is the case for the test data
        if 'FIELD_ID' in header:
            obj = header['FIELD_ID']
        else:
            obj = header['OBJECT']

        # remove all non-alphanumeric characters from [obj] except for
        # '-' and '_'
        #obj = ''.join(e for e in obj if e.isalnum() or e=='-' or e=='_')
        # now assuming field_id or object is ML/BG field number 
        # pad with zeros
        obj = '{:0>5}'.format(obj)
                
        fits_out = fits_out.replace('.fits', '_red.fits')
        fits_out_mask = fits_out.replace('_red.fits', '_mask.fits')
        
        # and reference image
        ref_path = '{}/{}'.format(get_par(set_bb.ref_dir,tel), obj)
        make_dir (ref_path)
        ref_fits_out = '{}/{}_{}_red.fits'.format(ref_path, tel, filt)
        
        exists, ref_fits_temp = already_exists (ref_fits_out, get_filename=True)
        if exists:
            header_ref = read_hdulist(ref_fits_temp, get_data=False,
                                      get_header=True)
            utdate_ref, uttime_ref = get_date_time(header_ref)
            if utdate_ref==utdate and uttime_ref==uttime:
                q.put(logger.info ('this image {} is the current reference image '
                                   'of field {}; skipping'
                                   .format(fits_out.split('/')[-1], obj)))
                return


    if already_exists (fits_out):
        q.put(logger.warning ('corresponding reduced {} image {} already exist; '
                              'skipping'.format(imgtype, fits_out.split('/')[-1])))
        return

    
    q.put(logger.info('\nprocessing {}'.format(filename)))
    #q.put(logger.info('-'*(len(filename)+11)))

    if imgtype == 'object':
        # prepare directory to store temporary files related to this
        # OBJECT image.  This is set to the tmp directory defined by
        # [set_bb.tmp_dir] with subdirectory the name of the reduced
        # image without the .fits extension.
        tmp_path = '{}/{}'.format(get_par(set_bb.tmp_dir,tel),
                                  fits_out.split('/')[-1].replace('.fits',''))
        make_dir (tmp_path, empty=True)
        
        
    # now that output filename is known, create a logger that will
    # append the log commands to [logfile]
    if imgtype != 'object':
        # for biases, darks and flats
        logfile = fits_out.replace('.fits','.log')
    else:
        # for object files, prepare the logfile in [tmp_path]
        logfile = '{}/{}'.format(tmp_path, fits_out.split('/')[-1]
                                 .replace('.fits','.log'))
    global log
    log = create_log (logfile)

    # immediately write some info to the log
    log.info('processing {}'.format(filename))
    log.info('image type: {}, filter: {}, exptime: {:.1f}s'
             .format(imgtype, filt, exptime))

    log.info('write_path: {}'.format(write_path))
    log.info('bias_path: {}'.format(bias_path))
    log.info('dark_path: {}'.format(dark_path))
    log.info('flat_path: {}'.format(flat_path))
    if imgtype == 'object':
        log.info('tmp_path: {}'.format(tmp_path))
        log.info('ref_path: {}'.format(ref_path))

                                  
    # general log file
    header['LOG'] = (genlogfile.split('/')[-1], 'name general logfile')
    # image log file
    header['LOG-IMA'] = (logfile.split('/')[-1], 'name image logfile')
    # reduced filename
    
    # now also read in the data
    data = read_hdulist(filename, dtype='float32')   

    # determine number of pixels with infinite/nan values
    mask_infnan = ~np.isfinite(data)
    n_infnan = np.sum(mask_infnan)
    header['N-INFNAN'] = (n_infnan, 'number of pixels with infinite/nan values')
    if n_infnan > 0:
        log.info('warning: {} pixels with infinite/nan values; replacing with zero'
                 .format(n_infnan))
        data[mask_infnan] = 0
        
    
    #snapshot1 = tracemalloc.take_snapshot()

    # gain correction
    #################
    try:
        log.info('correcting for the gain')
        gain_processed = False
        data = gain_corr(data, header, tel=tel, log=log)
    except Exception as e:
        #q.put(logger.info(traceback.format_exc()))
        #q.put(logger.error('exception was raised during [gain_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [gain_corr]: {}'.format(e))
    else:
        gain_processed = True
    finally:
        header['GAIN'] = (1, '[e-/ADU] effective gain all channels')
        header['GAIN-P'] = (gain_processed, 'corrected for gain?')

        
    #snapshot2 = tracemalloc.take_snapshot()
    #top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    #print("[ Top 10 differences ]")
    #for stat in top_stats[:10]:
    #    print(stat)

    if get_par(set_zogy.display,tel):
        ds9_arrays(gain_cor=data)

    
    # crosstalk correction
    ######################
    if imgtype == 'object':
        # not needed for biases, darks or flats
        try: 
            log.info('correcting for the crosstalk')
            xtalk_processed = False
            crosstalk_file = get_par(set_bb.crosstalk_file,tel)
            data = xtalk_corr (data, crosstalk_file)
        except Exception as e:
            q.put(logger.info(traceback.format_exc()))
            q.put(logger.error('exception was raised during [xtalk_corr]: {}'
                               .format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during [xtalk_corr]: {}'.format(e))
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
        data = os_corr(data, header, imgtype, tel=tel, log=log)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [os_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [os_corr]: {}'.format(e))
    else:
        os_processed = True
    finally:
        header['OS-P'] = (os_processed, 'corrected for overscan?')
        

    if get_par(set_zogy.display,tel):
        ds9_arrays(os_cor=data)


    # non-linearity correction
    ##########################
    nonlin_corr_processed = False
    header['NONLIN-P'] = (nonlin_corr_processed, 'corrected for non-linearity?')
    header['NONLIN-F'] = ('None', 'name non-linearity correction file')
    
    if imgtype != 'bias' and get_par(set_bb.correct_nonlin,tel):

        try:
            log.info('correcting for the non-linearity')
            nonlin_corr_file = get_par(set_bb.nonlin_corr_file,tel)
            data = nonlin_corr(data, nonlin_corr_file, log=log)
            header['NONLIN-F'] = (nonlin_corr_file.split('/')[-1],
                                  'name non-linearity correction file')
        except Exception as e:
            #q.put(logger.info(traceback.format_exc()))
            #q.put(logger.error('exception was raised during [nonlin_corr]: '
            #                   '{}'.format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during [nonlin_corr]: {}'
                      .format(e))
        else:
            nonlin_corr_processed = True
        finally:
            header['NONLIN-P'] = nonlin_corr_processed


    # if IMAGETYP=bias or dark, write [data] to fits and return
    if imgtype == 'bias' or imgtype == 'dark':
        # call [run_qc_check] to update header with any QC flags
        run_qc_check (header, tel, log=log)
        header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(fits_out, data.astype('float32'), header, overwrite=True)
        return
    
    
    # master bias creation
    ######################
    try:
        # put an multi-processing lock on this block so that only 1
        # process at a time can create the master bias
        lock.acquire()

        # prepare or point to the master bias
        fits_mbias = master_prep (np.shape(data), bias_path, date_eve, 'bias', 
                                  log=log)

    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised in bias [master_prep]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during bias [master_prep]: {}'.format(e))

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
            data_mbias, header_mbias = read_hdulist(fits_mbias, get_header=True)
            data -= data_mbias
            header['MBIAS-F'] = fits_mbias.split('/')[-1].split('.fits')[0]

            # for object image, add number of days separating image and master bias
            if imgtype == 'object':
                mjd_obs = header['MJD-OBS']
                mjd_obs_mb = header_mbias['MJD-OBS']
                header['MB-NDAYS'] = (
                    np.abs(mjd_obs-mjd_obs_mb), 
                    '[days] time between image and master bias used')

        except Exception as e:
            q.put(logger.info(traceback.format_exc()))
            q.put(logger.error('exception was raised during master bias subtraction: {}'
                               .format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during master bias subtraction: {}'.format(e))
        else:
            mbias_processed = True
        finally:
            header['MBIAS-P'] = mbias_processed


    # display
    if get_par(set_zogy.display,tel):
        ds9_arrays(bias_sub=data)
        

    # create initial mask array
    ###########################
    if imgtype == 'object':
        try:
            log.info('preparing the initial mask')
            mask_processed = False
            data_mask, header_mask = mask_init (data, header)
        except Exception as e:
            q.put(logger.info(traceback.format_exc()))
            q.put(logger.error('exception was raised during [mask_init]: {}'.format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during [mask_init]: {}'.format(e))
        else:
            mask_processed = True
        finally:
            header['MASK-P'] = (mask_processed, 'mask image created?')


        if get_par(set_zogy.display,tel):
            ds9_arrays(mask=data_mask)


        # set edge pixel values to zero
        data[data_mask==get_par(set_zogy.mask_value['edge'],tel)] = 0


    # if IMAGETYP=flat, write [data] to fits and return
    if imgtype == 'flat':
        # first add some statistics to header
        if os_processed:
            get_flatstats (data, header, imgtype)
        # call [run_qc_check] to update header with any QC flags
        run_qc_check (header, tel, log=log)
        # write to fits
        header['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
        fits.writeto(fits_out, data.astype('float32'), header, overwrite=True)
        return


    # master flat creation
    ######################
    try:
        # put an multi-processing lock on this block so that only 1
        # process at a time can create the master flat
        lock.acquire()

        # prepare or point to the master flat
        fits_mflat = master_prep(np.shape(data), flat_path, date_eve, 'flat',
                                 filt=filt, log=log)

    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during flat [master_prep]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during flat [master_prep]: {}'.format(e))

    finally:
        lock.release()

        
    # master flat division
    ######################
    mflat_processed = False
    header['MFLAT-F'] = ('None', 'name of master flat applied')

    if fits_mflat is not None:
        try:
            # and divide the object image by the master flat
            log.info('dividing by the master flat')
            data_mflat, header_mflat = read_hdulist(fits_mflat, get_header=True)
            data /= data_mflat
            header['MFLAT-F'] = (fits_mflat.split('/')[-1].split('.fits')[0],
                                 'name of master flat applied')
            # for object image, add number of days separating image and master bias
            if imgtype == 'object':
                mjd_obs = header['MJD-OBS']
                mjd_obs_mf = header_mflat['MJD-OBS']
                header['MF-NDAYS'] = (
                    np.abs(mjd_obs-mjd_obs_mf), 
                    '[days] time between image and master flat used')

        except Exception as e:
            q.put(logger.info(traceback.format_exc()))
            q.put(logger.error('exception was raised during master flat division: {}'
                               .format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during master flat division: {}'.format(e))
        else:
            mflat_processed = True
        finally:
            header['MFLAT-P'] = (mflat_processed, 'corrected for master flat?')



    # PMV 2018/12/20: fringe correction is not yet done, but
    # still add these keywords to the header
    header['MFRING-P'] = (False, 'corrected for master fringe map?')
    header['MFRING-F'] = ('None', 'name of master fringe map applied')


    if get_par(set_zogy.display,tel):
        ds9_arrays(flat_cor=data)
        data_precosmics = np.copy(data)


    # cosmic ray detection and correction
    #####################################
    try:
        log.info('detecting cosmic rays')
        cosmics_processed = False
        data, data_mask = cosmics_corr(data, header, data_mask, header_mask)
    except Exception as e:
        header['NCOSMICS'] = ('None', '[/s] number of cosmic rays identified')
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [cosmics_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [cosmics_corr]: {}'.format(e))
    else:
        cosmics_processed = True
    finally:
        header['COSMIC-P'] = (cosmics_processed, 'corrected for cosmic rays?')

    
    if get_par(set_zogy.display,tel):
        ds9_arrays(data=data_precosmics, cosmic_cor=data, mask=data_mask)
        print (header['NCOSMICS'])
        

    # satellite trail detection
    ###########################
    try: 
        sat_processed = False
        if get_par(set_bb.detect_sats,tel):
            log.info('detecting satellite trails')
            data_mask = sat_detect(data, header, data_mask, header_mask,
                                   tmp_path)
    except Exception as e:
        header['NSATS'] = ('None', 'number of satellite trails identified')
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [sat_detect]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [sat_detect]: {}'.format(e))
    else:
        if get_par(set_bb.detect_sats,tel):
            sat_processed = True
    finally:
        header['SAT-P'] = (sat_processed, 'processed for satellite trails?')


    # add some more info to mask header
    result = mask_header(data_mask, header_mask)

    if get_par(set_zogy.display,tel):
        ds9_arrays(mask=data_mask)
        print (header['NSATS'])

    # output images and catalogs to refer to [tmp] directory
    new_fits = '{}/{}'.format(tmp_path, fits_out.split('/')[-1])
    new_fits_mask = new_fits.replace('_red.fits', '_mask.fits')
    fits_tmp_cat = new_fits.replace('.fits', '_cat.fits')
    fits_tmp_trans = new_fits.replace('.fits', '_trans.fits')

    tmp_base = new_fits.split('_red.fits')[0]
    new_base = fits_out.split('_red.fits')[0]

    # if header of object image contains a red flag, create dummy
    # binary fits catalogs (both 'new' and 'trans') and return,
    # skipping zogy's [optimal subtraction] below
    qc_flag = run_qc_check (header, tel, log=log)
    if qc_flag=='red':
        log.error('red QC flag in image {}; making dummy catalogs and returning'
                  .format(fits_out))
        run_qc_check (header, tel, 'new', fits_tmp_cat, log=log)
        run_qc_check (header, tel, 'trans', fits_tmp_trans, log=log)

        # copy selected output files to new directory and remove tmp folder
        # corresponding to the object image
        result = copy_files2keep(tmp_base, new_base, get_par(set_bb.all_2keep,tel),
                                 move=False, log=log)
        clean_tmp(tmp_base)

        return

    
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


    # run zogy's [optimal_subtraction]
    ##################################
    log.info ('running optimal image subtraction')
        
    # using the function [check_ref], check if the reference image
    # with the same header OBJECT and FILTER as the currently
    # processed image happens to be made right now, using a lock
    lock.acquire()

    # change to [tmp_path]; only necessary if making plots as
    # PSFEx is producing its diagnostic output fits and plots in
    # the current directory
    if get_par(set_zogy.make_plots,tel):
        os.chdir(tmp_path)
        
    # this extra second is to provide a head start to the process
    # that is supposed to be making the reference image; that
    # process needs to add its OBJECT and FILTER to the queue
    # [ref_ID_filt] before the next process is calling [check_ref]
    time.sleep(1)
    ref_being_made = check_ref(ref_ID_filt, (obj, filt), put_lock=False)
    log.info ('is reference image for same OBJECT: {} and FILTER: {} '
              'being made now?: {}'.format(obj, filt, ref_being_made))

    # release lock
    lock.release()
    
    if ref_being_made:
        # if reference in this filter is being made, let the affected
        # process wait until reference building is done
        while True:
            log.info ('waiting for reference image to be made for '
                      'OBJECT: {}, FILTER: {}'.format(obj, filt))
            time.sleep(5)
            if not check_ref(ref_ID_filt, (obj, filt)):
                break
        log.info ('done waiting for reference image to be made for '
                  'OBJECT: {}, FILTER: {}'.format(obj, filt))
        

    # if ref image has not yet been processed:
    ref_present = already_exists (ref_fits_out)
    log.info('has reference image: {} been made already?: {}'
             .format(ref_fits_out, ref_present))
                         
    if not ref_present:
        # update [ref_ID_filt] queue with a tuple with this OBJECT
        # and FILTER combination
        ref_ID_filt.put((obj, filt))
        
        log.info('making reference image: {}'.format(ref_fits_out))
        log.info('new_fits: {}'.format(new_fits))
        log.info('new_fits_mask: {}'.format(new_fits_mask))

        
        try:
            zogy_processed = False
            header_optsub = optimal_subtraction(
                ref_fits=new_fits, ref_fits_mask=new_fits_mask, 
                set_file='set_zogy', log=log, verbose=None,
                nthread=get_par(set_bb.nthread,tel), telescope=tel)
        except Exception as e:
            log.info(traceback.format_exc())
            log.error('exception was raised during reference [optimal_subtraction]: {}'
                      .format(e))
        else:
            zogy_processed = True
        finally:
            if not zogy_processed:
                log.error('due to exception: returning without copying reference files')

                # copy selected output files to red directory and remove tmp folder
                # corresponding to the image
                result = copy_files2keep(tmp_base, new_base, get_par(set_bb.all_2keep,tel),
                                         move=False, log=log)
                clean_tmp(tmp_base)

                return
            else:
                # feed [header_optsub] to [run_qc_check], and make
                # dummy catalogs if there is a red flag
                qc_flag = run_qc_check (header_optsub, tel, log=log)
                if qc_flag=='red':
                    log.error('red QC flag in [header_optsub] returned by reference '
                              '[optimal_subtraction]; making dummy catalogs as if it '
                              'were a new image')
                    run_qc_check (header_optsub, tel, 'new', fits_tmp_cat, log=log)
                    run_qc_check (header_optsub, tel, 'trans', fits_tmp_trans, log=log)

                else:
                    # update catalog header with latest qc-flags
                    with fits.open(fits_tmp_cat, 'update') as hdulist:
                        for key in header_optsub:
                            if 'QC' in key:
                                hdulist[-1].header[key] = (
                                    header_optsub[key], header_optsub.comments[key])


        # update reduced image header with extended header from ZOGY [header_optsub]
        with fits.open(new_fits, 'update') as hdulist:
            hdulist[0].header = header_optsub

            
        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t_blackbox_reduce, label='blackbox_reduce',
                               log=log)
                
        # copy the set of files [all_2keep] to the reduced
        # directory for future building of new reference image
        result = copy_files2keep(tmp_base, new_base, 
                                 get_par(set_bb.all_2keep,tel), move=False,
                                 log=log)

        if qc_flag == 'red':
            # also copy dummy transient catalog in case of red flag
            result = copy_files2keep(tmp_base, new_base, ['_trans.fits'],
                                     log=log)
            clean_tmp(tmp_base)
        else:
            # now move [ref_2keep] to the reference directory
            ref_base = ref_fits_out.split('_red.fits')[0]
            result = copy_files2keep(tmp_base, ref_base,
                                     get_par(set_bb.ref_2keep,tel),
                                     move=False, log=log)
            clean_tmp(tmp_base)
            
        # now that reference is built, remove this reference ID
        # and filter combination from the [ref_ID_filt] queue
        result = check_ref(ref_ID_filt, (obj, filt), method='remove')

        if qc_flag != 'red':
            log.info('finished making reference image: {}'.format(ref_fits_out))
        else:
            log.info('encountered red flag; not using image: {} (original name: {}) '
                     'as reference'.format(header['REDFILE'], header['ORIGFILE']))
            
    else:

        # make symbolic links to all files in the reference
        # directory with the same filter
        ref_files = glob.glob('{}/{}_{}_*'.format(ref_path, tel, filt))
        for ref_file in ref_files:
            # and funpack/unzip if necessary (before symbolic link
            # to avoid unpacking multiple times during the same night)
            ref_file = unzip(ref_file)
            # and create symbolic link
            os.symlink(ref_file, '{}/{}'.format(tmp_path, ref_file.split('/')[-1]))
                
        ref_fits = '{}/{}'.format(tmp_path, ref_fits_out.split('/')[-1])
        ref_fits_mask = ref_fits.replace('_red.fits', '_mask.fits')
        
        log.info('new_fits: {}'.format(new_fits))
        log.info('new_fits_mask: {}'.format(new_fits_mask))
        log.info('ref_fits: {}'.format(ref_fits))
        log.info('ref_fits_mask: {}'.format(ref_fits_mask))


        try:
            zogy_processed = False
            header_optsub = optimal_subtraction(
                new_fits=new_fits, ref_fits=ref_fits, new_fits_mask=new_fits_mask,
                ref_fits_mask=ref_fits_mask, set_file='set_zogy', log=log, 
                verbose=None, nthread=get_par(set_bb.nthread,tel), telescope=tel)
        except Exception as e:
            log.info(traceback.format_exc())
            log.error('exception was raised during new [optimal_subtraction]: {}'.format(e))
        else:
            zogy_processed = True
        finally:
            if not zogy_processed:
                log.error('due to exception: returning without copying new files')

                # copy selected output files to red directory and remove tmp folder
                # corresponding to the image
                result = copy_files2keep(tmp_base, new_base, get_par(set_bb.all_2keep,tel),
                                         move=False, log=log)
                clean_tmp(tmp_base)

                return
            else:
                # feed [header_optsub] to [run_qc_check], and if there
                # is a red flag make output dummy catalog and return
                qc_flag = run_qc_check (header_optsub, tel, log=log)
                if qc_flag=='red':
                    log.error('red QC flag in [header_optsub] returned by new '
                              '[optimal_subtraction]: making dummy catalogs')
                    run_qc_check (header_optsub, tel, 'new', fits_tmp_cat, log=log)
                    run_qc_check (header_optsub, tel, 'trans', fits_tmp_trans, log=log)

                else:
                    # update catalog header with latest qc-flags
                    with fits.open(fits_tmp_cat, 'update') as hdulist:
                        for key in header_optsub:
                            if 'QC' in key:
                                hdulist[-1].header[key] = (
                                    header_optsub[key], header_optsub.comments[key])
                    # same for transient catalog header
                    with fits.open(fits_tmp_trans, 'update') as hdulist:
                        for key in header_optsub:
                            if 'QC' in key:
                                hdulist[-1].header[key] = (
                                    header_optsub[key], header_optsub.comments[key])


        # update reduced image header with extended header from ZOGY [header_optsub]
        with fits.open(new_fits, 'update') as hdulist:
            hdulist[0].header = header_optsub
        
                    
        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t_blackbox_reduce, label='blackbox_reduce', log=log)

        # copy selected output files to new directory
        result = copy_files2keep(tmp_base, new_base,
                                 get_par(set_bb.new_2keep,tel),
                                 move=False, log=log)
        clean_tmp(tmp_base)

    log.info('reached the end of function blackbox_reduce')
    
    return


################################################################################

def get_flatstats (data, header, imgtype, nsubs_side=6):
    
    # add some header keywords with the statistics
    sec_temp = get_par(set_bb.flat_norm_sec,tel)
    value_temp = '[{}:{},{}:{}]'.format(
        sec_temp[0].start+1, sec_temp[0].stop+1, 
        sec_temp[1].start+1, sec_temp[1].stop+1) 
    header['STATSEC'] = (
        value_temp, 'pre-defined statistics section [y1:y2,x1:x2]')
    
    # statistics on STATSEC
    index_stat = get_rand_indices(data[sec_temp].shape)
    __, median_sec, std_sec = sigma_clipped_stats(data[sec_temp][index_stat]
                                                  .astype('float64'),
                                                  mask_value=0)
    header['MEDSEC'] = (median_sec, '[e-] median flat over STATSEC')
    header['STDSEC'] = (std_sec, '[e-] sigma (STD) flat over STATSEC')
    header['RSTDSEC'] = (std_sec/median_sec, 'relative sigma (STD) flat '
                         'over STATSEC')

    # full image statistics
    index_stat = get_rand_indices(data.shape)
    __, median, std = sigma_clipped_stats(data[index_stat].astype('float64'),
                                          mask_value=0)
    header['FLATMED'] = (median, '[e-] median flat')
    header['FLATSTD'] = (std, '[e-] sigma (STD) flat')
    header['FLATRSTD'] = (std/median, 'relative sigma (STD) flat')
    
    # add the channel median level to the flatfield header
    chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red = (
        define_sections(np.shape(data), tel=tel))
    nchans = np.shape(data_sec)[0]
    
    for i_chan in range(nchans):
        
        median_temp = np.median(data[data_sec_red[i_chan]])
        header['FLATM{}'.format(i_chan+1)] = (
            median_temp,
            '[e-] channel {} median flat (bias-subtracted)'.format(i_chan+1))
        
        std_temp = np.std(data[data_sec_red[i_chan]])
        header['FLATS{}'.format(i_chan+1)] = (
            std_temp,
            '[e-] channel {} sigma (STD) flat'.format(i_chan+1))
        
        header['FLATRS{}'.format(i_chan+1)] = (
            std_temp/median_temp,
            'channel {} relative sigma (STD) flat'.format(i_chan+1))
        

    # split image in 6x6 subimages and calculate a few additional
    # statistics
    
    # determine boxsize
    ysize, xsize = data.shape
    boxsize = int(ysize / nsubs_side)
    
    # reshape
    data_reshaped = data.reshape(
        nsubs_side,boxsize,-1,boxsize).swapaxes(1,2).reshape(
            nsubs_side,nsubs_side,-1)
    
    # get clipped statistics
    index_stat = get_rand_indices((data_reshaped.shape[2],))
    mini_mean, mini_median, mini_std = sigma_clipped_stats (
        data_reshaped[:,:,index_stat].astype('float64'),
        sigma=3, axis=2, mask_value=0)

    # statistic used by Danielle, or the maximum relative difference
    # between the boxes' medians
    danstat = ((np.amax(mini_median) - np.amin(mini_median)) /
               (np.amax(mini_median) + np.amin(mini_median)))
    
    header['NSUBS'] = (nsubs_side**2, 'number of subimages for statistics')
    header['RDIF-MAX'] = (danstat, '(max(subs)-min(subs)) / (max(subs)+min(subs))')

    mask_nonzero = (mini_median != 0)
    if np.sum(mask_nonzero) != 0:
        rstd = mini_std[mask_nonzero] / mini_median[mask_nonzero]
        index_max = np.argmax(rstd)
        rstd_max = rstd.ravel()[index_max]
        header['RSTD-MAX'] = (rstd_max, 'maximum relative sigma (STD) of subimages')
    
    return


################################################################################

def check_ref (queue_ref, obj_filt, method=None, put_lock=True):

    if put_lock:
        lock.acquire()
        
    # initialize list with copy of queue 
    mycopy = []
    ref_being_made = False

    while True:
        try:
            # get (return and remove) next element from input queue
            # [queue_ref]
            elem = queue_ref.get(False)
        except:
            # if no more elements left, break
            break
        else:
            # add element to copy of queue, unless element matches
            # [obj_filt] and [method] is set to remove
            if not (elem == obj_filt and method=='remove'):
                mycopy.append(elem)

                
    # loop over copy of queue to put elements back into input queue
    # N.B.: input queue is empty at this point
    for elem in mycopy:
        # put element back into original queue
        queue_ref.put(elem, False)
        # if element matches input [obj_filt] (tuple: (object,
        # filter)), reference image is still being made
        if elem == obj_filt:
            ref_being_made = True

    # need to wait a little bit because putting an element in the
    # queue is not instantaneous, and so it could (and does) happen
    # that queue is checked again before element is present
    time.sleep(0.1)

    if put_lock:
        lock.release()

    return ref_being_made

                
################################################################################

def try_func (func, args_in, args_out):

    """Helper function to avoid duplication when executing the different
       functions."""

    func_name = func.__name__

    try: 
        log.info('executing [{}]'.format(func_name))
        proc_ok = False
        args[0] = func (args[1:])
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [{}]: {}'
                           .format(func_name, e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [{}]: {}'
                  .format(func_name, e))
    else:
        proc_ok = True

    return proc_ok

    
################################################################################

def create_log (logfile):

    #log = logging.getLogger() #create logger
    #log.setLevel(logging.INFO) #set level of logger
    #formatter = logging.Formatter("%(asctime)s %(funcName)s %(lineno)d %(levelname)s %(message)s") #set format of logger
    #logging.Formatter.converter = time.gmtime #convert time in logger to UTC
    #filehandler = logging.FileHandler(fits_out.replace('.fits','.log'), 'w+') #create log file
    #filehandler.setFormatter(formatter) #add format to log file
    #log.addHandler(filehandler) #link log file to logger

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s, %(process)s] '
                                     '%(message)s [%(funcName)s, line %(lineno)d]',
                                     '%Y-%m-%dT%H:%M:%S')
    logging.Formatter.converter = time.gmtime #convert time in logger to UTC

    fileHandler = logging.FileHandler(logfile, 'w+')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    log.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logFormatter)
    streamHandler.setLevel(logging.WARNING)
    log.addHandler(streamHandler)

    return log
    

################################################################################

def make_dir(path, empty=False):

    """Wrapper function to lock make_dir_nolock so that it's only used by
       1 process. """

    lock.acquire()
    make_dir_nolock (path, empty)
    lock.release()

    return


################################################################################

def make_dir_nolock(path, empty=False):

    """Function to make directory. If [empty] is True and the directory 
       already exists, it will first be removed.
    """

    # if already exists but needs to be empty, remove it first
    if os.path.isdir(path) and empty:
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)

    return


################################################################################

def clean_tmp(temp_base):

    """ Function that removes the tmp folder corresponding to the
        reduced image / reference image if [set_bb.keep_tmp] not True.
    """

    lock.acquire()
    # change to [run_dir]
    if get_par(set_zogy.make_plots,tel):
        os.chdir(get_par(set_bb.run_dir,tel))
    # and delete [temp_path] if [set_bb.keep_tmp] not True
    temp_path = temp_base.rsplit('/',1)[0]
    if not get_par(set_bb.keep_tmp,tel) and os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    lock.release()

    return


################################################################################

def copy_files2keep (tmp_base, dest_base, ext2keep, move=True, log=None):

    """Function to copy/move files with base name [tmp_base] and
    extensions [ext2keep] to files with base name [dest_base] with the
    same extensions. The base names should include the full path.
    """
    
    # list of all files starting with [tmp_base]
    tmpfiles = glob.glob('{}*'.format(tmp_base))
    # loop this list
    for tmpfile in tmpfiles:
        # determine extension of file 
        tmp_ext = tmpfile.split(tmp_base)[-1]
        # check if the extension is present in [ext2keep]
        for ext in ext2keep:
            if ext == tmpfile[-len(ext):]:
                destfile = '{}{}'.format(dest_base, tmp_ext)
                # if so, and the source and destination names are not
                # identical, go ahead and copy
                if tmpfile != destfile:
                    if not move:
                        if log is not None:
                            log.info('copying {} to {}'.
                                     format(tmpfile, destfile))
                        shutil.copyfile(tmpfile, destfile)
                    else:
                        if log is not None:
                            log.info('moving {} to {}'
                                     .format(tmpfile, destfile))
                        shutil.move(tmpfile, destfile)

                        
    return


################################################################################

def sat_detect (data, header, data_mask, header_mask, tmp_path):

    # could also try skimage.transform.probabilistic_hough_line()
    
    if get_par(set_zogy.timing,tel):
        t = time.time()

    #bin data
    binned_data = data.reshape(np.shape(data)[0] // get_par(set_bb.sat_bin,tel),
                               get_par(set_bb.sat_bin,tel),
                               np.shape(data)[1] // get_par(set_bb.sat_bin,tel),
                               get_par(set_bb.sat_bin,tel)).sum(3).sum(1)
    satellite_fitting = False

    for j in range(3):
        #write binned data to tmp file
        fits_binned_mask = ('{}/{}'.format(
            tmp_path, tmp_path.split('/')[-1].replace('_red','_binned_satmask.fits')))
        fits.writeto(fits_binned_mask, binned_data, overwrite=True)
        #detect satellite trails
        try:
            results, errors = detsat(fits_binned_mask, chips=[0],
                                     n_processes=get_par(set_bb.nthread,tel),
                                     buf=40, sigma=3, h_thresh=0.2, plot=False,
                                     verbose=False)
        except Exception as e:
            log.error('exception was raised during [detsat]: {}'.format(e))
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
                mask_binned = make_mask(fits_binned_mask, 0, trail_segment, sublen=5,
                                        pad=0, sigma=5, subwidth=5000).astype(np.uint8)
            except ValueError:
                #if error occurs, add comment
                log.info ('Warning: satellite trail found but could not be fitted for file {} and is not included in the mask.'
                          .format(tmp_path.split('/')[-1]))
                break
            satellite_fitting = True
            binned_data[mask_binned == 1] = np.median(binned_data)
            fits_old_mask = '{}/old_mask.fits'.format(tmp_path)
            if os.path.isfile(fits_old_mask):
                old_mask = read_hdulist(fits_old_mask)
                mask_binned = old_mask+mask_binned
            fits.writeto(fits_old_mask, mask_binned, overwrite=True)
        else:
            break
    
    if satellite_fitting == True:
        #unbin mask
        mask_sat = np.kron(mask_binned, np.ones((get_par(set_bb.sat_bin,tel),
                                                 get_par(set_bb.sat_bin,tel)))).astype(np.uint8)
        # add pixels affected by satellite trails to [data_mask]
        data_mask[mask_sat==1] += get_par(set_zogy.mask_value['satellite trail'],tel)
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
    
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='sat_detect', log=log)

    return data_mask

        
################################################################################

def cosmics_corr (data, header, data_mask, header_mask):

    if get_par(set_zogy.timing,tel):
        t = time.time()
    
    satlevel_electrons = (get_par(set_bb.satlevel,tel) *
                          np.mean(get_par(set_bb.gain,tel)) - header['BIASMEAN'])
    mask_cr, data = astroscrappy.detect_cosmics(
        data, inmask=(data_mask!=0), sigclip=get_par(set_bb.sigclip,tel),
        sigfrac=get_par(set_bb.sigfrac,tel), objlim=get_par(set_bb.objlim,tel),
        niter=get_par(set_bb.niter,tel), readnoise=header['RDNOISE'],
        satlevel=satlevel_electrons, cleantype='medmask', 
        sepmed=get_par(set_bb.sepmed,tel))

    # from astroscrappy 'manual': To reproduce the most similar
    # behavior to the original LA Cosmic (written in IRAF), set inmask
    # = None, satlevel = np.inf, sepmed=False, cleantype='medmask',
    # and fsmode='median'.
    #mask_cr, data = astroscrappy.detect_cosmics(
    #    data, inmask=None, sigclip=get_par(set_bb.sigclip,tel),
    #    sigfrac=get_par(set_bb.sigfrac,tel), objlim=get_par(set_bb.objlim,tel),
    #    niter=get_par(set_bb.niter,tel),
    #    readnoise=header['RDNOISE'], satlevel=np.inf)
    #
    #print 'np.sum(data_mask!=0)', np.sum(data_mask!=0)
    #print 'np.sum(mask_cr)', np.sum(mask_cr)
    #print 'np.sum((mask_cr) & (data_mask==0))', np.sum((mask_cr) & (data_mask==0))
    
    # add pixels affected by cosmic rays to [data_mask]
    data_mask[mask_cr==1] += get_par(set_zogy.mask_value['cosmic ray'],tel)

    # determining number of cosmics; 2 pixels are considered from the
    # same cosmic also if they are only connected diagonally
    struct = np.ones((3,3), dtype=bool)
    __, ncosmics = ndimage.label(mask_cr, structure=struct)
    ncosmics_persec = ncosmics / header['EXPTIME']
    header['NCOSMICS'] = (ncosmics_persec, '[/s] number of cosmic rays identified')
    # also add this to header of mask image
    header_mask['NCOSMICS'] = (ncosmics_persec, '[/s] number of cosmic rays identified')
    log.info('Number of cosmic rays identified: {}'.format(ncosmics))
    
    
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='cosmics_corr', log=log)

    return data, data_mask


################################################################################

def mask_init (data, header):

    """Function to create initial mask from the bad pixel mask (defining
       the bad and edge pixels), and pixels that are saturated and
       pixels connected to saturated pixels.

    """
    
    if get_par(set_zogy.timing,tel):
        t = time.time()
    
    fits_bpm = get_par(set_bb.bad_pixel_mask,tel)

    bpm_present, fits_bpm = already_exists (fits_bpm, get_filename=True)
    if bpm_present:
        # if it exists, read it
        data_mask = read_hdulist(fits_bpm)
    else:
        # if not, create uint8 array of zeros with same shape as
        # [data]
        log.info('Warning: bad pixel mask {} does not exist'.format(fits_bpm))
        data_mask = np.zeros(np.shape(data), dtype='uint8')

    # mask of pixels with non-finite values in [data]
    mask_infnan = ~np.isfinite(data)
    # replace those pixel values with zeros
    data[mask_infnan] = 0
    # and add them to [data_mask] with same value defined for 'bad' pixels
    # unless that pixel was already masked
    data_mask[(mask_infnan) & (data_mask==0)] += get_par(
        set_zogy.mask_value['bad'],tel)
    
    # identify saturated pixels; saturation level (ADU) is taken from
    # blackbox settings file, which needs to be mulitplied by the gain
    # and have the mean biaslevel subtracted
    satlevel_electrons = (get_par(set_bb.satlevel,tel) *
                          np.mean(get_par(set_bb.gain,tel)) - header['BIASMEAN'])
    mask_sat = (data >= satlevel_electrons)
    # add them to the mask of edge and bad pixels
    data_mask[mask_sat] += get_par(set_zogy.mask_value['saturated'],tel)

    # determining number of saturated objects; 2 saturated pixels are
    # considered from the same object also if they are only connected
    # diagonally
    struct = np.ones((3,3), dtype=bool)
    __, nobj_sat = ndimage.label(mask_sat, structure=struct)
    
    # and pixels connected to saturated pixels
    struct = np.ones((3,3), dtype=bool)
    mask_satconnect = ndimage.binary_dilation(mask_sat, structure=struct,
                                              iterations=2)
    # add them to the mask
    data_mask[(mask_satconnect) & (~mask_sat)] += get_par(
        set_zogy.mask_value['saturated-connected'],tel)

    # create initial mask header 
    header_mask = fits.Header()
    header_mask['SATURATE'] = (satlevel_electrons, '[e-] adopted saturation '
                               'threshold')
    header['NOBJ-SAT'] = (nobj_sat, 'number of saturated objects')
    # also add these to the header of image itself
    header['SATURATE'] = (satlevel_electrons, '[e-] adopted saturation threshold')
    header['NOBJ-SAT'] = (nobj_sat, 'number of saturated objects')
    # rest of the mask header entries are added in one go using
    # function [mask_header] once all the reduction steps have
    # finished
    
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='mask_init', log=log)

    return data_mask.astype('uint8'), header_mask


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

def master_prep (data_shape, path, date_eve, imtype, filt=None, log=None):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    if imtype=='flat':
        fits_master = '{}/{}_{}_{}.fits'.format(path, imtype, date_eve, filt)
    elif imtype=='bias':
        fits_master = '{}/{}_{}.fits'.format(path, imtype, date_eve)

    # check if already present (if fpacked, fits_master below will
    # point to fpacked file)
    master_present, fits_master = already_exists (fits_master, get_filename=True)

    # check if master bias/flat does not contain any red flags:
    master_ok = True
    if master_present:
        header_master = read_hdulist (fits_master, get_data=False, get_header=True)
        qc_flag = run_qc_check (header_master, tel, log=log)
        if qc_flag=='red':
            master_ok = False

    if not (master_present and master_ok):
        # prepare master image from files in [path] +/- the specified
        # time window
        if imtype=='flat':
            nwindow = int(get_par(set_bb.flat_window,tel))
        elif imtype=='bias':
            nwindow = int(get_par(set_bb.bias_window,tel))

        file_list = []
        red_dir = get_par(set_bb.red_dir,tel)
        for n_day in range(-nwindow, nwindow+1):
            # determine mjd at noon (local or UTC, does not matter) of date_eve +- n_day
            mjd_noon = date2mjd('{}'.format(date_eve), time_str='12:00') + n_day
            # corresponding path
            date_temp = Time(mjd_noon, format='mjd').isot.split('T')[0].replace('-','/')
            path_temp = '{}/{}/{}'.format(red_dir, date_temp, imtype)
            if imtype=='flat':
                file_list.append(sorted(
                    glob.glob('{}/{}_*_{}.fits*'.format(path_temp, tel, filt))))
            else:
                file_list.append(sorted(
                    glob.glob('{}/{}_*.fits*'.format(path_temp, tel))))

        # clean up [file_list]
        file_list = [f for sublist in file_list for f in sublist]

        # do not consider image with header QC-FLAG set to red
        mask_keep = np.ones(len(file_list), dtype=bool)
        for i_file, filename in enumerate(file_list):
            header_temp = read_hdulist (filename, get_data=False, get_header=True)
            if 'QC-FLAG' in header_temp:
                if header_temp['QC-FLAG'] == 'red':
                    mask_keep[i_file] = False
        file_list = np.array(file_list)[mask_keep]

        # initialize cube of images to be combined
        nfiles = len(file_list)

        # if master bias/flat contains a red flag, look for a nearby
        # master flat instead
        if nfiles < 3 or not master_ok:
            fits_master_close = get_closest_biasflat(date_eve, imtype, filt=filt)

            if fits_master_close is not None:

                # if master bias subtraction switch is off, the master
                # bias is still prepared; only show message below in
                # case switch is on, otherwise it is confusing
                if ((imtype=='bias' and get_par(set_bb.subtract_mbias,tel))
                    or imtype=='flat'):
                    if log is not None:
                        log.info ('Warning: too few images to produce master {} '
                                  'for evening date {} +/- window of {} days\n'
                                  'instead using: {}'
                                  .format(imtype, date_eve, nwindow,
                                          fits_master_close))
                # previously we created a symbolic link so future
                # files would automatically use this as the master
                # file, but as this symbolic link is confusing, let's
                # not do that; searching for nearby master frame takes
                # a negligible amount of time
                # os.symlink(fits_master_close, fits_master)
                fits_master = fits_master_close
                
            else:
                if ((imtype=='bias' and get_par(set_bb.subtract_mbias,tel))
                    or imtype=='flat'):
                    if log is not None:
                        log.error('no alternative master {} found'
                                  .format(imtype))
                return None
                
        else:
            if imtype=='bias':
                q.put(logger.info ('making master {} for night {}'
                                   .format(imtype, date_eve)))
                if not get_par(set_bb.subtract_mbias,tel):
                    q.put(logger.info ('(but will not be applied to input image '
                                       'as [subtract_mbias] is set to False)'))
            if imtype=='flat':
                q.put(logger.info ('making master {} in filter {} for night {}'
                                   .format(imtype, filt, date_eve)))

            # assuming that individual flats/biases have the same
            # shape as the input data
            ysize, xsize = data_shape
            master_cube = np.zeros((nfiles, ysize, xsize), dtype='float32')

            # initialize master header
            header_master = fits.Header()        

            # fill the cube
            ra_flats = []
            dec_flats = []
            for i_file, filename in enumerate(file_list):

                master_cube[i_file], header_temp = read_hdulist(filename,
                                                                get_header=True)

                if imtype=='flat':
                    # divide by median over the region [set_bb.flat_norm_sec]
                    index_flat_norm = get_par(set_bb.flat_norm_sec,tel)
                    __, median, std = sigma_clipped_stats(
                        master_cube[i_file][index_flat_norm].astype('float64'),
                        mask_value=0)
                    if log is not None:
                        log.info ('flat name: {}, median: {}, std: {}'
                                  .format(filename, median, std))
                    master_cube[i_file] /= median

                    # collect RA and DEC to check for dithering
                    if 'RA' in header_temp and 'DEC' in header_temp:
                        ra_flats.append(header_temp['RA'])
                        dec_flats.append(header_temp['DEC'])

                # copy some header keyword values from first file
                if i_file==0:
                    for key in ['IMAGETYP', 'DATE-OBS', 'FILTER', 'RA', 'DEC',
                                'XBINNING', 'YBINNING', 'MJD-OBS', 'AIRMASS', 
                                'ORIGIN', 'TELESCOP', 'PYTHON-V', 'BB-V']:
                        if key in header_temp:
                            header_master[key] = (header_temp[key],
                                                  header_temp.comments[key])


                if imtype=='flat':
                    comment = 'name reduced flat'
                elif imtype=='bias':
                    comment = 'name gain/os-corrected bias frame'

                header_master['{}{}'.format(imtype.upper(), i_file+1)] = (
                    filename.split('/')[-1].split('.fits')[0],
                    '{} {}'.format(comment, i_file+1))
                
                if 'ORIGFILE' in header_temp.keys():
                    header_master['{}OR{}'.format(imtype.upper(), i_file+1)] = (
                        header_temp['ORIGFILE'], 'name original {} {}'
                        .format(imtype, i_file+1))

                # also copy a few header keyword values from the last file
                if i_file==nfiles-1:
                    for key in ['DATE-END', 'MJD-END']:
                        if key in header_temp:
                            header_master[key] = (header_temp[key],
                                                  header_temp.comments[key])
                    
            # determine the median
            master_median = np.median(master_cube, axis=0)

            # add number of files combined
            header_master['N{}'.format(imtype.upper())] = (
                nfiles, 'number of {} frames combined'.format(imtype.lower()))

            # add time window used
            header_master['{}-WIN'.format(imtype.upper())] = (
                nwindow, '[days] input time window to include {} frames'
                .format(imtype.lower()))

            # add some header keywords to the master flat
            if imtype=='flat':
                sec_temp = get_par(set_bb.flat_norm_sec,tel)
                value_temp = '[{}:{},{}:{}]'.format(
                    sec_temp[0].start+1, sec_temp[0].stop+1, 
                    sec_temp[1].start+1, sec_temp[1].stop+1) 
                header_master['STATSEC'] = (
                    value_temp, 'pre-defined statistics section [y1:y2,x1:x2]')

                header_master['MFMEDSEC'] = (
                    np.median(master_median[sec_temp]), 
                    'median master flat over STATSEC')
                
                header_master['MFSTDSEC'] = (
                    np.std(master_median[sec_temp]),
                    'sigma (STD) master flat over STATSEC')

                # "full" image statistics
                index_stat = get_rand_indices(master_median.shape)
                __, median_master, std_master = sigma_clipped_stats(
                    master_median[index_stat].astype('float64'), mask_value=0)
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
                                                np.roll(ra_flats,1), np.roll(dec_flats,1))
                    # count how many were offset by at least 5"
                    mask_off = (offset >= 5)
                    noffset = np.sum(mask_off)
                    if noffset > 0:
                        offset_mean = np.mean(offset[mask_off])

                        
                header_master['N-OFFSET'] = (noffset, 
                                             'number of flats with offsets > 5 arcsec')
                header_master['OFF-MEAN'] = (offset_mean, 
                                             '[arcsec] mean dithering offset')
                if float(noffset)/nfiles >= 0.66:
                    flat_dithered = True
                else:
                    flat_dithered = False
                header_master['FLATDITH'] = (flat_dithered, 'majority of flats were dithered')

                # set edge and non-positive pixels to 1; edge pixels
                # are idenfitied by reading in bad pixel mask as
                # master preparation is not necessariliy linked to the
                # mask of an object image, e.g. in function
                # [masters_left]
                fits_bpm = get_par(set_bb.bad_pixel_mask,tel)
                bpm_present, fits_bpm = already_exists (fits_bpm, get_filename=True)
                if bpm_present:
                    # if mask exists, read it
                    data_mask = read_hdulist(fits_bpm)
                    mask_replace = ((data_mask==get_par(set_zogy.mask_value['edge'],tel)) |
                                    (master_median<=0))
                    master_median[mask_replace] = 1
                    
                # now that master flat is produced, normalize the
                # different channels such that the final image
                # appears smooth without any jumps in levels 
                # between the different channels

                __, __, __, __, data_sec_red = define_sections(data_shape, tel=tel)
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
                    # statistics of right side of previous channel pair
                    data_stat2 = master_median_corr[y_index-nrows:y_index+nrows,
                                                    x_index:x_index+ncols]
                    ratio = np.median(data_stat1)/np.median(data_stat2)
                    # correct relevant channels
                    master_median_corr[data_sec_red[i]] *= ratio
                    master_median_corr[data_sec_red[i+nx]] *= ratio
                    # update correction factor
                    factor_chan[i] *= ratio
                    factor_chan[i+nx] *= ratio


                # normalise corrected master to [flat_norm_sec] section
                sec_temp = get_par(set_bb.flat_norm_sec,tel)
                ratio_norm = np.median(master_median_corr[sec_temp])
                master_median_corr /= ratio_norm
                factor_chan /= ratio_norm

                # add factor_chan values to header
                for i_chan in range(nchans):
                    header_master['GAINCF{}'.format(i_chan+1)] = (
                        factor_chan[i_chan], 'channel {} gain correction factor'
                        .format(i_chan+1))

                    
            elif imtype=='bias':

                # add some header keywords to the master bias
                index_stat = get_rand_indices(master_median.shape)
                mean_master, __, std_master = sigma_clipped_stats(
                    master_median[index_stat].astype('float64'), mask_value=0)
                header_master['MBMEAN'] = (mean_master, '[e-] mean master bias')
                header_master['MBRDN'] = (std_master, '[e-] sigma (STD) master bias')

                # including the means and standard deviations of the master
                # bias in the separate channels
                __, __, __, __, data_sec_red = define_sections(data_shape, tel=tel)
                nchans = np.shape(data_sec_red)[0]
                mean_chan = np.zeros(nchans)
                std_chan = np.zeros(nchans)

                for i_chan in range(nchans):
                    data_chan = master_median[data_sec_red[i_chan]]
                    index_stat = get_rand_indices(data_chan.shape)
                    mean_chan[i_chan], __, std_chan[i_chan] = sigma_clipped_stats(
                        data_chan[index_stat].astype('float64'), mask_value=0)
                for i_chan in range(nchans):
                    header_master['MBIASM{}'.format(i_chan+1)] = (
                        mean_chan[i_chan], '[e-] channel {} mean master bias'.format(i_chan+1))
                for i_chan in range(nchans):
                    header_master['MBRDN{}'.format(i_chan+1)] = (
                        std_chan[i_chan], '[e-] channel {} sigma (STD) master bias'.format(i_chan+1))

            # call [run_qc_check] to update master header with any QC flags
            run_qc_check (header_master, tel, log=log)
            # make dir for output file if it doesn't exist yet
            make_dir_nolock (os.path.split(fits_master)[0])
            # write to output file
            header_master['DATEFILE'] = (Time.now().isot, 'UTC date of writing file')
            fits.writeto(fits_master, master_median.astype('float32'), header_master,
                         overwrite=True)

    if get_par(set_zogy.timing,tel):
        if log is not None:
            log_timing_memory (t0=t, label='master_prep', log=log)

    return fits_master


################################################################################

def get_closest_biasflat (date_eve, file_type, filt=None):

    red_dir = get_par(set_bb.red_dir,tel)
    search_str = '{}/*/*/*/{}/{}_????????'.format(red_dir, file_type, file_type)
    if filt is None:
        search_str = '{}.fits*'.format(search_str)
    else:
        search_str = '{}_{}.fits*'.format(search_str, filt)

    files = glob.glob(search_str)
    nfiles = len(files)

    if nfiles > 0:
        # find file that is closest in time to [date_eve]
        mjds = np.array([date2mjd(files[i].split('/')[-1].split('_')[1])
                         for i in range(nfiles)])
        # these mjds corresponding to the very start of the day
        # (midnight) but in the comparison this offset cancels out
        i_close = np.argmin(abs(mjds - date2mjd(date_eve)))
        return files[i_close]

    else:
        return None
    

################################################################################

def date2mjd (date_str, time_str=None, get_jd=False):
    
    """convert [date_str] and [time_str] to MJD or JD with possible
    formats: yyyymmdd or yyyy-mm-dd for [date_str] and hh:mm[:ss.s]
    for [time_str]

    """
    
    if '-' not in date_str:
        date_str = '{}-{}-{}'.format(date_str[0:4], date_str[4:6], date_str[6:8])

    if time_str is not None:
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
            q.put(logger.error('crucial keyword {} not present in header; '
                               'not processing {}'.format(key, filename)))
            header_ok = False
            # return immediately in this case so that presence of 
            # IMAGETYP does not need to be checked below
            return header_ok


    # define imgtype
    imgtype = header['IMAGETYP'].lower()

    
    # for early ML data, header keyword FIELD_ID rather than OBJECT
    # was used for the field identification 
    if 'FIELD_ID' in header:
        obj = header['FIELD_ID']
    elif 'OBJECT' in header:
        obj = header['OBJECT']
    else:
        if imgtype=='object':
            # if neither FIELD_ID nor OBJECT present in header of an
            # object image, then also bail out
            q.put(logger.error('FIELD_ID and OBJECT keywords not present in '
                               'header; not processing {}'.format(filename)))
            header_ok = False
            # return right away as otherwise [obj] not defined, which
            # is used below
            return header_ok
            

    # check if OBJECT keyword value contains digits only
    if imgtype=='object':
        try:
            int(obj)
        except Exception as e:
            q.put(logger.error('keyword OBJECT (or FIELD_ID if present) does '
                               'not contain digits only; not processing {}'
                               .format(filename)))
            header_ok = False
            return header_ok
                  
    
    # remaining important keywords; for biases, darks and flats, these
    # keywords are not strictly necessary (although for flats they are
    # used to check if they were dithered; if RA and DEC not present,
    # any potential dithering will not be detected)
    for keys in ['EXPTIME', 'RA', 'DEC']:
        if imgtype=='object':
            if key not in header:
                q.put(logger.error('crucial keyword {} not present in header; '
                                   'not processing {}'.format(key, filename)))
                header_ok = False
                return header_ok
            
            
    return header_ok


################################################################################

def check_header2 (header, filename):
    
    header_ok = True

    # check if the field ID and RA-REF, DEC-REF combination is
    # consistent with definition of ML/BG field IDs; threshold used:
    # 10 arc minutes
    mlbg_fieldIDs = get_par(set_bb.mlbg_fieldIDs,tel)
    table_ID = ascii.read(mlbg_fieldIDs, names=['ID', 'RA', 'DEC'], data_start=0)
    imgtype = header['IMAGETYP'].lower()
    if imgtype=='object':
        obj = header['OBJECT']
        if int(obj)!=0 and int(obj)<=20000:
            
            if 'RA-REF' in header and 'DEC-REF' in header:
                ra_deg = Angle(header['RA-REF'], unit=u.hour).degree
                dec_deg = Angle(header['DEC-REF'], unit=u.deg).degree
            else:
                # use RA and DEC instead
                ra_deg = Angle(header['RA'], unit=u.hour).degree
                dec_deg = Angle(header['DEC'], unit=u.deg).degree

            # check if there is a match with the defined field IDs
            mask_match = (table_ID['ID']==int(obj))
            if sum(mask_match) == 0:
                # observed field is not present in definition of field IDs
                header_ok = False
                q.put(logger.error('input header field ID not present in '
                                   'definition of field IDs:\n{}\n'
                                   'header field ID: {}, RA-REF: {}, DEC-REF: {}\n'
                                   'not processing {}'
                                   .format(mlbg_fieldIDs, obj, ra_deg, dec_deg, filename)))
            else:
                i_ID = np.nonzero(mask_match)[0][0]
                if haversine(table_ID['RA'][i_ID], table_ID['DEC'][i_ID], 
                             ra_deg, dec_deg) > 10./60:
                    q.put(logger.error('input header field ID, RA and DEC combination '
                                       'is inconsistent (>10\') with definition of field IDs\n'
                                       'header field ID: {}, RA-REF: {}, DEC-REF: {}\n'
                                       'vs.    field ID: {}, RA:     {}, DEC:     {} '
                                       'in {}\n'
                                       'not processing {}'
                                       .format(obj, ra_deg, dec_deg, 
                                               table_ID['ID'][i_ID], 
                                               table_ID['RA'][i_ID],
                                               table_ID['DEC'][i_ID], 
                                               mlbg_fieldIDs, filename)))
                    header_ok = False


    # if binning is not 1x1, also skip processing
    if 'XBINNING' in header and 'YBINNING' in header: 
        if header['XBINNING'] != 1 or header['YBINNING'] != 1:
            q.put(logger.error('BINNING not 1x1; not processing {}'
                               .format(filename)))
            header_ok = False


    return header_ok


################################################################################

def set_header(header, filename):

    def edit_head (header, key, value=None, comments=None, dtype=None):
        # update value
        if value is not None:
            if key in header:
                if header[key] != value and value != 'None':
                    print ('warning: value of existing keyword {} updated from {} to {}'
                           .format(key, header[key], value))
                    header[key] = value
            else:
                header[key] = value                    
        # update comments
        if comments is not None:
            if key in header:
                header.comments[key] = comments
            else:
                print ('warning: keyword {} does not exist: comment is not updated'
                       .format(key))
        # update dtype
        if dtype is not None:
            if key in header and header[key] != 'None':
                header[key] = dtype(header[key])
            else:
                print ('warning: dtype of keyword {} is not updated'.format(key))
                
                
    edit_head(header, 'NAXIS', comments='number of array dimensions')
    edit_head(header, 'NAXIS1', comments='length of array axis')
    edit_head(header, 'NAXIS2', comments='length of array axis')

    edit_head(header, 'BUNIT', value='e-', comments='Physical unit of array values')
    edit_head(header, 'BSCALE', comments='value = fits_value*BSCALE+BZERO')
    edit_head(header, 'BZERO', comments='value = fits_value*BSCALE+BZERO')
    #edit_head(header, 'CCD-AMP', value='', comments='Amplifier mode of the CCD camera')
    #edit_head(header, 'CCD-SET', value='', comments='CCD settings file')
   
    edit_head(header, 'CCD-TEMP', value='None', comments='[C] Current CCD temperature')
        
    if 'XBINNING' in header:
        edit_head(header, 'XBINNING', comments='[pix] Binning factor X axis')
    else:
        xsize = header['NAXIS1']
        nx = get_par(set_bb.nx,tel)
        dx = get_par(set_bb.xsize_chan,tel)
        xbinning = int(np.ceil(float(nx*dx)/xsize))
        edit_head(header, 'XBINNING', value=xbinning, comments='[pix] Binning factor X axis')
        
    if 'YBINNING' in header:
        edit_head(header, 'YBINNING', comments='[pix] Binning factor Y axis')
    else:
        ysize = header['NAXIS2']
        ny = get_par(set_bb.ny,tel)
        dy = get_par(set_bb.ysize_chan,tel)
        ybinning = int(np.ceil(float(ny*dy)/ysize))
        edit_head(header, 'YBINNING', value=ybinning, comments='[pix] Binning factor Y axis')


    edit_head(header, 'ALTITUDE', comments='[deg] Altitude in horizontal coordinates')
    edit_head(header, 'AZIMUTH', comments='[deg] Azimuth in horizontal coordinates')
    edit_head(header, 'RADESYS', value='ICRS', comments='Coordinate reference frame')
    edit_head(header, 'EPOCH', value=2015.5, comments='Coordinate reference epoch')
    
    # RA
    if 'RA' in header:
        if ':' in str(header['RA']):
            # convert sexagesimal to decimal degrees
            ra_deg = Angle(header['RA'], unit=u.hour).degree
        else:
            # convert RA decimal hours to degrees
            ra_deg = header['RA'] * 15.
        edit_head(header, 'RA', value=ra_deg, comments='[deg] Telescope right ascension')

    # DEC
    if 'DEC' in header:
        if ':' in str(header['DEC']):
            # convert sexagesimal to decimal degrees
            dec_deg = Angle(header['DEC'], unit=u.deg).degree
        else:
            # for airmass determination below
            dec_deg = header['DEC']
        edit_head(header, 'DEC', value=dec_deg, comments='[deg] Telescope declination')

    edit_head(header, 'FLIPSTAT', value='None', comments='Telescope side of the pier')
    edit_head(header, 'EXPTIME', comments='[s] Requested exposure time')

    if 'ISTRACKI' in header:
        # convert the string value to boolean
        value = (header['ISTRACKI']=='True')
        edit_head(header, 'ISTRACKI', value=value, comments='Telescope is tracking')


    # record original DATE-OBS and END-OBS in ACQSTART and ACQEND
    edit_head(header, 'ACQSTART', value=header['DATE-OBS'], comments='start of acquisition (server timing)')
    if 'END-OBS' in header:
        edit_head(header, 'ACQEND', value=header['END-OBS'], comments='end of acquisition (server timing)')
    else:
        edit_head(header, 'ACQEND', value='None', comments='end of acquisition (server timing)')
        

    # midexposure DATE-OBS is based on GPSSTART and GPSEND; if these
    # keywords are not present in the header, or if the image is a
    # bias or dark frame (which both should not contain these
    # keywords, and if they do, the keyword values are actually
    # identical to those of the image preceding the bias/dark), then
    # just adopt the original DATE-OBS (=ACQSTART) as the date of
    # observation
    imgtype = header['IMAGETYP'].lower()
    if 'GPSSTART' in header and 'GPSEND' in header and imgtype == 'object':
        
        # replace DATE-OBS with (GPSSTART+GPSEND)/2
        gps_mjd = Time([header['GPSSTART'], header['GPSEND']], format='isot').mjd
        mjd_obs = np.sum(gps_mjd)/2.
        date_obs_str = Time(mjd_obs, format='mjd').isot
        edit_head(header, 'DATE-OBS', value=date_obs_str, comments='Midexp. date @img cntr:(GPSSTART+GPSEND)/2')
        date_obs = Time(date_obs_str, format='isot') # change from a string to time class

        # also add keyword to check (GPSEND-GPSSTART) - EXPTIME
        gps_shut = (gps_mjd[1]-gps_mjd[0])*24*3600. - header['EXPTIME']
        edit_head(header, 'GPS-SHUT', value=gps_shut, comments='[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME')
        
    else:
        date_obs_str = header['DATE-OBS']
        date_obs = Time(date_obs_str, format='isot')
        # DATE-OBS already present; just edit the comments
        edit_head(header, 'DATE-OBS', comments='Date at start (=ACQSTART)')
        mjd_obs = Time(date_obs, format='isot').mjd
        
        
    if 'GPSSTART' not in header:
        edit_head(header, 'GPSSTART', value='None', comments='GPS timing start of opening shutter')
    if 'GPSEND' not in header:
        edit_head(header, 'GPSEND', value='None', comments='GPS timing end of opening shutter')
    if 'GPS-SHUT' not in header and imgtype == 'object':
        edit_head(header, 'GPS-SHUT', value='None', comments='[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME')

    edit_head(header, 'MJD-OBS', value=mjd_obs, comments='[d] MJD (based on DATE-OBS)')
    
    # in degrees:
    lon_temp = get_par(set_zogy.obs_lon,tel)
    lst = date_obs.sidereal_time('apparent', longitude=lon_temp)
    lst_deg = lst.deg
    # in hh:mm:ss.ssss
    lst_str = lst.to_string(sep=':', precision=3)
    edit_head(header, 'LST', value=lst_str, comments='apparent LST (based on DATE-OBS)')
        
    utc = (mjd_obs-np.floor(mjd_obs)) * 3600. * 24
    edit_head(header, 'UTC', value=utc, comments='[s] UTC (based on DATE-OBS)')
    edit_head(header, 'TIMESYS', value='UTC', comments='Time system used')
    

    # telescope latitude, longitude and height (used for AIRMASS and
    # SITELONG, SITELAT and ELEVATIO)
    lat = get_par(set_zogy.obs_lat,tel)
    lon = get_par(set_zogy.obs_lon,tel)
    height = get_par(set_zogy.obs_height,tel)
    
    if 'RA' in header and 'DEC' in header:
        # for ML1, the RA and DEC were incorrectly referring to the
        # subsequent image up to 9 Feb 2019 (except when put in by
        # hand with the sexagesimal notation, in which case keywords
        # RA-TEL and DEC-TEL are not present in the header); for these
        # images we replace the RA and DEC by the RA-REF and DEC-REF
        if tel=='ML1':
            tcorr_radec = Time('2019-02-09T00:00:00', format='isot').mjd
            if mjd_obs < tcorr_radec and 'RA-REF' in header and 'DEC-REF' in header:
                ra_deg = Angle(header['RA-REF'], unit=u.hour).degree
                dec_deg = Angle(header['DEC-REF'], unit=u.deg).degree
                edit_head(header, 'RA', value=ra_deg,
                          comments='[deg] Telescope right ascension (=RA-REF)')
                edit_head(header, 'DEC', value=dec_deg,
                          comments='[deg] Telescope declination (=DEC-REF)')

        # determine airmass
        airmass = get_airmass(ra_deg, dec_deg, date_obs_str, lat, lon, height)
        edit_head(header, 'AIRMASS', value=float(airmass), 
                  comments='Airmass (based on RA, DEC, DATE-OBS)')

        
    edit_head(header, 'SITELAT',  value=lat, comments='[deg] Site latitude')
    edit_head(header, 'SITELONG', value=lon, comments='[deg] Site longitude')
    edit_head(header, 'ELEVATIO', value=height, comments='[m] Site elevation')
   
    # update -REF and -TEL of RAs and DECs; if the -REFs do not exist
    # yet, create them with 'None' values - needed for the Database
    edit_head(header, 'RA-REF', value='None', comments='Requested right ascension')
    edit_head(header, 'RA-TEL', comments='[deg] Telescope right ascension')
    edit_head(header, 'DEC-REF', value='None', comments='Requested declination')
    edit_head(header, 'DEC-TEL', comments='[deg] Telescope declination')
    
    # now that RA/DEC are (potentially) corrected, determine local
    # hour angle this keyword was in the raw image header for a while,
    # but seems to have disappeared during the 2nd half of March 2019
    if 'RA' in header:
        lha_deg = lst_deg - ra_deg
        edit_head(header, 'HA', value=lha_deg, comments='[deg] Local hour angle (=LST-RA)')

    
    # Weather headers required for Database
    edit_head(header, 'CL-BASE',  value='None', dtype=float, comments='[m] Reinhardt cloud base altitude')
    edit_head(header, 'RH-MAST',  value='None', dtype=float, comments='Vaisala RH mast')
    edit_head(header, 'RH-DOME',  value='None', dtype=float, comments='CilSense2 RH dome')
    edit_head(header, 'RH-AIRCO', value='None', dtype=float, comments='CilSense3 RH server room airco')
    edit_head(header, 'RH-PIER',  value='None', dtype=float, comments='CilSense1 RH pier')
    edit_head(header, 'PRESSURE', value='None', dtype=float, comments='[hPa] Vaisala pressure mast')
    edit_head(header, 'T-PIER',   value='None', dtype=float, comments='[C] CilSense1 temperature pier')
    edit_head(header, 'T-DOME',   value='None', dtype=float, comments='CilSense2 temperature dome')
    edit_head(header, 'T-ROOF',   value='None', dtype=float, comments='[C] Reinhardt temperature roof')
    edit_head(header, 'T-AIRCO',  value='None', dtype=float, comments='[C] CilSense3 temperature server room airco')
    edit_head(header, 'T-MAST',   value='None', dtype=float, comments='[C] Vaisala temperature mast')
    edit_head(header, 'T-STRUT',  value='None', dtype=float, comments='[C] Temperature carbon strut between M1 and M2')
    edit_head(header, 'T-CRING',  value='None', dtype=float, comments='[C] Temperature main carbon ring around M1')
    edit_head(header, 'T-SPIDER', value='None', dtype=float, comments='[C] Temperature carbon spider above M2')
    edit_head(header, 'T-FWN',    value='None', dtype=float, comments='[C] Temperature filter wheel housing North')
    edit_head(header, 'T-FWS',    value='None', dtype=float, comments='[C] Temperature filter wheel housing South')
    edit_head(header, 'T-M2HOLD', value='None', dtype=float, comments='[C] Temperature aluminium M2 holder')
    edit_head(header, 'T-GUICAM', value='None', dtype=float, comments='[C] Temperature guide camera')
    edit_head(header, 'T-M1',     value='None', dtype=float, comments='[C] Temperature backside M1')
    edit_head(header, 'T-CRYWIN', value='None', dtype=float, comments='[C] Temperature Cryostat window')
    edit_head(header, 'T-CRYGET', value='None', dtype=float, comments='[C] Temperature Cryostat getter')
    edit_head(header, 'T-CRYCP',  value='None', dtype=float, comments='[C] Temperature Cryostat cold plate')
    edit_head(header, 'PRES-CRY', value='None', dtype=float, comments='[bar] Cryostat vacuum pressure')
    edit_head(header, 'WINDAVE',  value='None', dtype=float, comments='[km/h] Vaisala wind speed mast')
    edit_head(header, 'WINDGUST', value='None', dtype=float, comments='[km/h] Vaisala wind gust mast')
    edit_head(header, 'WINDDIR',  value='None', dtype=float, comments='[deg] Vaisala wind direction mast')
    
    
    edit_head(header, 'FILTER', comments='Filter')
    if tel=='ML1':
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

        transition_mjd = Time(['2017-11-19T00:00:00', '2018-02-24T23:59:59'], format='isot').mjd
        tcorr_mjd = Time(['2018-02-25T00:00:00', '2019-01-13T15:00:00'], format='isot').mjd
        if mjd_obs >= transition_mjd[0] and mjd_obs <= transition_mjd[1]:
            if 'OBSERVER' in header and header['OBSERVER'].lower()=='abot':
                filt_old = header['FILTER']
                edit_head(header, 'FILTER', value=filt_corr[filt_old], comments='Filter (corrected)')
        elif mjd_obs >= tcorr_mjd[0] and mjd_obs <= tcorr_mjd[1]:
            filt_old = header['FILTER']
            edit_head(header, 'FILTER', value=filt_corr[filt_old], comments='Filter (corrected)')


    edit_head(header, 'CCD-ID',   value='None', dtype=str, comments='CCD camera ID')
    edit_head(header, 'CONTROLL', value='None', dtype=str, comments='CCD controller')
    edit_head(header, 'DETSPEED', value='None', dtype=int, comments='[kHz] Detector read speed')
    edit_head(header, 'CCD-NW',   dtype=int, comments='Number of channels in width')
    edit_head(header, 'CCD-NH',   dtype=int, comments='Number of channels in height')
    edit_head(header, 'INSTRUME', value='None', dtype=str, comments='Instrument name')
    edit_head(header, 'FOCUSPOS', value='None', dtype=int, comments='[micron] Focuser position')

    edit_head(header, 'IMAGETYP', dtype=str, comments='Image type')
    edit_head(header, 'OBJECT',   dtype=str, comments='Name of object observed (field ID)')

    if header['IMAGETYP'].lower()=='object':
        if 'FIELD_ID' in header:
            obj = header['FIELD_ID']
        else:
            obj = header['OBJECT']
        edit_head(header, 'OBJECT', value='{:0>5}'.format(obj), comments='Name of object observed (field ID)')

    if tel=='ML1':
        origin = 'SAAO-Sutherland (K94)'
        telescop = 'MeerLICHT-{}'.format(tel[2:])
    if tel[0:2]=='BG':
        origin = 'ESO-LaSilla (809)'
        telescop = 'BlackGEM-{}'.format(tel[2:])
    edit_head(header, 'ORIGIN', value=origin, comments='Origin of data (MPEC Observatory code)')
    edit_head(header, 'TELESCOP', value=telescop, comments='Telescope ID')
    
    # do not add ARCFILE name for the moment
    #arcfile = '{}.{}'.format(tel, date_obs_str)
    #edit_head(header, 'ARCFILE', value=arcfile, comments='Archive filename')
    edit_head(header, 'ORIGFILE', value=filename.split('/')[-1].split('.fits')[0], comments='ABOT filename')

    
    edit_head(header, 'OBSERVER', value='None', dtype=str, comments='Robotic observations software and PC ID')
    edit_head(header, 'ABOTVER',  value='None', dtype=str, comments='ABOT version')
    edit_head(header, 'PROGNAME', value='None', dtype=str, comments='Program name')
    edit_head(header, 'PROGID',   value='None', dtype=str, comments='Program ID')
    edit_head(header, 'GUIDERST', value='None', dtype=str, comments='Guider status')
    edit_head(header, 'GUIDERFQ', value='None', dtype=float, comments='[Hz] Guide loop frequency')
    edit_head(header, 'TRAKTIME', value='None', dtype=float, comments='[s] Autoguider exposure time during imaging')
    edit_head(header, 'ADCX',     value='None', dtype=float, comments='[mm] Position offset ADC lens in x')
    edit_head(header, 'ADCY',     value='None', dtype=float, comments='[mm] Position offset ADC lens in y')
    
    
    # remove the following keywords:
    keys_2remove = ['FILTWHID', 'FOC-ID', 'EXPOSURE', 'END-OBS', 'FOCUSMIT', 
                    'FOCUSAMT', 'OWNERGNM', 'OWNERGID', 'OWNERID',
                    'AZ-REF', 'ALT-REF', 'CCDFULLH', 'CCDFULLW', 'RADECSYS']
    for key in keys_2remove:
        if key in header:
            print ('removing keyword {}'.format(key))
            header.remove(key, remove_all=True)
    
            
    # put some order in the header
    keys_sort = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                 'BUNIT', 'BSCALE', 'BZERO',
                 'XBINNING', 'YBINNING',
                 'ALTITUDE', 'AZIMUTH', 'RADESYS', 'EPOCH',
                 'RA', 'RA-REF', 'RA-TEL', 'DEC', 'DEC-REF', 'DEC-TEL',
                 'HA', 'FLIPSTAT', 'ISTRACKI',
                 'OBJECT', 'IMAGETYP', 'FILTER', 'EXPTIME',
                 'ACQSTART', 'ACQEND', 'GPSSTART', 'GPSEND', 'GPS-SHUT',
                 'DATE-OBS', 'MJD-OBS', 'LST', 'UTC', 'TIMESYS',
                 'SITELAT', 'SITELONG', 'ELEVATIO', 'AIRMASS',
                 'SET-TEMP', 'CCD-TEMP', 'CCD-ID', 'CONTROLL', 'DETSPEED', 
                 'CCD-NW', 'CCD-NH', 'FOCUSPOS',
                 'ORIGIN', 'TELESCOP', 'INSTRUME', 
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
            print ('keyword {} not in header'.format(key))            

    return header_sort


################################################################################

def define_sections (data_shape, tel=None):

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

    ysize_chan = get_par(set_bb.ysize_chan,tel)
    xsize_chan = get_par(set_bb.xsize_chan,tel)
    ysize_os = (ysize-ny*ysize_chan) // ny
    xsize_os = (xsize-nx*xsize_chan) // nx

    # the sections below are defined such that e.g. chan_sec[0] refers
    # to all pixels of the first channel, where the channel indices
    # are currently defined to be located on the CCD as follows:
    #
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]

    # channel section slices including overscan; shape=(16,2)
    chan_sec = tuple([(slice(y,y+dy), slice(x,x+dx))
                      for y in range(0,ysize,dy) for x in range(0,xsize,dx)])
    #print ('chan_sec: {}'.format(chan_sec))

    # channel data section slices; shape=(16,2)
    data_sec = tuple([(slice(y,y+ysize_chan), slice(x,x+xsize_chan))
                      for y in range(0,ysize,dy+ysize_os) for x in range(0,xsize,dx)])
    #print ('data_sec: {}'.format(data_sec))

    # channel vertical overscan section slices; shape=(16,2)
    # cut off [ncut] pixels to avoid including pixels on the edge of the
    # overscan that are contaminated with flux from the image
    # and also discard last column as can have high value
    ncut = 5
    os_sec_vert = tuple([(slice(y,y+dy), slice(x+xsize_chan+ncut,x+dx-1))
                         for y in range(0,ysize,dy) for x in range(0,xsize,dx)])
    #print ('os_sec_vert: {}'.format(os_sec_vert))

    # channel horizontal overscan sections; shape=(16,2)
    # cut off [ncut] pixels to avoid including pixels on the edge of the
    # overscan that are contaminated with flux from the image
    ysize_os_cut = ysize_os - ncut
    os_sec_hori = tuple([(slice(y,y+ysize_os_cut), slice(x,x+dx))
                         for y in range(dy-ysize_os_cut,dy+ysize_os_cut,ysize_os_cut)
                         for x in range(0,xsize,dx)])
    #print ('os_sec_hori: {}'.format(os_sec_hori))
    
    # channel reduced data section slices; shape=(16,2)
    data_sec_red = tuple([(slice(y,y+ysize_chan), slice(x,x+xsize_chan))
                          for y in range(0,ysize-ny*ysize_os,ysize_chan)
                          for x in range(0,xsize-nx*xsize_os,xsize_chan)])
    #print ('data_sec_red: {}'.format(data_sec_red))
    
    return chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red


################################################################################

def os_corr(data, header, imgtype, tel=None, log=None):

    """Function that corrects [data] for the overscan signal in the
       vertical and horizontal overscan strips. The definitions of the
       different data/overscan/channel sections are taken from
       [set_blackbox].  The function returns a data array that consists of
       the data sections only, i.e. without the overscan regions. The
       [header] is update in plac.

    """
 
    if get_par(set_zogy.timing,tel):
        t = time.time()

    chan_sec, data_sec, os_sec_hori, os_sec_vert, data_sec_red = (
        define_sections(np.shape(data), tel=tel))
        
    # use median box filter with width [dcol] to decrease the noise
    # level in the overscan column's clipped mean for the horizontal
    # overscan when it has a limited amount of pixels
    nrows_hos = np.shape(data[os_sec_hori[0]])[0]
    if nrows_hos <= 100:
        dcol = 15 # after testing, 15-21 seem decent widths to use
    else:
        # otherwise, determine it per column
        dcol = 1
    dcol_half = int(dcol/2.)+1

    # number of data columns and rows in the channel (without overscans)
    ncols = get_par(set_bb.xsize_chan,tel)
    nrows = get_par(set_bb.ysize_chan,tel)
    
    # initialize output data array (without overscans)
    ny = get_par(set_bb.ny,tel)
    nx = get_par(set_bb.nx,tel)
    data_out = np.zeros((nrows*ny, ncols*nx), dtype='float32')

    # and arrays to calculate average means and stds over all channels
    nchans = np.shape(data_sec)[0]
    mean_vos = np.zeros(nchans)
    std_vos = np.zeros(nchans)

    vos_poldeg = get_par(set_bb.voscan_poldeg,tel)
    nrows_chan = np.shape(data[chan_sec[0]])[0]
    
    for i_chan in range(nchans):

        # -----------------
        # vertical overscan
        # -----------------
        
        # first subtract a low-order polynomial fit to the clipped
        # mean (not median!) of the vertical overcan section from the
        # entire channel

        # determine clipped mean for each row
        data_vos = data[os_sec_vert[i_chan]]
        mean_vos_col, __, __ = sigma_clipped_stats(data_vos.astype('float64'),
                                                   axis=1, mask_value=0)

        y_vos = np.arange(nrows_chan)
        # fit low order polynomial
        try:
            p = np.polyfit(y_vos, mean_vos_col, vos_poldeg)
        except Exception as e:
            if log is not None:
                log.info(traceback.format_exc())
                log.info('exception was raised during polynomial fit to channel {} '
                         'vertical overscan'.format(i_chan))

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
        # subtract this off the entire channel
        data[chan_sec[i_chan]] -= fit_vos_col.reshape(nrows_chan,1)
        
        #plt.plot(y_vos, mean_vos_col, color='black')
        #plt.plot(y_vos, fit_vos_col, color='red')
        #plt.savefig('test_poly_{}.pdf'.format(i_chan))
        #plt.close()        
        
        data_vos = data[os_sec_vert[i_chan]]
        # determine mean and std of overscan subtracted vos:
        __, __, std_vos[i_chan] = sigma_clipped_stats(data_vos.astype('float64'),
                                                      mask_value=0)
        # the above mean_vos is close to zero; instead use the mean polyfit value
        # (but keep the std_vos calculated above)
        mean_vos[i_chan] = np.mean(fit_vos_col)
                
        # -------------------
        # horizontal overscan
        # -------------------

        # determine the running clipped mean of the overscan using all
        # values across [dcol] columns, for [ncols] columns
        data_hos = data[os_sec_hori[i_chan]]

        # replace very high values (due to bright objects on edge of
        # channel) with function [replace_pix] in zogy.py
        mask_hos = (data_hos > 2000.)
        # add couple of pixels connected to this mask
        mask_hos = ndimage.binary_dilation(mask_hos, structure=np.ones((3,3)).astype('bool'))

        # interpolate spline over these pixels
        if imgtype == 'object':
            data_hos_replaced = inter_pix (data_hos, std_vos[i_chan], mask_hos, dpix=10, k=2,
                                           log=log)
        
        # determine clipped mean for each column
        mean_hos, __, __ = sigma_clipped_stats(data_hos.astype('float64'), axis=0,
                                               mask_value=0)
        if dcol > 1:
            oscan = [np.median(mean_hos[max(k-dcol_half,0):min(k+dcol_half+1,ncols)])
                     for k in range(ncols)]
            # do not use the running mean for the first column(s)
            oscan[0:dcol_half] = mean_hos[0:dcol_half]
        else:
            oscan = mean_hos[0:ncols]
            
        # subtract horizontal overscan
        data[data_sec[i_chan]] -= oscan
        # place into [data_out]
        data_out[data_sec_red[i_chan]] = data[data_sec[i_chan]] 


        
    def add_stats(mean_arr, std_arr, label, key_label):
        # add headers outside above loop to make header more readable
        for i_chan in range(nchans):
            header['BIASM{}{}'.format(i_chan+1, key_label)] = (
                mean_arr[i_chan], '[e-] channel {} mean vert. overscan {}'
                .format(i_chan+1, label))
        for i_chan in range(nchans):
            header['RDN{}{}'.format(i_chan+1, key_label)] = (
                std_arr[i_chan], '[e-] channel {} sigma (STD) vert. overscan {}'
                .format(i_chan+1, label))

    result = add_stats(mean_vos, std_vos, '', '')

    # write the average from both the means and standard deviations
    # determined for each channel to the header
    header['BIASMEAN'] = (np.mean(mean_vos), '[e-] average all channel means vert. overscan')
    header['RDNOISE'] = (np.mean(std_vos), '[e-] average all channel sigmas vert. overscan')

        
    # if the image is a flatfield, add some header keywords with
    # the statistics of [data_out]
    if imgtype == 'flat':
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
        __, median, std = sigma_clipped_stats(data_out[index_stat].astype('float64'),
                                              mask_value=0)
        header['FLATMED'] = (median, '[e-] median flat')
        header['FLATSTD'] = (std, '[e-] sigma (STD) flat')


    if log is not None:
        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t, label='os_corr', log=log)

    return data_out


################################################################################

def xtalk_corr (data, crosstalk_file):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    # read file with corrections
    victim, source, correction = np.loadtxt(crosstalk_file,unpack=True)
    # convert to indices
    victim -= 1
    source -= 1
        
    # channel image sections
    chan_sec, __, __, __, __ = define_sections(np.shape(data), tel=tel)
    # number of channels
    nchans = np.shape(chan_sec)[0]
        
    # the following 2 lines are to shift the channels to those
    # corresponding to the new channel definition with layout:
    #
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    #
    # this is assuming that the crosstalk corrections were
    # determined for the following layout
    #
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    #victim = (victim+nchans/2) % nchans
    #source = (source+nchans/2) % nchans
    #
    # apparently the corrections were determined for the new layout,
    # so the above swap is not necessary
    
    # loop arrays in file and correct the channels accordingly
    for k in range(len(victim)):
        data[chan_sec[int(victim[k])]] -= data[chan_sec[int(source[k])]]*correction[k]


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
    
    # N.B.: note that the channel numbering here:
    #
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # 
    # is not the same as that assumed with the gain.
    #
    # height,width = 5300, 1500 # = ccd_sec()
    # for victim in range(1,17):
    #     if victim < 9:
    #         j, i = 1, 0
    #     else:
    #         j, i = 0, 8
    #     print (victim, height*j, height*(j+1), width*(int(victim)-1-i), width*(int(victim)-i))
    #
    # victim is not the channel index, but number
    #
    # [vpn224246:~] pmv% python test_xtalk.py
    # 1 5300 10600 0 1500
    # 2 5300 10600 1500 3000
    # 3 5300 10600 3000 4500
    # 4 5300 10600 4500 6000
    # 5 5300 10600 6000 7500
    # 6 5300 10600 7500 9000
    # 7 5300 10600 9000 10500
    # 8 5300 10600 10500 12000
    # 9 0 5300 0 1500
    # 10 0 5300 1500 3000
    # 11 0 5300 3000 4500
    # 12 0 5300 4500 6000
    # 13 0 5300 6000 7500
    # 14 0 5300 7500 9000
    # 15 0 5300 9000 10500
    # 16 0 5300 10500 12000

        
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='xtalk_corr', log=log)

    return data

    
################################################################################

def nonlin_corr(data, nonlin_corr_file, log=None):

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


    if log is not None:
        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t, label='nonlin_corr', log=log)
        
    return data


################################################################################

def gain_corr(data, header, tel=None, log=None):

    """Returns [data] corrected for the [gain] defined in [set_bb.gain]
       for the different channels

    """
 
    if get_par(set_zogy.timing,tel):
        t = time.time()

    gain = get_par(set_bb.gain,tel)
    # channel image sections
    chan_sec, __, __, __, __ = define_sections(np.shape(data), tel=tel)

    nchans = np.shape(chan_sec)[0]
    for i_chan in range(nchans):
        data[chan_sec[i_chan]] *= gain[i_chan]
        header['GAIN{}'.format(i_chan+1)] = (gain[i_chan], 'gain applied to '
                                             'channel {}'.format(i_chan+1))

    if log is not None:
        if get_par(set_zogy.timing,tel):
            log_timing_memory (t0=t, label='gain_corr', log=log)
        
    return data

    # check if different channels in [set_bb.gain] correspond to the
    # correct channels; currently indices of gain correspond to the
    # channels as follows:
    #
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
    # [ 0, 1,  2,  3,  4,  5,  6,  7]

    # g = gain()
    # height,width = 5300, 1500
    # for (j,i) in [(j,i) for j in range(2) for i in range(8)]:
    #     data[height*j:height*(j+1),width*i:width*(i+1)]*=g[i+(j*8)]
    #
    # height, width = 5300, 1500
    # for (j,i) in [(j,i) for j in range(2) for i in range(8)]: print (height*j, height*(j+1),width*i, width*(i+1), i+(j*8))
    # 0 5300 0 1500 0
    # 0 5300 1500 3000 1
    # 0 5300 3000 4500 2
    # 0 5300 4500 6000 3
    # 0 5300 6000 7500 4
    # 0 5300 7500 9000 5
    # 0 5300 9000 10500 6
    # 0 5300 10500 12000 7
    # 5300 10600 0 1500 8
    # 5300 10600 1500 3000 9
    # 5300 10600 3000 4500 10
    # 5300 10600 4500 6000 11
    # 5300 10600 6000 7500 12
    # 5300 10600 7500 9000 13
    # 5300 10600 9000 10500 14
    # 5300 10600 10500 12000 15


################################################################################

def get_path (date, dir_type):

    # define path

    # date can be any of yyyy/mm/dd, yyyy.mm.dd, yyyymmdd,
    # yyyy-mm-dd or yyyy-mm-ddThh:mm:ss.s; if the latter is
    # provided, make sure to set [date_dir] to the date of the
    # evening before UT midnight
    #
    # rewrite this block using astropy.time; very confusing now
    if 'T' in date:
        if '.' in date:
            # rounds date to microseconds as more digits can't be defined in the format (next line)
            date = str(Time(date, format='isot')) 
            date_format = '%Y-%m-%dT%H:%M:%S.%f'
            high_noon = 'T12:00:00.0'
        else:
            date_format = '%Y-%m-%dT%H:%M:%S'
            high_noon = 'T12:00:00'

        date_ut = dt.datetime.strptime(date, date_format).replace(tzinfo=gettz('UTC'))
        date_noon = date.split('T')[0]+high_noon
        date_local_noon = dt.datetime.strptime(date_noon, date_format).replace(
            tzinfo=gettz(get_par(set_zogy.obs_timezone,tel)))
        if date_ut < date_local_noon: 
            # subtract day from date_only
            date = (date_ut - dt.timedelta(1)).strftime('%Y-%m-%d')
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
        log.error('[dir_type] not one of "read" or "write"')
        
    path = '{}/{}'.format(root_dir, date_dir)
    if '//' in path:
        print ('replacing double slash in path name: {}'.format(path))
        path = path.replace('//','/')
    
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

    """Function to sort raw files by type.  Globs all files in read_path
       and to sorts files into bias, flat and science images using the
       IMAGETYP header keyword.  Similar to Kerry's function in
       BGreduce, slightly adapted as sorting by filter is not needed.

    """
       
    #glob all raw files and sort
    if recursive:
        all_files = sorted(glob.glob('{}/**/{}'.format(read_path, search_str),
                                     recursive=recursive))
    else:
        all_files = sorted(glob.glob('{}/{}'.format(read_path, search_str)))


    biases = [] #list of biases
    darks = [] #list of darks
    flats = [] #list of flats
    objects = [] # list of science images
    others = [] # list of other images 
    
    for i, filename in enumerate(all_files): #loop through raw files

        header = read_hdulist(filename, get_data=False, get_header=True)
        
        if 'IMAGETYP' not in header:
            q.put(logger.info('keyword IMAGETYP not present in header of image '
                              '{}; skipping it'.format(filename)))
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

def unzip(imgname, put_lock=True, timeout=None):

    """Unzip a gzipped of fpacked file.
       Same [subpipe] function STAP_unzip.
    """

    if put_lock:
        lock.acquire()

    if '.gz' in imgname:
        print ('gunzipping {}'.format(imgname))
        subprocess.call(['gunzip',imgname])
        imgname = imgname.replace('.gz','')
    elif '.fz' in imgname:
        print ('funpacking {}'.format(imgname))
        subprocess.call(['funpack','-D',imgname])
        imgname = imgname.replace('.fz','')

    if put_lock:
        lock.release()

    return imgname
        
    
################################################################################

class MyLogger(object):
    '''Logger to control logging and uploading to slack.

    :param log: pipeline log file
    :type log: Logger
    :param mode: mode of pipeline
    :type mode: str
    :param log_stream: stream for log file
    :type log_stream: instance
    :param slack_upload: upload to slack
    :type slack_upload: bool
    '''

    def __init__(self, log, mode, log_stream, slack_upload):
        self._log = log
        self._mode = mode
        self._log_stream = log_stream
        self._slack_upload = slack_upload

    def info(self, text):
        '''Function to log at the INFO level.
        
        Logs messages to log file at the INFO level. If the night mode of the pipeline
        is running and 'Successfully' appears in the message, upload the message to slack.
        This allows only the overall running of the night pipeline to be uploaded to slack.
        
        :param text: message from pipeline
        :type text: str
        :exceptions: ConnectionError
        '''
        self._log.info(text)
        message = self._log_stream.getvalue()
        #only allow selected messages in night mode of pipeline to upload to slack
        if self._slack_upload is True and self._mode == 'night' and 'Successfully' in message: 
            try:
                self.slack(self._mode,text) #upload to slack
            except ConnectionError: #if connection error occurs, add to log
                self._log.error('Connection error: failed to connect to slack. Above meassage not uploaded.')

    def warning(self, text):
        '''Function to log at the INFO level.

        Logs messages to log file at the WARN level.'''

        self._log.warning(text)
        message = self._log_stream.getvalue()

    def error(self, text):
        '''Function to log at the ERROR level.

        Logs messages to log file at the ERROR level. If the night mode of the pipeline
        is running, upload the message to slack. This allows only the overall running of
        the night pipeline to be uploaded to slack.

        :param text: message from pipeline
        :type text: str
        :exceptions: ConnectionError
        '''
        self._log.error(text)
        message = self._log_stream.getvalue()
        if self._slack_upload is True and self._mode == 'night': #only night mode of pipeline uploads to slack
            try:
                self.slack(self._mode,text) #upload to slack
            except ConnectionError: #if connection error occurs, add to log
                self._log.error('Connection error: failed to connect to slack. Above meassage not uploaded.')

    def critical(self, text):
        '''Function to log at the CRITICAL level.

        Logs messages to log file at the CRITICAL level. If the night mode of the pipeline
        is running, upload the message to slack. This allows only the overall running of
        the night pipeline to be uploaded to slack. Pipeline will exit on critical errror.
        
        :param text: message from pipeline
        :type text: str
        :exceptions: ConnectionError
        :raises: SystemExit
        '''
        self._log.critical(text)
        message = self._log_stream.getvalue()
        if self._slack_upload is True and self._mode == 'night': #only night mode of pipeline uploads to slack
            try:
                self.slack('critical',text) #upload to slack
            except ConnectionError:
                self._log.error('Connection error: failed to connect to slack. Above meassage not uploaded.') #if connection error occurs, add to log
        raise SystemExit

    def slack(self, channel, message):
        '''Slack bot for uploading messages to slack.

        :param message: message to upload
        :type message: str
        '''
        slack_client().api_call("chat.postMessage", channel=channel,  text=message, as_user=True)


################################################################################

def copying(file):
    '''Waits for file size to stablize.

    Function that waits until the given file size is no longer changing before returning.
    This ensures the file has finished copying before the file is accessed.

    :param file: file
    :type file: str
    '''
    copying_file = True #file is copying
    size_earlier = -1 #set inital size of file
    while copying_file:
        size_now = os.path.getsize(file) #get current size of file
        if size_now == size_earlier: #if the size of the file has not changed, return
            time.sleep(1)
            return
        else: #if the size of the file has changed
            size_earlier = os.path.getsize(file) #get new size of file
            time.sleep(1) #wait


################################################################################

def action(queue):
    '''Action to take during night mode of pipeline.

    For new events, continues if it is a fits file. '''

    while True:

        if queue.empty():
            q.put(logger.info('queue is empty for now'))

        # get event from queue
        event = queue.get(True)

        try:
            # get name of new file
            filename = str(event.src_path)

            q.put(logger.info('detected a new file: {}'.format(filename)))
            
            #only continue if event is a fits
            if 'fits' in filename: 

                # if filename is a temporary rsync copy (default
                # behaviour of rsync is to create a temporary file
                # starting with .[filename].[randomstr]; can be
                # changed with option "--inplace"), then let filename
                # refer to the eventual file created by rsync
                fn_head, fn_tail = os.path.split(filename)
                if fn_tail[0] == '.':
                    filename = '{}/{}'.format(
                        fn_head, '.'.join(fn_tail.split('.')[1:-1]))
                    q.put(logger.info('changed filename from rsync copy {} to {}'
                                      .format(event.src_path, filename)))
                
                # this while loop below replaces the old [copying]
                # function; it times out after wait_max is reached
                wait_max = 30
                t0 = time.time()
                nsleep = 0
                while time.time()-t0 < wait_max:
                    
                    try:
                        # give file a bit of time to arrive
                        time.sleep(5)
                        nsleep += 1
                        # read the file
                        data = read_hdulist(filename)
                    except Exception as e:
                        process = False
                        if nsleep==1:
                            q.put(logger.info(
                                'problem reading file {} but will keep trying '
                                'for {}s; current exception: {}'.format(
                                    filename, wait_max, e)))
                    else:
                        # if fits file was read fine, set process flag to True
                        process = True


                if process:
                    # if fits file was read fine, process it
                    try_blackbox_reduce (filename)
                    
                else:
                    q.put(logger.info('wait time for file {} exceeded {}s; '
                                      'bailing out with final exception: {}'
                                      .format(filename, wait_max, e)))

            else: #if 'fits' in filename: 
                q.put(logger.info('{} is not a fits file; skipping it'
                                  .format(filename)))
                    
        except AttributeError:
            # when blackbox is started in night mode, any file already
            # present in the relevant path will lead to an
            # AttributeError (in filename=str(event.src_path)); these
            # could be processed as well, but these are already added
            # to the queue before at the top in [run_blackbox]
            filename = event
            if False:
                q.put(logger.info('detected an old file: {}; not processing it '
                                  .format(filename)))

    return


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
    
    params = argparse.ArgumentParser(description='User parameters')
    params.add_argument('--telescope', type=str, default='ML1', 
                        help='Telescope name (ML1, BG2, BG3 or BG4); default=\'ML1\'')
    params.add_argument('--mode', type=str, default='day', 
                        help='Day or night mode of pipeline; default=\'day\'')
    params.add_argument('--date', type=str, default=None, help='Date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd or yyyy.mm.dd); default=None')
    params.add_argument('--read_path', type=str, default=None, help='Full path to the input raw data directory; if not defined it is determined from [set_blackbox.raw_dir], [telescope] and [date]; default=None') 
    params.add_argument('--recursive', type=str2bool, default=False, help='Recursively include subdirectories for input files; default=False')
    params.add_argument('--imgtypes', type=str, default=None, help='Only consider this(these) image type(s); default=None')
    params.add_argument('--filters', type=str, default=None, help='Only consider this(these) filter(s); default=None')
    params.add_argument('--image', type=str, default=None, help='Only process this particular image (requires full path); default=None')
    params.add_argument('--image_list', type=str, default=None, help='Only process images listed in ASCII file with this name; default=None')
    params.add_argument('--master_date', type=str, default=None, help='Create master file of type(s) [imgtypes] and filter(s) [filters] for this(these) date(s) (e.g. 2019 or 2019/10 or 2019-10-14); default=None')
    params.add_argument('--slack', default=True, help='Upload messages for night mode to slack; default=True')
    args = params.parse_args()

    run_blackbox (telescope=args.telescope, mode=args.mode, date=args.date, 
                  read_path=args.read_path, recursive=args.recursive, 
                  imgtypes=args.imgtypes, filters=args.filters, image=args.image,
                  image_list=args.image_list, master_date=args.master_date, 
                  slack=args.slack)

