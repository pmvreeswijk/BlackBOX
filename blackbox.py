
import os
from Settings import set_zogy, set_blackbox as set_bb

# setting number of threads through environment variable (used by
# e.g. astroscrappy) needs to be done before numpy is imported in
# [zogy]
# this needs to be done before numpy is imported in [zogy]
os.environ['OMP_NUM_THREADS'] = str(set_bb.nthread)

from zogy import *

import re   # Regular expression operations
import glob # Unix style pathname pattern expansion 
from multiprocessing import Pool, Manager, Lock, Queue
import datetime as dt 
from dateutil.tz import gettz
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle
from astropy.time import Time
from astropy import units as u
from scipy import ndimage
import astroscrappy
from acstools.satdet import detsat, make_mask, update_dq
import shutil
from StringIO import StringIO
#from slackclient import SlackClient as sc
import ephem  
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import ctypes

__version__ = '0.7.3'

#def init(l):
#    global lock
#    lock = l
    
def run_blackbox (telescope=None, mode=None, date=None, read_path=None, slack=None):

    global tel
    tel = telescope

    if get_par(set_zogy.timing,tel):
        t_run_blackbox = time.time()
    
    # initialize logging
    ####################

    if not os.path.isdir(get_par(set_bb.log_dir,tel)):
        os.makedirs(get_par(set_bb.log_dir,tel))
    
    global q, logger
    q = Manager().Queue() #create queue for logging

    genlog = logging.getLogger() #create logger
    genlog.setLevel(logging.INFO) #set level of logger
    formatter = logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s") #set format of logger
    logging.Formatter.converter = time.gmtime #convert time in logger to UCT
    genlogfile = '{}/{}_{}.log'.format(get_par(set_bb.log_dir,tel), telescope,
                                       dt.datetime.now().strftime('%Y%m%d_%H%m%S'))
    filehandler = logging.FileHandler(genlogfile, 'w+') #create log file
    filehandler.setFormatter(formatter) #add format to log file
    genlog.addHandler(filehandler) #link log file to logger

    log_stream = StringIO() #create log stream for upload to slack
    streamhandler_slack = logging.StreamHandler(log_stream) #add log stream to logger
    streamhandler_slack.setFormatter(formatter) #add format to log stream
    genlog.addHandler(streamhandler_slack) #link logger to log stream
    logger = MyLogger(genlog,mode,log_stream,slack) #load logger handler

    q.put(logger.info('processing in {} mode'.format(mode)))
    q.put(logger.info('log file: {}'.format(genlogfile)))
    q.put(logger.info('number of processes: {}'.format(get_par(set_bb.nproc,tel))))
    q.put(logger.info('number of threads: {}'.format(get_par(set_bb.nthread,tel))))

    # [read_path] is assumed to be the full path to the directory with
    # raw images to be processed; if not provided as input parameter,
    # it is defined using the input [date] with the function
    # [get_path]
    if read_path is None:
        read_path, __ = get_path(date, 'read')
        q.put(logger.info('processing files from directory: {}'.format(read_path)))
    else:
        # if it is provided but does not exist, exit
        if not os.path.isdir(read_path):
            loggger.critical('[read_path] directory provided does not exist:\n{}'
                             .format(read_path))
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

    if mode == 'day':

        # if in day mode, feed all bias, flat and science images (in
        # this order) to [blackbox_reduce] using multiprocessing
        filenames = sort_files(read_path, '*fits*')

        if get_par(set_bb.nproc,tel)==1 :

            # if only 1 process is requested, run it witout
            # multiprocessing; this will allow images to be shown on
            # the fly if [set_zogy.display] is set to True. In
            # multiprocessing mode this is not allowed (at least not a
            # macbook).
            print ('running with single processor')
            for filename in filenames:
                result = blackbox_reduce(filename, telescope, mode, read_path)

        else:

            # use [pool_func] to process list of files
            result = pool_func (blackbox_reduce, filenames, telescope, mode, read_path)


    elif mode == 'night':
        # if in night mode, check if anythin changes in input directory
        # and if there is a new file, feed it to [blackbox_reduce]

        # create queue for submitting jobs
        queue = Queue()
        # create pool with given number of processes and queue feeding
        # into action function
        pool = Pool(get_par(set_bb.nproc,tel), action, (queue,))

        # create and setup observer, but do not start just yet
        observer = Observer()
        observer.schedule(FileWatcher(queue, telescope, mode, read_path),
                          read_path, recursive=False)

        # glob any files already there
        filenames = sort_files(read_path, '*fits*')
        # loop through waiting files and add to pool
        for filename in filenames: 
            queue.put([filename, telescope, mode, read_path])

        # determine time of next sunrise
        obs = ephem.Observer()
        obs.lat = str(get_par(set_zogy.obs_lat,tel))
        obs.long = str(get_par(set_zogy.obs_long,tel))
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
        log_timing_memory (t0=t_run_blackbox, label='run_blackbox before fpacking', log=genlog)

    # now that all files have been processed, fpack!
    # create list of files to fpack
    list_2pack = prep_packlist (date)
    # use [pool_func] to process the list
    result = pool_func (fpack, list_2pack)

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t_run_blackbox, label='run_blackbox after fpacking', log=genlog)

    logging.shutdown()
    return


################################################################################

def pool_func (func, filelist, *args):

    #results = []
    #def log_result(result):
    #    results.append(result)

    try:
        pool = Pool(get_par(set_bb.nproc,tel))
        for filename in filelist:
            args_temp = [filename]
            for arg in args:
                args_temp.append(arg)
            pool.apply_async(func, args_temp) #, callback=log_result)
        pool.close()
        pool.join()
        #q.put(logger.info(results))
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [pool.apply_async({})]: {}'
                           .format(func, e)))


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
    else:
        # just add all fits files in [set_bb.raw_dir]/*/*/*/*.fits
        list_2pack.append(glob.glob('{}/*/*/*/*.fits'
                                        .format(get_par(set_bb.raw_dir,tel))))
        # just add all fits files in [telescope]/red/*/*/*/*.fits
        # (could do it more specifically by going through raw fits
        # files and finding out where their reduced images are)
        list_2pack.append(glob.glob('{}/*/*/*/*.fits'
                                       .format(get_par(set_bb.red_dir,tel))))
        # add fits files in bias and flat directories
        list_2pack.append(glob.glob('{}/*/*/*/bias/*.fits'
                                       .format(get_par(set_bb.red_dir,tel))))
        list_2pack.append(glob.glob('{}/*/*/*/flat/*.fits'
                                       .format(get_par(set_bb.red_dir,tel))))
    # leave the ref directories uncompressed, at least for the moment

    # flatten this list of files
    list_2pack = [item for sublist in list_2pack for item in sublist]
    # and get the unique items
    list_2pack = list(set(list_2pack))

    return list_2pack
    

################################################################################

def fpack (filename):

    """Fpack fits images; skip fits tables"""

    # fits check if extension is .fits
    if filename.split('.')[-1] == 'fits':
        # check if it is an image
        header = read_hdulist(filename, ext_header=0)
        if header['NAXIS']==2:
            # determine if integer or float image
            if header['BITPIX'] > 0:
                cmd = ['fpack', '-D', '-Y', filename]
            else:
                cmd = ['fpack', '-q', '16', '-D', '-Y', filename]
            subprocess.call(cmd)

    return filename
        
    
################################################################################
    
def blackbox_reduce (filename, telescope, mode, read_path):

    """Function that takes as input a single raw fits image and works to
       work through entire chain of reduction steps, from correcting
       for the gain and overscan to running ZOGY on the reduced image.

    """

    if get_par(set_zogy.timing,tel):
        t_blackbox_reduce = time.time()

    # For night mode, the image needs to be moved out of the directory
    # that is being monitored immediately, for one thing because it
    # will first get unzipped, and the unzipped file will be
    # recognized by the watchdog as a new file, which is a problem.
    #
    # We can do the same for day mode, as it is interesting to be able
    # to put all kinds of files in a single directory, and to let
    # BlackBOX move the files to the correct raw directory.

    if '.fz' in filename:
        ext = 1
    else:
        ext = 0
    # just read the header for the moment
    header = read_hdulist(filename, ext_header=ext)
    # and determine the raw data path (which is not necessarily the
    # same as the input [read_path])
    raw_path, __ = get_path(header['DATE-OBS'], 'read')

    if raw_path == read_path:

        if mode == 'night':
            # in night mode, [read_path] should not be the same as
            # [raw_path] because the images will be transferred to and
            # unpacked in [raw_path], which is problematic if that is the
            # same as the directory that is being monitored for new images
            q.put(logger.critical('in night mode, the directory [read_path] that '+
                                  'is being monitored should not be identical to '+
                                  'the standard [raw_path] directory: {}'
                                  .format(raw_path)))
    else:

        # move the image to [raw_path] if it does not already exist
        src = filename
        dest = '{}/{}'.format(raw_path, filename.split('/')[-1])
        if os.path.isfile(dest) or os.path.isfile(dest.replace('.fz','')):
            q.put(logger.warn('{} already exists'.format(dest)))
        else:
            shutil.move(src, dest)

        # and let [filename] refer to the image in [raw_path]
        filename = dest


    # read in image data and header; unzip image first if needed
    data, header = read_hdulist(unzip(filename), ext_data=0, ext_header=0,
                                dtype='float32')

    # extend the header with some useful keywords
    result = set_header(header, filename)

    q.put(logger.info('processing {}'.format(filename)))

    # defining various paths and output file names
    ##############################################
    
    # define [write_path] using the header DATE-OBS
    write_path, date_eve = get_path(header['DATE-OBS'], 'write')
    make_dir (write_path)
    bias_path = '{}/bias'.format(write_path)
    make_dir (bias_path)
    flat_path = '{}/flat'.format(write_path)
    make_dir (flat_path)

    # UT date (yyyymmdd) and time (hhmmss)
    utdate, uttime = date_obs_get(header).split('_')

    # if output file already exists, do not bother to redo it
    path = {'bias': bias_path, 'flat': flat_path, 'object': write_path}
    # 'IMAGETYP' keyword in lower case
    imgtype = header['IMAGETYP'].lower()
    filt = header['FILTER']
    exptime = int(header['EXPTIME'])
    fits_out = '{}/{}_{}_{}.fits'.format(path[imgtype], telescope, utdate, uttime)
    if imgtype == 'flat':
        fits_out = fits_out.replace('.fits', '_{}.fits'.format(filt))

    if imgtype == 'object':
        # if 'FIELD_ID' keyword is present in the header, which
        # is the case for the test
        if 'FIELD_ID' in header:
            obj = header['FIELD_ID']
        else:
            obj = header['OBJECT']
        # remove all non-alphanumeric characters from [obj] except for
        # '-' and '_'
        obj = ''.join(e for e in obj if e.isalnum() or e=='-' or e=='_')
        fits_out = fits_out.replace('.fits', '_red.fits')
        fits_out_mask = fits_out.replace('_red.fits', '_mask.fits')

        # and reference image
        ref_path = '{}/{}'.format(get_par(set_bb.ref_dir,tel), obj)
        make_dir (ref_path)
        ref_fits_out = '{}/{}_{}_red.fits'.format(ref_path, telescope, filt)
        ref_fits_out_mask = ref_fits_out.replace('_red.fits', '_mask.fits')

        if os.path.isfile(unzip(ref_fits_out)):
            header_ref = read_hdulist(ref_fits_out, ext_header=0)
            utdate_ref, uttime_ref = date_obs_get(header_ref).split('_')
            if utdate_ref==utdate and uttime_ref==uttime:
                q.put(logger.warn ('this image {} is the current reference image; skipping'
                                   .format(fits_out.split('/')[-1])))
                return

    # add reduced filename to header
    header['REDFILE'] = (fits_out.split('/')[-1], 'BlackBOX reduced file name')
            
    if os.path.isfile(fits_out):
        q.put(logger.warn ('corresponding reduced image {} already exist; skipping'
                           .format(fits_out.split('/')[-1])))
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
        # for biases and flats
        logfile = fits_out.replace('.fits','.log')
    else:
        # for object files, prepare the logfile in [tmp_path]
        logfile = '{}/{}'.format(tmp_path, fits_out.split('/')[-1]
                                 .replace('.fits','.log'))
    global log
    log = create_log (logfile)

    # immediately write some info to the log
    log.info('processing {}'.format(filename))
    log.info('image type: {}, filter: {}, exptime: {}s'
             .format(imgtype, filt, exptime))

    log.info('write_path: {}'.format(write_path))
    log.info('bias_path: {}'.format(bias_path))
    log.info('flat_path: {}'.format(flat_path))
    if imgtype == 'object':
        log.info('tmp_path: {}'.format(tmp_path))
        log.info('ref_path: {}'.format(ref_path))
    

    # gain correction
    #################
    try:
        log.info('correcting for the gain')
        gain_processed = False
        data = gain_corr(data, header)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [gain_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [gain_corr]: {}'.format(e))
    else:
        gain_processed = True
        header['GAIN'] = (1, '[e-/ADU] effective gain all channels')
    # following line needs to be outside if/else statements
    header['GAIN-P'] = (gain_processed, 'corrected for gain?')

    if get_par(set_zogy.display,tel):
        ds9_arrays(gain_cor=data)

    #args_in = [data, header}
    #args_out = data
    #proc_ok = try_func (gain_corr, args_in, args_out)
    #header['GAIN-P'] = (proc_ok, 'corrected for gain?')
    #if proc_ok:
    #    header['GAIN'] = (1, '[e-/ADU] effective gain all channels')

    
    # crosstalk correction
    ######################
    if imgtype == 'object':
        # not needed for biases or flats
        try: 
            log.info('correcting for the crosstalk')
            xtalk_processed = False
            data_old = xtalk_corr (data, get_par(set_bb.crosstalk_file,tel))
        except Exception as e:
            q.put(logger.info(traceback.format_exc()))
            q.put(log.error('exception was raised during [xtalk_corr]: {}'.format(e)))
            log.info(traceback.format_exc())
            log.error('exception was raised during [xtalk_corr]: {}'.format(e))
        else:
            xtalk_processed = True
        # following line needs to be outside if/else statements
        header['XTALK-P'] = (xtalk_processed, 'corrected for crosstalk?')
        header['XTALK-F'] = (get_par(set_bb.crosstalk_file,tel).strip('/')[-1],
                             'name crosstalk coefficients file')

        if get_par(set_zogy.display,tel):
            ds9_arrays(Xtalk_cor=data)
            
            
    # PMV 2018/12/20: non-linearity correction is not yet done, but
    # still add these keywords to the header
    header['NONLIN-P'] = (False, 'corrected for non-linearity?')
    header['NONLIN-F'] = ('', 'name non-linearity correction file')


    # overscan correction
    #####################
    try: 
        log.info('correcting for the overscan')
        os_processed = False
        data = os_corr(data, header)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [os_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [os_corr]: {}'.format(e))
    else:
        os_processed = True
    # following line needs to be outside if/else statements
    header['OS-P'] = (os_processed, 'corrected for overscan?')


    if get_par(set_zogy.display,tel):
        ds9_arrays(os_cor=data)


    # if IMAGETYP=bias, write [data] to fits and leave [blackbox_reduce]
    if imgtype == 'bias':
        fits.writeto(fits_out, data.astype('float32'), header, overwrite=True)
        return
        

    # master bias creation and subtraction
    ######################################
    try: 
        log.info('subtracting the master bias')
        mbias_processed = False
        lock.acquire()
        data = master_corr(data, header, bias_path, date_eve, 'bias')
        lock.release()
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [mbias_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [mbias_corr]: {}'.format(e))
    else:
        mbias_processed = True
    # following line needs to be outside if/else statements
    header['MBIAS-P'] = (mbias_processed, 'corrected for master bias?')

    
    # if IMAGETYP=flat, write [data] to fits and leave [blackbox_reduce]
    if imgtype == 'flat':
        fits.writeto(fits_out, data.astype('float32'), header, overwrite=True)
        return

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
        # following line needs to be outside if/else statements
        header['MASK-P'] = (mask_processed, 'mask image created?')


    if get_par(set_zogy.display,tel):
        ds9_arrays(mask=data_mask)


    # master flat creation and correction
    #####################################
    try: 
        log.info('flatfielding')
        mflat_processed = False
        lock.acquire()
        data = master_corr(data, header, flat_path, date_eve, 'flat',
                           data_mask=data_mask, filt=filt)
        lock.release()
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [mflat_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [mflat_corr]: {}'.format(e))
    else:
        mflat_processed = True
    # following line needs to be outside if/else statements
    header['MFLAT-P'] = (mflat_processed, 'corrected for master flat?')

    
    # PMV 2018/12/20: fringe correction is not yet done, but
    # still add these keywords to the header
    header['MFRING-P'] = (False, 'corrected for master fringe map?')
    header['MFRING-F'] = ('', 'name of master fringe map applied')


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
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [cosmics_corr]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [cosmics_corr]: {}'.format(e))
    else:
        cosmics_processed = True
    # following line needs to be outside if/else statements
    header['COSMIC-P'] = (cosmics_processed, 'corrected for cosmic rays?')

    
    if get_par(set_zogy.display,tel):
        ds9_arrays(data=data_precosmics, cosmic_cor=data, mask=data_mask)
        print (header['NCOSMICS'])
        

    # satellite trail detection
    ###########################
    try: 
        log.info('detecting satellite trails')
        sat_processed = False
        data_mask = sat_detect(data, header, data_mask, header_mask,
                               tmp_path)
    except Exception as e:
        q.put(logger.info(traceback.format_exc()))
        q.put(logger.error('exception was raised during [sat_detect]: {}'.format(e)))
        log.info(traceback.format_exc())
        log.error('exception was raised during [sat_detect]: {}'.format(e))
    else:
        sat_processed = True
    # following line needs to be outside if/else statements
    header['SAT-P'] = (sat_processed, 'corrected for cosmic rays?')
    
    # add some more info to mask header
    result = mask_header(data_mask, header_mask)
    
    # write data and mask to output images in [tmp_path]
    log.info('writing reduced image and mask to {}'.format(tmp_path))
    new_fits = '{}/{}'.format(tmp_path, fits_out.split('/')[-1]) 
    new_fits_mask = new_fits.replace('_red.fits', '_mask.fits')
    fits.writeto(new_fits, data.astype('float32'), header, overwrite=True)
    fits.writeto(new_fits_mask, data_mask.astype('uint8'), header_mask,
                 overwrite=True)
    
    if get_par(set_zogy.display,tel):
        ds9_arrays(mask=data_mask)
        print (header['NSATS'])

        
    # run zogy's [optimal_subtraction]
    ##################################
    try: 
        log.info ('running optimal image subtraction')
        zogy_processed = False
        
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
        ref_being_made = check_ref(ref_ID_filt, (obj, filt))
        log.info('is reference for same OBJECT and FILTER being_made now?: {}'
                 .format(ref_being_made))
        lock.release()
        
        if ref_being_made:
            # if reference in this filter is being made, let the affected
            # process wait until reference building is done
            if ref_being_made:
                while check_ref(ref_ID_filt, (obj, filt)):
                    log.info ('waiting for reference job to be finished for '+
                              'OBJECT: {}, FILTER: {}'.format(obj, filt))
                    time.sleep(5)
                log.info ('done waiting for reference job to be finished for '+
                          'OBJECT: {}, FILTER: {}'.format(obj, filt))
                    
        # lock the following block to allow only a single process to
        # execute the reference image creation
        #lock.acquire()

        # if ref image has not yet been processed:
        if not os.path.isfile(unzip(ref_fits_out)):
#            refjob = True
#        else:
#            refjob = False
#        if refjob:
            
            # update [ref_ID_filt] queue with a tuple with this OBJECT
            # and FILTER combination
            ref_ID_filt.put((obj, filt))

            log.info('making ref image')

            log.info('new_fits: {}'.format(new_fits))
            log.info('new_fits_mask: {}'.format(new_fits_mask))

            result = optimal_subtraction(ref_fits=new_fits,
                                         ref_fits_mask=new_fits_mask,
                                         set_file='Settings.set_zogy',
                                         log=log, verbose=None,
                                         nthread=get_par(set_bb.nthread,tel),
                                         telescope=telescope)

            if get_par(set_zogy.timing,tel):
                log_timing_memory (t0=t_blackbox_reduce, label='blackbox_reduce', log=log)
                
            # copy selected output files to reference directory
            ref_base = ref_fits_out.split('_red.fits')[0]
            tmp_base = new_fits.split('_red.fits')[0]
            result = copy_files2keep(tmp_base, ref_base, get_par(set_bb.ref_2keep,tel))

            # now that reference is built, remove this reference ID
            # and filter combination from the [ref_ID_filt] queue
            lock.acquire()
            result = check_ref(ref_ID_filt, (obj, filt), method='remove')
            lock.release()
            
        else:

         #lock.release()        
#        if not refjob:
            
            # make symbolic links to all files in the reference
            # directory with the same filter
            ref_files = glob.glob('{}/{}*{}*'.format(ref_path, telescope, filt))
            for ref_file in ref_files:
                # unzip file first if needed
                ref_file = unzip(ref_file)
                os.symlink(unzip(ref_file), '{}/{}'
                           .format(tmp_path, ref_file.split('/')[-1]))

            ref_fits = '{}/{}'.format(tmp_path, ref_fits_out.split('/')[-1])
            ref_fits_mask = '{}/{}'.format(tmp_path, ref_fits_out_mask.split('/')[-1])
            
            log.info('new_fits: {}'.format(new_fits))
            log.info('new_fits_mask: {}'.format(new_fits_mask))
            log.info('ref_fits: {}'.format(ref_fits))
            log.info('ref_fits_mask: {}'.format(ref_fits_mask))
        
            result = optimal_subtraction(new_fits=new_fits,
                                         ref_fits=ref_fits,
                                         new_fits_mask=new_fits_mask,
                                         ref_fits_mask=ref_fits_mask,
                                         set_file='Settings.set_zogy',
                                         log=log, verbose=None,
                                         nthread=get_par(set_bb.nthread,tel),
                                         telescope=telescope)

            if get_par(set_zogy.timing,tel):
                log_timing_memory (t0=t_blackbox_reduce, label='blackbox_reduce', log=log)

            # copy selected output files to new directory
            new_base = fits_out.split('_red.fits')[0]
            tmp_base = new_fits.split('_red.fits')[0]
            result = copy_files2keep(tmp_base, new_base, get_par(set_bb.new_2keep,tel))


        lock.acquire()
        # change to [run_dir]
        if get_par(set_zogy.make_plots,tel):
            os.chdir(get_par(set_bb.run_dir,tel))
        # and delete [tmp_path] if [set_bb.keep_tmp] not True
        if not get_par(set_bb.keep_tmp,tel) and os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)
        lock.release()
        
            
    except Exception as e:
        log.info(traceback.format_exc())
        log.error('exception was raised during [optimal_subtraction]: {}'.format(e))
    else:
        zogy_processed = True

        
    return
        

################################################################################

def check_ref (queue_ref, obj_filt, method=None):

    mycopy = []
    ref_being_made = False
    while True:
        try:
            elem = queue_ref.get(False)
        except:
            break
        else:
            mycopy.append(elem)

    for elem in mycopy:
        if elem == obj_filt:
            ref_being_made = True
            if method != 'remove':
                queue_ref.put(elem)
                time.sleep(0.1)

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

    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s, %(process)s] '+
                                     '%(message)s [%(funcName)s, line %(lineno)d]',
                                     '%Y-%m-%dT%H:%M:%S')
    logging.Formatter.converter = time.gmtime #convert time in logger to UTC
    log = logging.getLogger()

    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    log.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logFormatter)
    streamHandler.setLevel(logging.WARN)
    log.addHandler(streamHandler)

    return log
    

################################################################################

def make_dir(path, empty=False):

    """Function to make directory, which is locked to use by 1 process.
       If [empty] is True and the directory already exists, it will
       first be removed.
    """

    lock.acquire()
    # if already exists but needs to be empty, remove it first
    if os.path.isdir(path) and empty:
        shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)
    lock.release()
    return


################################################################################

def copy_files2keep (tmp_base, dest_base, ext2keep):

    """Function to copy files with base name [tmp_base] and extensions
    [ext2keep] to files with base name [dest_base] with the same
    extensions. The base names should include the full path.
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
                    log.info('copying {} to {}'.format(tmpfile, destfile))
                    shutil.copyfile(tmpfile, destfile)

    return


################################################################################

def sat_detect (data, header, data_mask, header_mask, tmp_path):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    #bin data
    binned_data = data.reshape(np.shape(data)[0]/get_par(set_bb.sat_bin,tel),
                               get_par(set_bb.sat_bin,tel),
                               np.shape(data)[1]/get_par(set_bb.sat_bin,tel),
                               get_par(set_bb.sat_bin,tel)).sum(3).sum(1)
    satellite_fitting = False

    for j in range(3):
        #write binned data to tmp file
        fits_binned_mask = ('{}/{}'.format(
            tmp_path, tmp_path.split('/')[-1].replace('_red','_binned_satmask.fits')))
        fits.writeto(fits_binned_mask, binned_data, overwrite=True)
        #detect satellite trails
        results, errors = detsat(fits_binned_mask, chips=[0],
                                 n_processes=get_par(set_bb.nthread,tel),
                                 buf=40, sigma=3, h_thresh=0.2)
        #create satellite trail if found
        trail_coords = results[(fits_binned_mask,0)] 
        #continue if satellite trail found
        if len(trail_coords) > 0:
            unique_dir = tmp_path #Danielle: unique_dir wasn't defined. I think it's supposed to be the tmp_path, or isn't it? 
            trail_segment = trail_coords[0]
            try: 
                #create satellite trail mask
                mask_binned = make_mask(fits_binned_mask, 0, trail_segment, sublen=5,
                                        pad=0, sigma=5, subwidth=5000).astype(np.uint8)
            except ValueError:
                #if error occurs, add comment
                print ('Warning: satellite trail found but could not be fitted for file {} and is not included in the mask.'
                       .format(unique_dir.split('/')[-1]))
                break
            satellite_fitting = True
            binned_data[mask_binned == 1] = np.median(binned_data)
            fits_old_mask = unique_dir+'/old_mask.fits'
            if os.path.isfile(fits_old_mask):
                old_mask = read_hdulist(fits_old_mask, ext_data=0)
                mask_binned = old_mask+mask_binned
            fits.writeto(fits_old_mask, mask_binned, overwrite=True)
        else:
            break
    if satellite_fitting == True:
        #unbin mask
        mask_sat = np.kron(mask_binned, np.ones((get_par(set_bb.sat_bin,tel),
                                                 get_par(set_bb.sat_bin,tel)))).astype(np.uint8)
        # add pixels affected by cosmic rays to [data_mask]
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

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='sat_detect', log=log)

    return data_mask

        
################################################################################

def cosmics_corr (data, header, data_mask, header_mask):

    if get_par(set_zogy.timing,tel):
        t = time.time()
    
    satlevel_electrons = get_par(set_bb.satlevel,tel) * np.mean(get_par(set_bb.gain,tel)) 
    mask_cr, data = astroscrappy.detect_cosmics(
        data, inmask=(data_mask!=0), sigclip=get_par(set_bb.sigclip,tel),
        sigfrac=get_par(set_bb.sigfrac,tel), objlim=get_par(set_bb.objlim,tel),
        niter=get_par(set_bb.niter,tel), readnoise=header['RDNOISE'],
        satlevel=satlevel_electrons, cleantype='medmask')
    
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
    header['NCOSMICS'] = (ncosmics, 'number of cosmic rays identified')

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

    fits_bpm = unzip(get_par(set_bb.bad_pixel_mask,tel))
    if os.path.isfile(fits_bpm):
        # if it exists, read it
        data_mask = read_hdulist(fits_bpm, ext_data=0)
    else:
        # if not, create uint8 array of zeros with same shape as
        # [data]
        data_mask = np.zeros(np.shape(data), dtype='uint8')

    # mask of pixels with non-finite values in [data]
    mask_infnan = ~np.isfinite(data)
    # replace those pixel values with zeros
    data[mask_infnan] = 0
    # and add them to [data_mask] with same value defined for 'bad' pixels
    # unless that pixel was already masked
    data_mask[(mask_infnan) & (data_mask==0)] += get_par(set_zogy.mask_value['bad'],tel)
    
    # identify saturated pixels
    satlevel_electrons = get_par(set_bb.satlevel,tel)*np.mean(get_par(set_bb.gain,tel))
    mask_sat = (data >= satlevel_electrons)
    # add them to the mask of edge and bad pixels
    data_mask[mask_sat] += get_par(set_zogy.mask_value['saturated'],tel)

    # and pixels connected to saturated pixels
    struct = np.ones((3,3), dtype=bool)
    mask_satconnect = ndimage.binary_dilation(mask_sat, structure=struct)
    # add them to the mask
    data_mask[(mask_satconnect) & (~mask_sat)] += get_par(set_zogy.mask_value['saturated-connected'],tel)

    # create initial mask header 
    header_mask = fits.Header()
    header_mask['SATURATE'] = (satlevel_electrons, '[e-] adopted saturation threshold')
    # also add this to the header of image itself
    header['SATURATE'] = (satlevel_electrons, '[e-] adopted saturation threshold')
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

def master_corr (data, header, path, date_eve, imtype, data_mask=None, filt=None):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    if imtype=='flat':
        fits_master = '{}/{}_{}_{}.fits'.format(path, imtype, date_eve, filt)
    elif imtype=='bias':
        fits_master = '{}/{}_{}.fits'.format(path, imtype, date_eve)

    log.info('fits_master: {}'.format(fits_master))
        
    if not os.path.isfile(unzip(fits_master)):

        # prepare master from files in [path]
        if imtype=='flat':
            file_list = sorted(glob.glob('{}/*_{}.fits*'.format(path, filt)))
        elif imtype=='bias':
            file_list = sorted(glob.glob('{}/*fits*'.format(path)))

        # initialize cube of images to be combined
        nfiles = np.shape(file_list)[0]

        # if there are too few frames to make tonight's master, look
        # for a nearby master flat instead
        if nfiles < 3:

            fits_master_close = get_closest_biasflat(date_eve, imtype, filt=filt)
            if fits_master_close is not None:

                fits_master_close = unzip(fits_master_close)
                print ('Warning: too few images available to produce master {}; instead using\n{}'
                       .format(imtype, fits_master_close))
                # create symbolic link so future files will automatically
                # use this as the master flat
                os.symlink(fits_master_close, fits_master)

            else:
                log.error('no alternative master {} found'.format(imtype))
                return data
                
        else:
            
            print ('making master {} in filter {}'.format(imtype, filt))

            # assuming that individual flats/biases have the same shape as the input data
            ysize, xsize = np.shape(data)
            master_cube = np.zeros((nfiles, ysize, xsize), dtype='float32')

            # fill the cube
            for i_file, filename in enumerate(file_list):
                master_cube[i_file], header_temp = read_hdulist(file_list[i_file],
                                                                ext_data=0, ext_header=0)

                if imtype=='flat':
                    # divide by median over the region [set_bb.flat_norm_sec]
                    mean, std, median = clipped_stats(
                        master_cube[i_file][get_par(set_bb.flat_norm_sec,tel)])
                    print ('flat name: {}, mean: {}, std: {}, median: {}'
                           .format(filename, mean, std, median))
                    master_cube[i_file] /= median
                    
                if i_file==0:
                    for key in header_temp.keys():
                        if 'BIASM' in key or 'RDN' in key:
                            del header_temp[key]
                    header_master = header_temp
                    
                if imtype=='flat':
                    comment = 'name reduced flat'
                elif imtype=='bias':
                    comment = 'name gain/os-corrected bias frame'

                header_master['{}{}'.format(imtype.upper(), i_file+1)] = (
                    filename.split('/')[-1], '{} {}'.format(comment, i_file+1))
                
                if 'ORIGFILE' in header_temp.keys():
                    header_master['{}OR{}'.format(imtype.upper(), i_file+1)] = (
                        header_temp['ORIGFILE'], 'name original {} {}'
                        .format(imtype, i_file+1))


            # determine the median
            master_median = np.median(master_cube, axis=0)
            
            # add some header keywords to the master flat
            if imtype=='flat':
                sec_temp = get_par(set_bb.flat_norm_sec,tel)
                value_temp = '[{}:{},{}:{}]'.format(sec_temp[0].start+1, sec_temp[0].stop+1,
                                                    sec_temp[1].start+1, sec_temp[1].stop+1) 
                header_master['STATSEC'] = (value_temp,
                                            'pre-defined statistics section [y1:y2,x1:x2]')
                header_master['SECMED'] = (np.median(master_median[sec_temp]),
                                           '[e-] median master flat over STATSEC')
                header_master['SECSTD'] = (np.std(master_median[sec_temp]),
                                           '[e-] sigma (STD) master flat over STATSEC')

                # for full image statistics, discard masked pixels
                mask_ok = (data_mask==0)
                header_master['FLATMED'] = (np.median(master_median[mask_ok]),
                                            '[e-] median master flat')
                header_master['FLATSTD'] = (np.std(master_median[mask_ok]),
                                            '[e-] sigma (STD) master flat')

            elif imtype=='bias':

                # add some header keywords to the master bias
                mean_master, std_master = clipped_stats(master_median, get_median=False)
                header_master['BIASMEAN'] = (mean_master, '[e-] mean master bias')
                header_master['RDNOISE'] = (std_master, '[e-] sigma (STD) master bias')

                # including the means and standard deviations of the master
                # bias in the separate channels
                data_sec_red = get_par(set_bb.data_sec_red,tel)
                nchans = np.shape(data_sec_red)[0]
                mean_chan = np.zeros(nchans)
                std_chan = np.zeros(nchans)

                for i_chan in range(nchans):
                    data_chan = master_median[data_sec_red[i_chan]]
                    mean_chan[i_chan], std_chan[i_chan] = clipped_stats(data_chan, get_median=False)
                for i_chan in range(nchans):
                    header_master['BIASM{}'.format(i_chan+1)] = (
                        mean_chan[i_chan], '[e-] channel {} mean master bias'.format(i_chan+1))
                for i_chan in range(nchans):
                    header_master['RDN{}'.format(i_chan+1)] = (
                        std_chan[i_chan], '[e-] channel {} sigma (STD) master bias'.format(i_chan+1))


            # write to output file
            fits.writeto(fits_master, master_median.astype('float32'), header_master,
                         overwrite=True)

            
    log.info('reading master {}'.format(imtype))
    master_median = read_hdulist(fits_master, ext_data=0)
    if os.path.islink(fits_master):
        master_name = os.readlink(fits_master)
    else:
        master_name = fits_master
    header['M{}-F'.format(imtype.upper())] = (
        master_name.split('/')[-1], 'name of master {} applied'.format(imtype))
    
    if imtype=='flat':
        # divide data by the normalised flat
        # do not consider pixels with zero values or edge pixels
        mask_ok = ((master_median != 0) & (data_mask != get_par(set_zogy.mask_value['edge'],tel)))
        data[mask_ok] /= master_median[mask_ok]
    elif imtype=='bias':
        # subtract from data
        data -= master_median
                
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='master_corr', log=log)

    return data


################################################################################

def mflat_corr(data, header, data_mask, flat_path, date_eve, filt):

    if get_par(set_zogy.timing,tel):
        t = time.time()
        
    fits_mflat = '{}/flat_{}_{}.fits'.format(flat_path, date_eve, filt)
    if not os.path.isfile(unzip(fits_mflat)):

        # prepare master flat from flats in [flat_path]
        flat_list = sorted(glob.glob('{}/*_{}.fits*'.format(flat_path, filt)))

        # initialize cube of flats to be combined
        nflat = np.shape(flat_list)[0]

        # if there are too few bias frames to make tonight's master
        # flat, look for a nearby master flat instead
        if nflat < 3:

            fits_mflat_close = get_closest_biasflat(date_eve, 'flat', filt=filt)

            if fits_mflat_close is not None:

                fits_mflat_close = unzip(fits_mflat_close)
                print ('Warning: too few flats available to produce master flat; instead using\n{}'
                       .format(fits_mflat_close))
                # create symbolic link so future files will automatically
                # use this as the master flat
                os.symlink(fits_mflat_close, fits_mflat)

            else:
                log.error('no alternative master flat found')
                return data
                
        else:
            
            print ('making master flat in filter {}'.format(filt))

            # assuming that flats have the same shape as the input data
            ysize, xsize = np.shape(data)
            flat_cube = np.zeros((nflat, ysize, xsize), dtype='float32')

            # fill the cube
            for i_flat, flat in enumerate(flat_list):
                flat_temp, header_temp = read_hdulist(flat_list[i_flat],
                                                      ext_data=0, ext_header=0)
                # divide by median over the region [set_bb.flat_norm_sec]
                mean, std, median = clipped_stats(flat_temp[get_par(set_bb.flat_norm_sec,tel)])
                print ('flat name: {}, mean: {}, std: {}, median: {}'.format(flat, mean, std, median))
                flat_cube[i_flat] = flat_temp / median

                if i_flat==0:
                    for key in header_temp.keys():
                        if 'BIASM' in key or 'RDN' in key:
                            del header_temp[key]
                    header_mflat = header_temp

                flat_short = flat.split('/')[-1]
                header_mflat['FLAT{}'.format(i_flat+1)] = (
                    flat_short, 'name reduced flat {}'.format(i_flat+1))
                if 'ORIGFILE' in header_temp.keys():
                    flat_orig = header_temp['ORIGFILE']
                    header_mflat['FLATOR{}'.format(i_flat+1)] = (
                        flat_orig, 'name original flat {}'.format(i_flat+1))

            
            # determine the clipped mean
            #flat_mean, flat_median, flat_std = sigma_clipped_stats(flat_cube, axis=0)
            # or simply the median:
            flat_median = np.median(flat_cube, axis=0)

            # add some header keywords to the master flat
            sec_temp = get_par(set_bb.flat_norm_sec,tel)
            value_temp = '[{}:{},{}:{}]'.format(sec_temp[0].start+1, sec_temp[0].stop+1,
                                                sec_temp[1].start+1, sec_temp[1].stop+1) 
            header_mflat['STATSEC'] = (value_temp, 'pre-defined statistics section [y1:y2,x1:x2]')
            header_mflat['SECMED'] = (np.median(flat_median[sec_temp]), '[e-] median master flat over STATSEC')
            header_mflat['SECSTD'] = (np.std(flat_median[sec_temp]),    '[e-] sigma (STD) master flat over STATSEC')

            # for full image statistics, discard masked pixels
            mask_ok = (data_mask==0)
            header_mflat['FLATMED'] = (np.median(flat_median[mask_ok]), '[e-] median master flat')
            header_mflat['FLATSTD'] = (np.std(flat_median[mask_ok]), '[e-] sigma (STD) master flat')
                
            # write to output file
            fits.writeto(fits_mflat, flat_median.astype('float32'), header_mflat,
                         overwrite=True)


    log.info('reading master flat')
    flat_median = read_hdulist(fits_mflat, ext_data=0)
    if os.path.islink(fits_mflat):
        mflat_name = os.readlink(fits_mflat)
    else:
        mflat_name = fits_mflat
    header['MFLAT-F'] = (mflat_name.split('/')[-1], 'name of master flat applied')
       
    # divide data by the normalised flat
    # do not consider pixels with zero values or edge pixels
    mask_ok = ((flat_median != 0) & (data_mask != get_par(set_zogy.mask_value['edge'],tel)))
    data[mask_ok] /= flat_median[mask_ok]
               
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='mflat_corr', log=log)

    return data
    

################################################################################

def mbias_corr(data, header, bias_path, date_eve):

    if get_par(set_zogy.timing,tel):
        t = time.time()
        
    fits_mbias = '{}/bias_{}.fits'.format(bias_path, date_eve)
    
    if not os.path.isfile(unzip(fits_mbias)):

        # prepare master bias from biases in [bias_path]
        bias_list = sorted(glob.glob(bias_path+'/*fits*'))

        # initialize cube of biases to be combined
        nbias = np.shape(bias_list)[0]
        
        # if there are too few bias frames to make tonight's master
        # bias, look for a nearby master bias instead
        if nbias < 5:
            
            fits_mbias_close = get_closest_biasflat(date_eve, 'bias')

            if fits_mbias_close is not None:

                fits_mbias_close = unzip(fits_mbias_close)
                print ('Warning: too few biases available to produce master bias; instead using\n{}'
                       .format(fits_mbias_close))            
                # create symbolic link so future files will automatically
                # use this as the master bias
                os.symlink(fits_mbias_close, fits_mbias)
                
            else:
                log.error('Error: no alternative master bias found')
                return data
            
        else:
            
            print ('making master bias')

            # assuming that biases have the same shape as the input data
            ysize, xsize = np.shape(data)
            bias_cube = np.zeros((nbias, ysize, xsize), dtype='float32')

            # fill the cube
            for i_bias, bias in enumerate(bias_list):
                bias_cube[i_bias], header_temp = read_hdulist(bias_list[i_bias],
                                                              ext_data=0, ext_header=0)
                if i_bias==0:
                    for key in header_temp.keys():
                        if 'BIASM' in key or 'RDN' in key:
                            del header_temp[key]
                    header_mbias = header_temp
                
                bias_short = bias.split('/')[-1]
                header_mbias['BIAS{}'.format(i_bias+1)] = (
                    bias_short, 'name gain/os-corrected bias frame {}'.format(i_bias+1))
                if 'ORIGFILE' in header_temp.keys():
                    bias_orig = header_temp['ORIGFILE']
                    header_mbias['BIASOR{}'.format(i_bias+1)] = (
                        bias_orig, 'name original bias frame {}'.format(i_bias+1))

            
            # determine the clipped mean
            #bias_mean, bias_median, bias_std = sigma_clipped_stats(bias_cube, axis=0)
            # or simply the mean:
            bias_median = np.median(bias_cube, axis=0)

            # add some header keywords to the master bias
            mean_mbias, std_mbias = clipped_stats(bias_median, get_median=False)
            header_mbias['BIASMEAN'] = (mean_mbias, '[e-] mean master bias')
            header_mbias['RDNOISE'] = (std_mbias, '[e-] sigma (STD) master bias')

            # including the means and standard deviations of the master
            # bias in the separate channels
            data_sec_red = get_par(set_bb.data_sec_red,tel)
            nchans = np.shape(data_sec_red)[0]
            mean_chan = np.zeros(nchans)
            std_chan = np.zeros(nchans)

            for i_chan in range(nchans):
                data_chan = bias_median[data_sec_red[i_chan]]
                mean_chan[i_chan], std_chan[i_chan] = clipped_stats(data_chan, get_median=False)
            for i_chan in range(nchans):
                header_mbias['BIASM{}'.format(i_chan+1)] = (
                    mean_chan[i_chan], '[e-] channel {} mean master bias'.format(i_chan+1))
            for i_chan in range(nchans):
                header_mbias['RDN{}'.format(i_chan+1)] = (
                    std_chan[i_chan], '[e-] channel {} sigma (STD) master bias'.format(i_chan+1))
        
            # write to output file
            fits.writeto(fits_mbias, bias_median.astype('float32'), header_mbias,
                         overwrite=True)


    log.info('reading master bias')
    bias_median = read_hdulist(fits_mbias, ext_data=0)

    if os.path.islink(fits_mbias):
        mbias_name = os.readlink(fits_mbias)
    else:
        mbias_name = fits_mbias
    header['MBIAS-F'] = (mbias_name.split('/')[-1], 'name of master bias applied')
    
    # subtract from data
    data -= bias_median
               
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='mbias_corr', log=log)

    return data
    

################################################################################

def get_closest_biasflat (date_eve, file_type, filt=None):

    search_str = '{}/*/*/*/{}/{}'.format(get_par(set_bb.red_dir,tel), file_type,
                                         file_type+'_????????')
    if filt is None:
        search_str += '.fits*'
    else:
        search_str += '_{}.fits*'.format(filt)

    files = glob.glob(search_str)
    nfiles = len(files)

    if nfiles > 0:
    
        # find file that is closest in time to [date_eve]
        mjds = np.array([date2mjd(files[i].split('/')[-1][5:13])
                         for i in range(nfiles)])
        i_close = np.argmin(abs(mjds - date2mjd(date_eve)))
        return files[i_close]

    else:
        return None
    

################################################################################

def date2mjd (date_str, get_jd=False, date_format='%Y%m%d'):
    
    """convert [date_str] in format [date_format] to MJD or JD if [get_jd]
       is set"""

    date = dt.datetime.strptime(date_str, date_format)
    jd = int(date.toordinal()) + 1721424.5
    
    if get_jd:
        return jd
    else:
        return jd - 2400000.5
    

################################################################################

def set_header(header, filename):

    def edit_head (header, key, value=None, comments=None, action='new'):
        # update value
        if value is not None:
            if key in header.keys():
                if header[key] != value:
                    print ('warning: value of existing keyword {} updated from {} to {}'
                           .format(key, header[key], value))
            header[key] = value
        # update comments
        if comments is not None:
            if key in header.keys():
                header.comments[key] = comments
            else:
                print ('warning: keyword {} does not exist: comment is not updated'
                       .format(key))

    edit_head(header, 'BUNIT', value='ADU', comments='Physical unit of array values')
    edit_head(header, 'BSCALE', comments='value = fits_value*BSCALE+BZERO')
    edit_head(header, 'BZERO', comments='value = fits_value*BSCALE+BZERO')
    #edit_head(header, 'CCD-AMP', value='', comments='Amplifier mode of the CCD camera')
    #edit_head(header, 'CCD-SET', value='', comments='CCD settings file')
    edit_head(header, 'XBINNING', value=1, comments='[pix] Binning factor X axis')
    edit_head(header, 'YBINNING', value=1, comments='[pix] Binning factor Y axis')
    edit_head(header, 'ALTITUDE', comments='[deg] Altitude in horizontal coordinates')
    edit_head(header, 'AZIMUTH', comments='[deg] Azimuth in horizontal coordinates')
    edit_head(header, 'HA', comments='[deg] Hour angle')
    edit_head(header, 'RADESYS', value='ICRS', comments='Coordinate reference frame')

    # RA and DEC
    if 'RA' in header.keys() and 'DEC' in header.keys():

        # Right ascension
        if ':' in str(header['RA']):
            # convert sexagesimal to decimal degrees
            ra_deg = Angle(header['RA'], unit=u.hour).degree
        else:
            # convert RA decimal hours to degrees
            ra_deg = header['RA'] * 15.
        edit_head(header, 'RA', value=ra_deg, comments='[deg] Right ascension of image centre')
        edit_head(header, 'RA-REF', comments='Requested right ascension')
        edit_head(header, 'RA-TEL', comments='[deg] Telescope right ascension')

        # Declination
        if ':' in str(header['DEC']):
            # convert sexagesimal to decimal degrees
            dec_deg = Angle(header['DEC'], unit=u.deg).degree
            edit_head(header, 'DEC', value=dec_deg, comments='[deg] Declination of image centre')
        else:
            edit_head(header, 'DEC', comments='[deg] Declination of image centre')
            edit_head(header, 'DEC-REF', comments='Requested declination')
            edit_head(header, 'DEC-TEL', comments='[deg] Telescope declination')
            # for airmass determination below
            dec_deg = header['DEC']

            
    edit_head(header, 'FLIPSTAT', comments='Telescope side of the pier')
    edit_head(header, 'EXPTIME', comments='[s] Requested exposure time')
    if 'ISTRACKI' in header.keys():
        edit_head(header, 'ISTRACKI', bool(header['ISTRACKI']), comments='Telescope is tracking')
    #edit_head(header, 'ACQSTART', value='', comments='Time of PC acquisition request sent to camera')
    #edit_head(header, 'ACQEND', value='', comments='Time of PC registering acquisition completion')
    edit_head(header, 'GPSSTART', comments='GPS timing start of opening shutter')
    edit_head(header, 'GPSEND', comments='GPS timing end of closing shutter')

    exptime_days = header['EXPTIME']/3600./24.
    keys = header.keys()
    if 'GPSSTART' in keys and 'GPSEND' in keys and 'EXPTIME' in keys:
        # replace DATE-OBS with (GPSSTART+GPSEND-EXPTIME)/2
        gps_mjd = Time([header['GPSSTART'], header['GPSEND']], format='isot').mjd
        mjd_obs = (np.sum(gps_mjd)-exptime_days)/2.
        date_obs = Time(mjd_obs, format='mjd').isot
        date_obs = Time(date_obs, format='isot') # change from a string to time class
        edit_head(header, 'DATE-OBS', value=str(date_obs),
                  comments='Date at start = (GPSSTART+GPSEND-EXPTIME)/2')
    else:
        date_obs = Time(header['DATE-OBS'], format='isot')
        edit_head(header, 'DATE-OBS', comments='Date at start')
        mjd_obs = Time(date_obs, format='isot').mjd

    mjd_end = mjd_obs + exptime_days
    date_end = Time(mjd_end, format='mjd').isot
    edit_head(header, 'DATE-END', value=date_end, comments='Date at end = (DATE-OBS+EXPTIME)')
    edit_head(header, 'MJD-OBS', value=mjd_obs, comments='[d] MJD at start (based on DATE-OBS)')
    edit_head(header, 'MJD-END', value=mjd_end, comments='[d] MJD at end (based on DATE-END)')
    lst = date_obs.sidereal_time('mean', longitude=Angle(get_par(set_zogy.obs_long,tel),
                                                         unit=u.deg)).hour * 3600.
    edit_head(header, 'LST', value=lst, comments='[s] LMST at start (based on DATE-OBS)')
    
    utc = (mjd_obs-np.floor(mjd_obs)) * 3600. * 24.
    edit_head(header, 'UTC', value=utc, comments='[s] UTC at start (based on DATE-OBS)')
    edit_head(header, 'TIMESYS', value='UTC', comments='Time system used')
    
    edit_head(header, 'FOCUSPOS', comments='[micron] Focuser position')
    edit_head(header, 'IMAGETYP', comments='Image type')
    edit_head(header, 'OBJECT', comments='Name of object observed')
    edit_head(header, 'FIELD_ID', comments='MeerLICHT/BlackGEM field ID')

    if 'RA' in header.keys() and 'DEC' in header.keys():
        lat = get_par(set_zogy.obs_lat,tel)
        lon = get_par(set_zogy.obs_long,tel)
        height = get_par(set_zogy.obs_height,tel)
        airmass = get_airmass(ra_deg, dec_deg, date_obs, lat, lon, height)
        edit_head(header, 'AIRMASS', value=float(airmass), comments='Airmass (based on RA, DEC, DATE-OBS)')
        
    arcfile = '{}.{}.fits'.format(tel, date_obs)
    edit_head(header, 'ARCFILE', value=arcfile, comments='Archive file name')
    edit_head(header, 'ORIGFILE', value=filename.split('/')[-1], comments='ABOT original file name')

    # filter is incorrectly identified in the header for data taken
    # from 2017-11-19T00:00:00 until 2019-01-13T15:00:00. This is the
    # correct mapping, correct filter = filt_corr[old filter],
    # determined by Paul, Oliver & Danielle (see also Redmine bug
    # #281)
    filt_corr = {'u':'q',
                 'g':'r',
                 'q':'i',
                 'r':'g', 
                 'i':'z',
                 'z':'u'}
    tcorr_mjd = Time(['2017-11-19T00:00:00', '2019-01-13T15:00:00'], format='isot').mjd
    if mjd_obs >= tcorr_mjd[0] and mjd_obs <= tcorr_mjd[1]:
        filt_old = header['FILTER']
        edit_head(header, 'FILTER', value=filt_corr[filt_old], comments='Filter (corrected)')
    else:
        edit_head(header, 'FILTER', comments='Filter')

    if tel=='ML1':
        origin = 'SAAO-Sutherland (K94)'
        telescop = 'MeerLICHT-'+tel[2:]
    if tel[0:2]=='BG':
        origin = 'ESO-LaSilla (809)'
        telescop = 'BlackGEM-'+tel[2:]
        
    edit_head(header, 'ORIGIN', value=origin, comments='Origin of data (MPEC Observatory code)')
    edit_head(header, 'TELESCOP', value=telescop, comments='Telescope ID')
    edit_head(header, 'OBSERVER', comments='Robotic observations software and PC ID')
    edit_head(header, 'ABOTVER', comments='ABOT version')
    
        
    # remove the following keywords:
    keys_2remove = ['FILTWHID', 'FOC-ID', 'EXPOSURE', 'END-OBS', 'FOCUSMIT', 
                   'FOCUSAMT', 'EPOCH', 'OWNERGNM', 'OWNERGID', 'OWNERID',
                   'AZ-REF', 'ALT-REF', 'CCDFULLH', 'CCDFULLW']
    for key in keys_2remove:
        if key in header.keys():
            header.remove(key)

    # put some order in the header
    #keys_ordered = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'BUNIT']
    #for nkey, key in enumerate(keys_ordered):
    #    if nkey==1:
    #        continue
    #    key_prev = keys_ordered[nkey-1]
    #    header.insert(key_prev, (key, header[key], header.comments[key]))

    #data_tmp = np.zeros((header['NAXIS2'], header['NAXIS1']), dtype='int')
    #fits.writeto('header.fits', data_tmp, header, overwrite=True)
    #raise SystemExit
    
    return header

    
################################################################################

def os_corr(data, header):

    """Function that corrects [data] for the overscan signal in the
       vertical and horizontal overscan strips. The definitions of the
       different data/overscan/channel sections are taken from
       [set_blackbox].  The function returns a data array that consists of
       the data sections only, i.e. without the overscan regions. The
       [header] is update in plac.

    """
 
    if get_par(set_zogy.timing,tel):
        t = time.time()

    chan_sec = get_par(set_bb.chan_sec,tel)
    data_sec = get_par(set_bb.data_sec,tel)
    os_sec_hori = get_par(set_bb.os_sec_hori,tel)
    os_sec_vert = get_par(set_bb.os_sec_vert,tel)
    data_sec_red = get_par(set_bb.data_sec_red,tel)
    
    # PMV 2018/08/01: this is a constant used inside the loop
    dcol = 11 # after testing, 21 seems a decent width to use

    # number of data columns and rows in the channel
    ncols = get_par(set_bb.dx,tel) - get_par(set_bb.os_xsize,tel)
    nrows = get_par(set_bb.dy,tel) - get_par(set_bb.os_ysize,tel)

    # initialize output data array (without overscan sections)
    ysize_out = get_par(set_bb.ysize,tel) - get_par(set_bb.ny,tel) * get_par(set_bb.os_ysize,tel)
    xsize_out = get_par(set_bb.xsize,tel) - get_par(set_bb.nx,tel) * get_par(set_bb.os_xsize,tel)
    data_out = np.zeros((ysize_out, xsize_out), dtype='float32')

    # and arrays to calculate average means and stds over all channels
    nchans = np.shape(data_sec)[0]
    mean_vos = np.zeros(nchans)
    std_vos = np.zeros(nchans)

    for i_chan in range(nchans):

        # first subtract the clipped mean (not median!) of the
        # vertical overcan section from the entire channel
        data_vos = data[os_sec_vert[i_chan]]
        mean_vos[i_chan], std_vos[i_chan] = clipped_stats(data_vos, get_median=False)
        #data[chan_sec[i_chan]] -= mean_vos[i_chan]
                
        # determine the running clipped mean of the overscan using all
        # values across [dcol] columns, for [ncols] columns
        data_hos = data[os_sec_hori[i_chan]]
        mean_hos, median_hos, std_hos = sigma_clipped_stats(data_hos, axis=0)
        oscan = [np.mean(mean_hos[max(k-int(dcol/2.),0):min(k+int(dcol/2.)+1,ncols)])
                 for k in range(ncols)]
        # do not use the running mean for the first column
        oscan[0] = mean_hos[0]
        # subtract horizontal overscan 
        data[data_sec[i_chan]] -= np.vstack([oscan]*nrows)
        # broadcast into [data_out]
        data_out[data_sec_red[i_chan]] = data[data_sec[i_chan]] 


    # add headers outside above loop to make header more readable
    for i_chan in range(nchans):
        header['BIASM{}'.format(i_chan+1)] = (
            mean_vos[i_chan], '[e-] channel {} mean vertical overscan'.format(i_chan+1))
    for i_chan in range(nchans):
        header['RDN{}'.format(i_chan+1)] = (
            std_vos[i_chan], '[e-] channel {} sigma (STD) vertical overscan'.format(i_chan+1))
                
    # write the average from both the means and standard deviations
    # determined for each channel to the header
    header['BIASMEAN'] = (np.mean(mean_vos), '[e-] average all channel means vert. overscan')
    header['RDNOISE'] = (np.mean(std_vos), '[e-] average all channel sigmas vert. overscan')
        
    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='os_corr', log=log)

    return data_out


################################################################################

def xtalk_corr (data, crosstalk_file):

    # basically the same as Kerry's function
        
    if get_par(set_zogy.timing,tel):
        t = time.time()

    victim, source, correction = np.loadtxt(crosstalk_file,unpack=True)
    corrected = []
    #data = data[0]
    height,width = get_par(set_bb.dy,tel), get_par(set_bb.dx,tel) # = ccd_sec()
    for k in range(len(victim)):
        if victim[k] < 9:
            j, i = 1, 0
        else:
            j, i = 0, 8
        data[height*j:height*(j+1),width*(int(victim[k])-1-i):width*(int(victim[k])-i)] -= data[height*j:height*(j+1),width*(int(source[k])-1-i):width*(int(source[k])-i)]*correction[k]

    if get_par(set_zogy.timing,tel):
        log_timing_memory (t0=t, label='xtalk_corr', log=log)
        
    return data

    # N.B.: note that the channel numbering here are not the same as that assumed
    # with the gain:
    # 
    # [ 0, 1,  2,  3,  4,  5,  6,  7]
    # [ 8, 9, 10, 11, 12, 13, 14, 15]
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

    
################################################################################

def gain_corr(data, header):

    if get_par(set_zogy.timing,tel):
        t = time.time()

    """Returns [data] corrected for the [gain] defined in [set_bb.gain]
       for the different channels

    """

    gain = get_par(set_bb.gain,tel)
    chan_sec = get_par(set_bb.chan_sec,tel)
    for i_chan in range(np.shape(chan_sec)[0]):
        data[chan_sec[i_chan]] *= gain[i_chan]
        header['GAIN{}'.format(i_chan+1)] = (gain[i_chan], 'gain applied to channel {}'.format(i_chan+1))

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
    if date is None:
        q.put(logger.critical('no [date] provided; exiting'))
        raise SystemExit
    else:
        # date can be any of yyyy/mm/dd, yyyy.mm.dd, yyyymmdd,
        # yyyy-mm-dd or yyyy-mm-ddThh:mm:ss.s; if the latter is
        # provided, make sure to set [date_dir] to the date of the
        # evening before UT midnight
        if 'T' in date:
            if '.' in date:
                date = str(Time(date, format='isot')) # rounds date to microseconds as more digits can't be defined in the format (next line)
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
    
def date_obs_get(header):
    '''Returns image observation date in the correct format.

    Returns the observation date of the image from the header in the correct format for file names.

    :param header: primary header
    :type header: header
    :returns: str -- '(date)_T(time)'
    '''
    date_obs = header['DATE-OBS'] #load date from header
    date_obs_split = re.split('-|:|T',date_obs) #split date into date and time
    return date_obs_split[0]+date_obs_split[1]+date_obs_split[2]+'_'+date_obs_split[3]+date_obs_split[4]+date_obs_split[5]

    
################################################################################

def sort_files(read_path, file_name):

    """Function to sort raw files by type.  Globs all files in read_path
       and to sorts files into bias, flat and science images using the
       IMAGETYP header keyword.  Similar to Kerry's function in
       BGreduce, slightly adapted as sorting by filter is not needed.

    """
       
    all_files = sorted(glob.glob(read_path+'/'+file_name)) #glob all raw files and sort
    bias = [] #list of biases
    flat = [] #list of flats
    science = [] # list of science images
    for i in range(len(all_files)): #loop through raw files

        if '.fz' not in all_files[i]:
            header = read_hdulist(all_files[i], ext_header=0)
        else:
            header = read_hdulist(all_files[i], ext_header=1)

        imgtype = header['IMAGETYP'].lower() #get image type
        
        if 'bias' in imgtype: #add bias files to bias list
            bias.append(all_files[i])
        if 'flat' in imgtype: #add flat files to flat list
            flat.append(all_files[i])
        if 'object' in imgtype: #add science files to science list
            science.append(all_files[i])

    list_temp = [bias, flat, science]
    return [item for sublist in list_temp for item in sublist]


################################################################################

def unzip(imgname, timeout=None):

    """Unzip a gzipped of fpacked file.
       Same [subpipe] function STAP_unzip.
    """

    #lock.acquire()

    if '.gz' in imgname:
        print ('gunzipping {}'.format(imgname))
        subprocess.call(['gunzip',imgname])
        imgname = imgname.replace('.gz','')
    elif '.fz' in imgname:
        print ('funpacking {}'.format(imgname))
        subprocess.call(['funpack','-D',imgname])
        imgname = imgname.replace('.fz','')

    #lock.release()

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

    def warn(self, text):
        '''Function to log at the INFO level.

        Logs messages to log file at the WARN level.'''

        self._log.warn(text)
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

def action(item_list):
    '''Action to take during night mode of pipeline.

    For new events, continues if it is a file. '''

    print ('event!')
    
    #get parameters for list
    event, telescope, mode, read_path = item_list.get(True)
    
    while True:
        try:
            filename = str(event.src_path) #get name of new file
            if 'fits' in filename: #only continue if event is a fits file
                if '_mask' or '_red' not in filename:
                    copying(filename) #check to see if write is finished writing
                    q.put(logger.info('Found new file '+filename))
        except AttributeError: #if event is a file
            filename = event
            q.put(logger.info('Found old file '+filename))
            
        blackbox_reduce (filename, telescope, mode, read_path)


################################################################################

class FileWatcher(FileSystemEventHandler, object):
    '''Monitors directory for new files.

    :param queue: multiprocessing queue for new files
    :type queue: multiprocessing.Queue'''
    
    def __init__(self, queue, telescope, mode, read_path):
        self._queue = queue
        self._telescope = telescope
        self._mode = mode
        self._read_path = read_path
        
    def on_created(self, event):
        '''Action to take for new files.

        :param event: new event found
        :type event: event'''
        self._queue.put([event, self._telescope, self._mode, self._read_path])

        
################################################################################

if __name__ == "__main__":
    
    params = argparse.ArgumentParser(description='User parameters')
    params.add_argument('--telescope', type=str, default='ML1', help='Telescope name (ML1, BG2, BG3 or BG4)')
    params.add_argument('--mode', type=str, default='day', help='Day or night mode of pipeline')
    params.add_argument('--date', type=str, default=None, help='Date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd or yyyy.mm.dd)')
    params.add_argument('--read_path', type=str, default=None, help='Full path to the input raw data directory; if not defined it is determined from [set_blackbox.raw_dir], [telescope] and [date]')
    params.add_argument('--slack', default=True, help='Upload messages for night mode to slack.')
    args = params.parse_args()

    run_blackbox (telescope=args.telescope, mode=args.mode, date=args.date, read_path=args.read_path, slack=args.slack)


