import os
import subprocess
import glob
import argparse
import multiprocessing as mp

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

import numpy as np
import astropy.io.fits as fits
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table, vstack, unique

import ephem
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders


__version__ = '0.7'


# hardcode the settings below, because they are not accessible on the
# Slurm login node through the usual set_zogy and set_blackbox
# settings files
tel = 'ML1'
data_dir = '/idia/projects/meerlicht'
raw_dir = '{}/{}/raw'.format(data_dir, tel)
red_dir = '{}/{}/red'.format(data_dir, tel)
genlog_dir = '{}/{}/log'.format(data_dir, tel)
# folder in which to run/save jobs and logs for the nightly processing
job_dir = '{}/Slurm'.format(genlog_dir)

# MeerLICHT observatory settings
obs_lat = -32.3799
obs_lon = 20.8112
obs_timezone = 'Africa/Johannesburg'


def run_blackbox_slurm (date=None, nthreads=4, runtime='4:00:00'):

    # create general logfile based on date/time
    genlogfile = '{}/{}_{}.log'.format(genlog_dir, tel,
                                       Time.now().strftime('%Y%m%d_%H%M%S'))
    fileHandler = logging.FileHandler(genlogfile, 'a')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel('INFO')
    log.addHandler(fileHandler)
    log.info ('general logfile created: {}'.format(genlogfile))

    # mode is night by default
    mode = 'night'

    # if date is not specified, set it to the date of the last local
    # noon; so just before 12:00 SAST yesterday's date will be used
    # SAST is always 2 hours ahead of UT, so can add 2/24 to make the
    # date change at local noon instead of noon UT.
    date_today = Time(int(Time.now().jd+2/24), format='jd').strftime('%Y%m%d')
    if date is None:
        date = date_today
    else:
        # if it is specified, check if it's today
        date = ''.join(e for e in date if e.isdigit())
        # if not, then run in 'day' mode, i.e. no need to monitor for
        # new files or wait for the end of the night once all images
        # have been processed
        if date != date_today:
            mode = 'day'


    # create list of all fits files already present in [read_path]
    read_path, date_eve = get_path(date, 'read')
    # make sure read_path exists (e.g. if no bias frames were taken,
    # it would not have been created by the transfer script)
    os.makedirs(read_path, exist_ok=True)
    filenames = sorted(glob.glob('{}/{}'.format(read_path, '*fits*')))


    log.info('Slurm-processing in {} mode for evening date: {}'
             .format(mode, date_eve))
    log.info('{} files already present in {}'.format(len(filenames), read_path))


    # prepare folder in which to save and run jobs and logs
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])
    jobnight = '{}/{}'.format(job_dir, date_dir)
    os.makedirs (jobnight, exist_ok=True)


    # set date_begin of reservation
    if mode == 'night':
        date_begin = '{}T{}'.format(date_dir.replace('/','-'), '18:00:00')
    else:
        # in day mode, do not make use of the node reservation, but
        # submit as a general Slurm job
        date_begin = None


    # determine time of next sunrise
    obs = ephem.Observer()
    # MeerLICHT longitude/latitude is hardcoded because there is no
    # access to the zogy settings file and this function is expected
    # to run for the MeerLICHT telescope only
    obs.lat = str(obs_lat)
    obs.lon = str(obs_lon)
    sunrise = obs.next_rising(ephem.Sun())


    # create queue for submitting jobs
    queue = mp.Queue()


    # add files that are already present in the read_path
    # directory to the night queue, to reduce these first
    for filename in filenames:
        queue.put(filename)


    # create and setup observer, but do not start just yet
    observer = PollingObserver()
    observer.schedule(FileWatcher(queue), read_path, recursive=False)


    # start monitoring [read_path] for incoming files
    observer.start()


    # keep monitoring queue - which is being filled with new files
    # detected by watchdog - as long as it is nighttime or the
    # queue is not empty yet
    jobnames = []
    #while (ephem.now()-sunrise < 10*ephem.minute or not queue.empty() or
    while (ephem.now()-sunrise < 0 or not queue.empty() or
           # in night mode, also wait until all jobs are finished
           (mode=='night' and len(list_active_jobs(jobnames)) > 0)):

        if queue.empty():
            if mode == 'night':
                time.sleep(10)
            else:
                # in day mode, no additional files should be coming
                # in, so can break out of while loop
                break

        else:
            filename = get_file (queue)
            if filename is not None:

                # Python command to execute
                python_cmdstr = (
                    'python /Software/BlackBOX/blackbox.py --img_reduce True '
                    '--cat_extract True --trans_extract True --keep_tmp False '
                    '--force_reproc_new False --image {}'.format(filename))

                log.info ('python command string to execute: {}'
                          .format(python_cmdstr))

                # process it through a SLURM batch job
                jobname = filename.split('/')[-1].split('.fits')[0]
                slurm_process (python_cmdstr, nthreads, runtime, jobname,
                               jobnight, date_begin=date_begin)

                # append filename to list of files processed
                jobnames.append(jobname)




    log.info ('night has finished and queue is empty')


    # watchdog can be stopped
    observer.stop() #stop observer
    observer.join() #join observer


    # now waiting until no more running jobs left setting maximum wait
    # time equal to twice the input [runtime]
    log.info ('checking if there are any running/pending jobs left')
    wait_max = 2 * np.sum(np.array(runtime.split(':')).astype(int)
                          *np.array([3600, 60, 1]))
    nsec_wait = wait4jobs2finish (jobnames, wait_max=wait_max)
    log.info ('waited for {:.2f} hours for all individual jobs to finish'
              .format(nsec_wait/3600))


    # create master frames
    python_cmdstr_master = ('python /Software/BlackBOX/blackbox.py '
                            '--master_date {}'.format(date_eve))
    # process masters through a SLURM batch job; could split this up
    # into several jobs, one for the master bias and one for each of
    # the master flats in a specific filter, but even with a single
    # task this takes only about 10-15 minutes
    jobname_masters = 'masters_{}'.format(date_eve)
    slurm_process (python_cmdstr_master, nthreads=4, runtime='1:00:00',
                   jobname=jobname_masters, jobnight=jobnight,
                   date_begin=date_begin)
    jobnames.append(jobname_masters)


    # wait for masters to finish; not really needed for the remaining
    # tasks - preparing and sending the night report and adding last
    # night's fits header keys to the big header tables
    nsec_wait = wait4jobs2finish ([jobname_masters], wait_max=3600)
    log.info ('waited for {:.0f}s for masters to finish'.format(nsec_wait))


    # create night report and weather screenshot; this needs to be done
    # inside the container because firefox is not available outside;
    # this requires create_obslog to be runable from the command line,
    # which can be done through the -c option:
    if mode == 'night':
        screenshot = True
    else:
        # in day mode, do not make screenshot, since it's an older
        # date that is being processed
        screenshot = False


    python_cmdstr = ('python -c \"import set_blackbox as set_bb; '
                     'from blackbox import create_obslog; '
                     'create_obslog (\'{}\', email=True, '
                     'tel=\'{}\', weather_screenshot={})\"'
                     .format(date, tel, screenshot))

    jobname = 'obslog_{}'.format(date_eve)
    slurm_process (python_cmdstr, nthreads=1, runtime='0:10:00',
                   jobname=jobname, jobnight=jobnight, date_begin=date_begin)
    jobnames.append(jobname)


    # add number of header keywords of newly reduced images to big
    # header fits tables
    try:

        # loop different catalogs
        for cat_type in ['cat', 'trans', 'sso']:

            # add night's headers to header file
            fits_header = ('{}/Headers/{}_headers_{}.fits'
                           .format(data_dir, tel, cat_type))
            path_full = '{}/{}'.format(red_dir, date_dir)
            search_str = '_{}.fits'.format(cat_type)
            end_str = ''


            # cmd
            python_cmdstr = ('python -c \"import set_blackbox as set_bb; '
                             'from blackbox import add_headkeys; '
                             'add_headkeys (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\')\"'
                             .format(path_full, fits_header,
                                     search_str, end_str, tel))

            jobname = 'add_headkeys_{}_{}'.format(cat_type, date_eve)
            slurm_process (python_cmdstr, nthreads=1, runtime='0:30:00',
                           jobname=jobname, jobnight=jobnight, date_begin=date_begin)
            jobnames.append(jobname)


    except Exception as e:
        log.exception('exception occurred during adding header keys: {}'
                      .format(e))


    # now that night has finished, collect individual logfiles
    # produced by Slurm and append them to the logging for this night
    try:
        for jobname in jobnames:
            logfile_slurm = '{}/{}.err'.format(jobnight, jobname)
            if os.path.isfile(logfile_slurm):
                with open(logfile_slurm, 'r') as f:
                    log.info (f.read())
    except Exception as e:
        log.exception('exception occurred during appending of individual '
                      'logfiles to the general logfile: {}'.format(e))


    return


################################################################################

def wait4jobs2finish (jobnames, wait_max=3600):

    # wait for a bit to make sure any recently submitted jobs have
    # made it to the queue
    t0 = time.time()
    time.sleep(20)

    jobnames_run = jobnames.copy()
    while time.time()-t0 < wait_max:

        # only list jobs that are running or pending
        jobnames_run = list_active_jobs (jobnames_run)
        njobs = len(jobnames_run)

        # if no more running or pending jobs, break
        if njobs==0:
            break
        else:
            log.info ('{} job(s) still running or pending'.format(njobs))

        # wait for a while
        time.sleep(300)

    else:
        log.warning ('maximum wait time of {}s reached'.format(wait_max))


    return time.time()-t0


################################################################################

def list_active_jobs (jobnames, states=['RUNNING', 'PENDING']):

    jobnames_out = []
    for jobname in jobnames:
        state = get_job_state(jobname)
        #log.info ('jobname: {}, state: {}'.format(jobname, state))
        if state in states:
            jobnames_out.append(jobname)

    return jobnames_out


################################################################################

def get_job_state (jobname):

    cmd = ('/opt/slurm/bin/sacct -S now-12hours -E now -o State -n -X --name {} '
           '| tail -1'.format(jobname))
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode('UTF-8').replace('\n','').strip()


################################################################################

def slurm_process (python_cmdstr, nthreads, runtime, jobname, jobnight,
                   date_begin=None, account='b19-meerlicht-ag',
                   reservation='meerlicht'):

    try:

        # create SLURM batch job in date_eve subfolder of nightjobs
        # folder with name based on input jobname
        jobfile = '{}/{}.sh'.format(jobnight, jobname)
        with open(jobfile, 'w') as f:

            f.write ('#!/bin/bash\n')
            f.write ('#SBATCH --nodes=1\n')
            f.write ('#SBATCH --ntasks-per-node=1\n')
            f.write ('#SBATCH --cpus-per-task={}\n'.format(nthreads))
            f.write ('#SBATCH --time={}\n'.format(runtime))
            f.write ('#SBATCH --mem-per-cpu=7GB\n')
            f.write ('#SBATCH --job-name={}\n'.format(jobname))
            f.write ('\n')
            f.write ('#SBATCH --account={}\n'.format(account))

            if False:
                # temporarily using cephfs reservation
                reservation='cephfs'
                f.write ('#SBATCH --reservation={}\n'.format(reservation))

            else:
                if date_begin is not None:
                    f.write ('#SBATCH --begin={}\n'.format(date_begin))
                    f.write ('#SBATCH --reservation={}\n'.format(reservation))
                    #f.write ('#SBATCH --nodelist={}\n'.format(nodelist))
                else:
                    f.write ('#SBATCH --partition=Main\n')

            f.write ('\n')
            f.write ('#SBATCH --open-mode=append\n')
            f.write ('#SBATCH --output={}/{}.out\n'.format(jobnight, jobname))
            f.write ('#SBATCH --error={}/{}.err\n'.format(jobnight, jobname))
            f.write ('#SBATCH --mail-user=paul.vreeswijk@blackgem.org\n')
            f.write ('#SBATCH --mail-type=FAIL,TIME_LIMIT\n')
            f.write ('\n')
            f.write ('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
            f.write ('\n')
            f.write ('echo "date              = $(date)"\n')
            f.write ('echo "hostname          = $(hostname -s)"\n')
            f.write ('echo "working directory = $(pwd)"\n')
            f.write ('\n')
            f.write ('echo "#task allocated   = $SLURM_NTASKS"\n')
            f.write ('echo "#cores/task       = $SLURM_CPUS_PER_TASK"\n')
            f.write ('\n')
            f.write ('/software/common/singularity/3.9.1/bin/singularity exec '
                     '--bind /idia/projects/meerlicht '
                     '--env MLBG_CALDIR=/idia/projects/meerlicht/CalFiles '
                     '/idia/projects/meerlicht/Containers/ML_latest.sif {}\n'
                     .format(python_cmdstr))
            f.write ('\n')



        # make batch script executable
        cmd = ['chmod', '+x', jobfile]
        subprocess.run(cmd)


        # submit batch script
        cmd = ['/opt/slurm/bin/sbatch', jobfile]
        result = subprocess.run(cmd, capture_output=True)
        log.info ('{}, jobfile: {}'.format(result.stdout.decode('UTF-8')
                                           .replace('\n',''), jobfile))
        #log.info ('{}, jobfile: {}'.format(result, jobfile))


        if False:
            # delete batch job
            log.info ('removing jobfile {}'.format(jobfile))
            os.remove(jobfile)


    except Exception as e:

        log.exception ('exception was raised in [slurm_process]: {}'
                       .format(e))


    return


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
            # rounds date to microseconds as more digits cannot be
            # defined in the format (next line)
            date = str(Time(date, format='isot'))
            date_format = '%Y-%m-%dT%H:%M:%S.%f'
            high_noon = 'T12:00:00.0'
        else:
            date_format = '%Y-%m-%dT%H:%M:%S'
            high_noon = 'T12:00:00'

        date_ut = dt.datetime.strptime(date, date_format).replace(tzinfo=gettz('UTC'))
        date_noon = date.split('T')[0]+high_noon
        date_local_noon = dt.datetime.strptime(date_noon, date_format).replace(
            tzinfo=gettz(obs_timezone))
        if date_ut < date_local_noon:
            # subtract day from date_only
            date = (date_ut - dt.timedelta(1)).strftime('%Y-%m-%d')
        else:
            date = date_ut.strftime('%Y-%m-%d')

    # this [date_eve] in format yyyymmdd is also returned
    date_eve = ''.join(e for e in date if e.isdigit())
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])


    if dir_type == 'read':
        root_dir = raw_dir
    elif dir_type == 'write':
        root_dir = red_dir
    else:
        log.error ('[dir_type] not one of "read" or "write"')

    path = '{}/{}'.format(root_dir, date_dir)
    if '//' in path:
        log.info ('replacing double slash in path name: {}'.format(path))
        path = path.replace('//','/')

    return path, date_eve


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
        time.sleep(1)
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

def read_hdulist (fits_file, get_data=True, get_header=False,
                  ext_name_indices=None, dtype=None, columns=None,
                  memmap=True):

    """Function to read the data (if [get_data] is True) and/or header (if
    [get_header] is True) of the input [fits_file].  The fits file can
    be an image or binary table, and can be compressed (with the
    compressions that astropy.io can handle, such as .gz and .fz
    files). If [ext_name_indices] is defined, which can be an integer,
    a string matching the extension's keyword EXTNAME or a list or
    numpy array of integers, those extensions are retrieved.

    """

    if os.path.exists(fits_file):
        fits_file_read = fits_file

    else:
        # if fits_file does not exist, look for compressed versions or
        # files without the .fz or .gz extension
        if os.path.exists('{}.fz'.format(fits_file)):
            fits_file_read = '{}.fz'.format(fits_file)
        elif os.path.exists(fits_file.replace('.fz','')):
            fits_file_read = fits_file.replace('.fz','')
        elif os.path.exists('{}.gz'.format(fits_file)):
            fits_file_read = '{}.gz'.format(fits_file)
        elif os.path.exists(fits_file.replace('.gz','')):
            fits_file_read = fits_file.replace('.gz','')
        else:
            raise FileNotFoundError ('file not found: {}'.format(fits_file))


    # open fits file into hdulist
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


    if columns is not None:
        # only return defined columns
        return [data[col] for col in columns if col in data.dtype.names]
    else:
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
                return


################################################################################

def main ():

    """Wrapper allowing [run_blackbox_slurm] to be run from the command line"""

    parser = argparse.ArgumentParser(description='Run BlackBOX on ilifu Slurm '
                                     'cluster')
    parser.add_argument('--date', type=str, default=None,
                        help='date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd '
                        'or yyyy.mm.dd); default=today (date change is at local noon)')
    parser.add_argument('--nthreads', type=int, default=4,
                        help='number of threads/CPUs to use per image; default=4')
    parser.add_argument('--runtime', type=str, default='4:00:00',
                        help='runtime requested per image; default=4:00:00')
    args = parser.parse_args()


    run_blackbox_slurm (args.date,
                        nthreads=args.nthreads,
                        runtime=args.runtime)


################################################################################

if __name__ == "__main__":
    main()
