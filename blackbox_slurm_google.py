import os
import subprocess
import glob
import argparse
import queue
import itertools

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
import astropy.units as u

import ephem

# to send email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

# google
from concurrent.futures import TimeoutError
from google.cloud import pubsub_v1, storage
project_id = 'blackgem'
subscription_id = 'monitor-blackgem-raw-sub'


__version__ = '0.6'


################################################################################

# hardcode the settings below, because they are not accessible on the
# Slurm login node through the usual set_zogy and set_blackbox
# settings files; at least not at ilifu, at Google this module could
# be run in a singularity container so that these settings are
# available
raw_dir = {}
red_dir = {}
tmp_dir = {}

#tels_running = ['BG2', 'BG3', 'BG4']
tels_running = ['BG2', 'BG4']
for tel in tels_running:
    raw_dir[tel] = 'gs://blackgem-raw/{}'.format(tel)
    red_dir[tel] = 'gs://blackgem-red/{}'.format(tel)
    tmp_dir[tel] = '/tmp/{}'.format(tel)

# folder in which to run/save jobs and logs for the nightly processing
log_dir = '{}/RunBlackBOX/log'.format(os.environ['HOME'])
os.makedirs(log_dir, exist_ok=True)
job_dir = '{}/Slurm'.format(log_dir)
os.makedirs(job_dir, exist_ok=True)

# BG observatory and time zone settings
obs_lat = -29.2575
obs_lon = -70.7380
obs_height = 2383
obs_timezone = 'America/Santiago'

# Email settings
sender = 'Paul Vreeswijk <paul.vreeswijk@blackgem.org>'
# comma-separated email addresses of recipients
recipients = 'bg-nightreports@blackgem.org'
#recipients = 'paul.vreeswijk@blackgem.org'
reply_to = 'paul.vreeswijk@blackgem.org'
smtp_server = 'smtp-relay.gmail.com'
port = 465
use_SSL = True

# home and calibration folder on Google slurm login node account
home_dir = '/home/sa_105685508700717199458'
cal_dir = os.path.join(home_dir, 'CalFiles')
mlbg_fieldIDs = '{}/MLBG_FieldIDs_Feb2022_nGaia.fits'.format(cal_dir)


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

def run_blackbox_slurm (date=None, telescope=None, mode='night',
                        runtime='6:00:00'):

    # create general logfile based on date/time
    genlogfile = '{}/BG_{}.log'.format(log_dir,
                                       Time.now().strftime('%Y%m%d_%H%M%S'))
    fileHandler = logging.FileHandler(genlogfile, 'a')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel('INFO')
    log.addHandler(fileHandler)
    log.info ('general logfile created: {}'.format(genlogfile))


    # read field ID table including number of expected Gaia sources
    table_grid = Table.read(mlbg_fieldIDs, memmap=True)

    # create ngaia dictionary with keys int(field_id) and
    # ngaia as values
    ngaia_dict = {}
    for i, field_id in enumerate(table_grid['field_id']):
        ngaia_dict[field_id] = table_grid['ngaia'][i]


    # if date is not specified, set it to the date of the last local
    # noon; so until noon yesterday's date will be used.  Chilean
    # local time is always 3 or 4 hours behind UT, depending on DST;
    # subtract 3.5/24 to make the date change at 11:30 or 12:30
    date_today = Time(int(Time.now().jd-3.5/24), format='jd').strftime('%Y%m%d')
    if date is None:
        date = date_today
    else:
        # if it is specified, check if it's today
        date = ''.join(e for e in date if e.isdigit())
        # if not, then run in 'day' mode, i.e. no need to monitor for
        # new files or wait for the end of the night once all images
        # have been processed
        if date != date_today:
            log.info ('input date specified {} is different from today\'s date '
                      '{}; forcing processing to run in day mode'
                      .format(date, date_today))
            mode = 'day'



    # create list of all fits files already present in [read_path]
    filenames = []
    # jobnight is dictionary indicating the telescope-dependent folder
    # where to save the Slurm job scripts and logs
    jobnight = {}
    for ntel, tel in enumerate(tels_running):
        # input [telescope] could be single telescope or 'BG' meaning all three
        if telescope in tel:
            read_path, date_eve = get_path(date, 'read', tel)
            filenames_tel = sorted(list_files(read_path, search_str='fits'))
            filenames.append(filenames_tel)

            if ntel==0:
                log.info('Slurm-processing in {} mode for evening date: {}'
                         .format(mode, date_eve))

            log.info('{} files already present in {}'.format(len(filenames_tel),
                                                             read_path))

            # prepare folder in which to save and run jobs and logs
            date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6],
                                         date_eve[6:8])
            jobnight[tel] = '{}/{}/{}'.format(job_dir, tel, date_dir)
            os.makedirs (jobnight[tel], exist_ok=True)



    # filenames contains one list for each telescope; flatten these lists
    filenames = list(itertools.chain.from_iterable(filenames))


    # determine time of next sunrise
    obs = ephem.Observer()
    # BlackGEM telescope longitude/latitude is hardcoded because there
    # is no access to the zogy settings file and this function is
    # expected to run for the BlackGEM telescopes only
    obs.lat = str(obs_lat)
    obs.lon = str(obs_lon)
    obs.elevation = obs_height
    # correct apparent horizon for observer elevation, which is
    # not taken into account in ephem
    obs = adjust_horizon(obs, obs_height)
    sunrise = obs.next_rising(ephem.Sun())


    # create queue for submitting jobs
    q = queue.Queue()


    # add files that are already present in the read_path
    # directory to the night queue, to reduce these first
    for filename in filenames:
        if 'fits' in filename:
            q.put(filename)



    # callback function for Pub/Sub subscriber defined below
    def callback(message):

        # Acknowledge the message
        message.ack()

        # extract the filename from the message attributes
        #filename = message.data.decode('utf-8')
        msg_atts = message.attributes
        log.info ('message attributes: {}'.format(msg_atts))
        filename = 'gs://{}/{}'.format(msg_atts['bucketId'],
                                       msg_atts['objectId'])

        if 'fits' in filename:
            log.info ('new file detected: {}; adding it to the queue'
                      .format(filename))
            # add filename to queue
            q.put(filename)



    if mode == 'night':

        # in night mode, set up Pub/Sub subscriber to monitor new
        # images saved to the bucket gs://blackgem-raw
        subscriber = pubsub_v1.SubscriberClient()
        # The `subscription_path` method creates a fully qualified
        # identifier in the form
        # `projects/{project_id}/subscriptions/{subscription_id}`
        subscription_path = subscriber.subscription_path(project_id,
                                                         subscription_id)
        log.info ('project_id: {}'.format(project_id))
        log.info ('subscription_path: {}'.format(subscription_path))


        # start streaming pull, listening for messages
        streaming_pull_future = subscriber.subscribe(subscription_path,
                                                     callback=callback)
        log.info ('listening for messages on {}'.format(subscription_path))



    # keep monitoring queue - which is being filled with new files
    # detected by subscriber - as long as it is nighttime or the queue
    # is not empty yet
    jobnames = []
    #while (ephem.now()-sunrise < 0*ephem.minute or not q.empty() or
    while (ephem.now()-sunrise < 15*ephem.minute or not q.empty() or
           # in night mode, also wait until all jobs are finished, to
           # keep the queue open for potential new files; the file
           # transfer into google cloud will be sluggish with 3
           # telescopes operating, so not clear when the last file of
           # the night will actually be arriving
           (mode=='night' and len(list_active_jobs(jobnames)) > 0)):

        if q.empty():

            if mode == 'night':
                time.sleep(10)
            else:
                # in day mode, no additional files should be coming
                # in, so can break out of while loop
                break

        else:

            filename = q.get()

            log.info ('filename: {}'.format(filename))

            process = False
            if filename:

                # infer telescope from first letters of filename
                tel = filename.split('/')[-1][0:3]

                # only process if tel in tels_running and it is
                # consistent with input telescope (which could be
                # 'BG')
                if tel in tels_running and telescope in tel:
                    process = True


            if process:

                # Python command to execute
                python_cmdstr = (
                    'python /Software/BlackBOX/blackbox.py --telescope {} '
                    '--img_reduce True --cat_extract True --trans_extract True '
                    '--force_reproc_new True --image {}'
                    .format(tel, filename))

                log.info ('python command string to execute: {}'
                          .format(python_cmdstr))


                # use different partitions for bias/flats images
                if np.any([s in filename.lower()
                           for s in ['bias','flat','dark']]):
                    partition = 'pc1gb8'

                else:
                    # for object images, use different partitions for
                    # fields with low and high number of expected gaia
                    # sources; default partition:
                    partition = 'pc2gb16'


                    # if field contains many Gaia sources, use
                    # different partition
                    try:
                        # extract field ID from header
                        header = read_hdulist(filename, get_data=False,
                                              get_header=True)
                        if 'OBJECT' in header:
                            field_id = int(header['OBJECT'])
                            # set partition depending on ngaia
                            ngaia = ngaia_dict[field_id]
                            if ngaia > 2e5:
                                # use different partition
                                partition = 'pc4gb32'

                            log.info ('estimated # Gaia sources in field ({}): '
                                      '{}; using partition {} for {}'
                                      .format(field_id, ngaia, partition,
                                              filename))
                    except Exception as e:
                        log.warning ('exception occurred when inferring ngaia '
                                     'for {}; using default partition ({}): {}'
                                     .format(filename, partition, e))



                # process it through a SLURM batch job
                jobname = filename.split('/')[-1].split('.fits')[0]
                slurm_process (python_cmdstr, partition, runtime, jobname,
                               jobnight[tel])


                # append filename to list of files processed
                jobnames.append(jobname)


                # sleep for a bit to avoid submitting jobs at the same
                # time
                time.sleep(1)



    log.info ('night has finished and queue is empty')


    # subscriber can be stopped
    if mode == 'night':
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.


    # now waiting until no more running jobs left setting maximum wait
    # time equal to twice the input [runtime]
    log.info ('checking if there are any running/pending jobs left')
    wait_max = 2 * np.sum(np.array(runtime.split(':')).astype(int)
                          *np.array([3600, 60, 1]))
    nsec_wait = wait4jobs2finish (jobnames, wait_max=wait_max)
    log.info ('waited for {:.2f} hours for all individual jobs to finish'
              .format(nsec_wait/3600))


    # create master frames
    jobname_masters = {}
    for tel in tels_running:
        if telescope in tel:
            python_cmdstr_master = ('python /Software/BlackBOX/blackbox.py '
                                    '--telescope {} --master_date {}'
                                    .format(tel, date_eve))
            # process masters through a SLURM batch job; could split
            # this up into several jobs, one for the master bias and
            # one for each of the master flats in a specific filter,
            # but even with a single task this takes only about 10-15
            # minutes
            jobname_masters[tel] = '{}_masters_{}'.format(tel, date_eve)

            partition = 'pc2gb32'
            slurm_process (python_cmdstr_master, partition, runtime='1:00:00',
                           jobname=jobname_masters[tel], jobnight=jobnight[tel])
            jobnames.append(jobname_masters[tel])




    # add number of header keywords of newly reduced images to header
    # fits tables; function [add_headkeys] is taking care of writing a
    # temporary updated table before moving it to the bucket
    for tel in tels_running:
        if telescope in tel:

            try:

                # loop different catalogs
                for cat_type in ['cat', 'trans', 'sso', 'bias', 'flat']:

                    # add night's headers to header file
                    fits_header = ('gs://blackgem-hdrtables/{}/{}_headers_{}.fits'
                                   .format(tel, tel, cat_type))
                    path_full = '{}/{}'.format(red_dir[tel], date_dir)

                    if cat_type in ['bias', 'flat']:
                        search_str = '/{}/{}_20'.format(cat_type, tel)
                        end_str = '.fits.fz'
                    elif cat_type == 'cat':
                        search_str = '_red_{}.fits'.format(cat_type)
                        end_str = ''
                    else:
                        search_str = '_{}.fits'.format(cat_type)
                        end_str = ''



                    # cmd
                    python_cmdstr = ('python -c \"import set_blackbox as set_bb;'
                                     ' from blackbox import add_headkeys; '
                                     'add_headkeys (\'{}\', \'{}\', \'{}\', '
                                     '\'{}\', \'{}\')\"'
                                     .format(path_full, fits_header, search_str,
                                             end_str, tel))

                    jobname = '{}_add_headkeys_{}_{}'.format(tel, cat_type,
                                                             date_eve)
                    partition = 'pc1gb8'
                    slurm_process (python_cmdstr, partition, runtime='0:30:00',
                                   jobname=jobname, jobnight=jobnight[tel])
                    jobnames.append(jobname)


            except Exception as e:
                log.exception('exception occurred while adding {} header keys '
                              'for {}: {}'.format(cat_type, tel, e))




    # for La Silla, can always make a screenshot of a particular
    # night (as opposed to Sutherland), so switch it on
    screenshot = True

    for tel in tels_running:
        if telescope in tel:

            try:

                if False:

                    # execute create_obslog on a compute node
                    python_cmdstr = ('python -c \"import set_blackbox as set_bb; '
                                     'from blackbox import create_obslog; '
                                     'create_obslog (\'{}\', email=True, '
                                     'tel=\'{}\', weather_screenshot={})\"'
                                     .format(date, tel, screenshot))

                    jobname = '{}_obslog_{}'.format(tel, date_eve)
                    partition = 'pc1gb8'
                    slurm_process (python_cmdstr, partition, runtime='0:10:00',
                                   jobname=jobname, jobnight=jobnight[tel])
                    jobnames.append(jobname)

                else:

                    # currently compute nodes cannot send emails, so
                    # execute create_obslog on the login node directly
                    # using adapted version of [create_obslog]
                    create_obslog (date, email=True, tel=tel,
                                   weather_screenshot=screenshot)


            except Exception as e:
                log.exception('exception occurred while creating {} obslog '
                              'for date {}: {}'.format(tel, date, e))




    # now that night has finished, collect individual logfiles
    # produced by Slurm and append them to the logging for this night
    if False:
        for tel in tels_running:
            if telescope in tel:

                try:
                    for jobname in jobnames:
                        logfile_slurm = '{}/{}.err'.format(jobnight[tel], jobname)
                        if os.path.isfile(logfile_slurm):
                            with open(logfile_slurm, 'r') as f:
                                log.info (f.read())

                except Exception as e:
                    log.exception('exception occurred during appending of individual '
                                  '{} logfiles to the general logfile: {}'
                                  .format(tel, e))


    log.info ('all done')

    return


################################################################################

def wait4jobs2finish (jobnames, wait_max=3600):

    # wait for a bit to make sure any recently submitted jobs have
    # made it to the queue
    t0 = time.time()
    time.sleep(20)

    jobnames_run = jobnames.copy()
    while time.time()-t0 < wait_max:

        # only list jobs that are running, pending or node_fail
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

def list_active_jobs (jobnames, states=['RUNNING', 'PENDING', 'NODE_FAIL']):

    jobnames_out = []
    for jobname in jobnames:
        state = get_job_state(jobname)
        #log.info ('jobname: {}, state: {}'.format(jobname, state))
        if state in states:
            jobnames_out.append(jobname)

    return jobnames_out


################################################################################

def get_job_state (jobname):

    cmd = ('/usr/local/bin/sacct -S now-12hours -E now -o State -n -X --name {} '
           '| tail -1'.format(jobname))
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode('UTF-8').replace('\n','').strip()


################################################################################

def slurm_process (python_cmdstr, partition, runtime, jobname, jobnight):

    try:

        # number of CPUs and RAM memory associated to different partitions
        ncpu_ram = {'c1gb8':  [1, '7500MB'],
                    'c2gb16': [2, '15500MB'],
                    'c2gb32': [2, '31400MB'],
                    'c4gb32': [4, '31400MB'],
                    'pc1gb8':  [1, '7500MB'],
                    'pc2gb16': [2, '15500MB'],
                    'pc2gb32': [2, '31400MB'],
                    'pc4gb32': [4, '31400MB']}

        ncpu, ram = ncpu_ram[partition]

        # create SLURM batch job in date_eve subfolder of nightjobs
        # folder with name based on input jobname
        jobfile = '{}/{}.sh'.format(jobnight, jobname)
        with open(jobfile, 'w') as f:

            f.write ('#!/bin/bash\n')
            f.write ('#SBATCH --nodes=1\n')
            f.write ('#SBATCH --ntasks-per-node=1\n')
            f.write ('#SBATCH --cpus-per-task={}\n'.format(ncpu))
            f.write ('#SBATCH --time={}\n'.format(runtime))
            f.write ('#SBATCH --mem={}\n'.format(ram))
            f.write ('#SBATCH --job-name={}\n'.format(jobname))
            f.write ('\n')
            f.write ('#SBATCH --partition={}\n'.format(partition))
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
            f.write ('/opt/apps/singularity/3.11.0/bin/singularity exec '
                     #'--env PYTHONPATH="/home/sa_105685508700717199458/BBtest:\$PYTHONPATH" '
                     '--env MLBG_CALDIR={} {}/Containers/MLBG_latest.sif {}\n'
                     .format(cal_dir, home_dir, python_cmdstr))
            f.write ('\n')


        # make batch script executable
        cmd = ['chmod', '+x', jobfile]
        subprocess.run(cmd)


        # submit batch script; turn off for now
        cmd = ['/usr/local/bin/sbatch', jobfile]
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
    red_path = red_dir[tel]
    full_path = '{}/{}'.format(red_path, date_dir)

    log.info ('full_path: {}'.format(full_path))


    # collect biases, darks, flats and science frames in different lists
    #bias_list = glob.glob('{}/bias/{}*.fits*'.format(full_path, tel))
    bias_list = list_files('{}/bias/{}'.format(full_path,tel),search_str='.fits')
    #dark_list = glob.glob('{}/dark/{}*.fits*'.format(full_path, tel))
    dark_list = list_files('{}/dark/{}'.format(full_path,tel),search_str='.fits')
    #flat_list = glob.glob('{}/flat/{}*.fits*'.format(full_path, tel))
    flat_list = list_files('{}/flat/{}'.format(full_path,tel),search_str='.fits')
    #object_list = glob.glob('{}/{}*_red.fits*'.format(full_path, tel))
    object_list = list_files('{}/{}'.format(full_path,tel),search_str='_red.fits')

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
    raw_path = raw_dir[tel]
    #raw_list = glob.glob('{}/{}/*.fits*'.format(raw_path, date_dir))
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
    rows = []
    for filename in filenames:

        # read file header
        header = read_hdulist (filename, get_data=False, get_header=True)

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
    tmp_path = tmp_dir[tel]
    # just in case it does not exist yet, create it
    make_dir (tmp_path)
    obslog_tmp = '{}/{}_{}_obslog.txt'.format(tmp_path, tel, date_eve)

    if len(rows)==0:
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
        webpage = 'https://archive.eso.org/asm/ambient-server?site=lasilla'
        #webpage = ('https://www.eso.org/asm/ui/publicLog?name=LaSilla&startDate='
        #           '{}'.format(date_eve))
        width = 1500
        height = 1150


    # define png destination
    png_dest = '{}/{}/{}'.format(red_path, date_dir, png_tmp.split('/')[-1])


    if weather_screenshot:
        try:

            cmd = ['/usr/local/bin/wkhtmltoimage', '--quiet', '--quality', '80',
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

    #cat_list = glob.glob('{}/{}*_red_cat.fits'.format(full_path, tel))
    cat_list = list_files('{}/{}'.format(full_path, tel),
                          end_str='_red_cat_hdr.fits')
    body += ('# full-source cats: {} ({} red-flagged)\n'.format(
        len(cat_list), count_redflags(cat_list)))

    #trans_list = glob.glob('{}/{}*_red_trans.fits'.format(full_path, tel))
    trans_list = list_files('{}/{}'.format(full_path, tel),
                            end_str='_red_trans_hdr.fits')
    body += ('# transient cats:   {} ({} red-flagged)\n'.format(
        len(trans_list), count_redflags(trans_list, key='TQC-FLAG')))

    #sso_list = glob.glob('{}/{}*_red_trans_sso.fits'.format(full_path, tel))
    sso_list = list_files('{}/{}'.format(full_path, tel),
                          end_str='_red_trans_sso.fits')
    body += ('# SSO cats:         {} ({} empty)\n'.format(
        len(sso_list), count_redflags(sso_list, key='SDUMCAT')))
    body += '\n'


    if email:
        # email the obslog (with the weather page for MeerLICHT as
        # attachment) to a list of interested people
        try:
            # subject
            subject = '{} night report {}'.format(tel, date_dir.replace('/','-'))

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


    if png_tmp:
        copy_file (png_tmp, png_dest, move=True)



    return


################################################################################

def count_redflags(catlist, key='QC-FLAG'):

    nredflags = 0

    for catname in catlist:

        # read file header
        header = read_hdulist (catname, get_data=False, get_header=True)

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

def get_path (date, dir_type, tel):

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
            tzinfo=gettz(obs_timezone))
        if date_ut < date_local_noon:
            # subtract day from date_only
            date = (date_ut - timedelta(1)).strftime('%Y-%m-%d')
        else:
            date = date_ut.strftime('%Y-%m-%d')


    # this [date_eve] in format yyyymmdd is also returned
    date_eve = ''.join(e for e in date if e.isdigit())
    date_dir = '{}/{}/{}'.format(date_eve[0:4], date_eve[4:6], date_eve[6:8])


    if dir_type == 'read':
        root_dir = raw_dir[tel]
    elif dir_type == 'write':
        root_dir = red_dir[tel]
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

def isdir (folder):

    if folder[0:5] == 'gs://':

        if folder[-1] != '/':
            folder = '{}/'.format(folder)

        storage_client = storage.Client()
        bucket_name, bucket_file = get_bucket_name (folder)
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=None, max_results=1)
        nblobs = len(list(blobs))

        return nblobs > 0

    else:

        return os.path.isdir(folder)


################################################################################

def list_files (path, search_str='', end_str='', start_str=None,
                recursive=False):

    """function to return list of files starting with [path] (can be a
       folder or google cloud bucket name and path; this does not have
       to be a precise or complete folder/path, e.g. [path] can be
       some_path/some_file_basename), with possible [end_str] and
       containing [search_str] without any wildcards. If path is an
       exact folder, then [start_str] can be used as the beginning of
       the filename.

    """

    # is google cloud being used?
    google_cloud = (path[0:5] == 'gs://')


    # split input [path] into folder_bucket and prefix; if path is a
    # folder, need to add a slash at the end, otherwise the prefix
    # will be the name of the deepest folder
    if isdir(path) and path[-1] != '/':
        path = '{}/'.format(path)


    # if path is indeed a folder, then prefix will be an empty string,
    # which is fine below
    folder_bucket, prefix = os.path.split(path.split('gs://')[-1])


    # if path consists of just the bucket name including gs:// or just
    # a path without any slashes at all, the above will lead to an
    # empty folder_bucket and prefix=path; turn these around
    if len(folder_bucket)==0:
        folder_bucket = prefix
        prefix = ''


    # if prefix is empty and [start_str] is defined, use
    # that as the prefix
    if prefix=='' and start_str is not None:
        prefix = start_str


    # if not dealing with google cloud buckets, use glob
    if not google_cloud:

        #glob files
        if recursive:
            files = glob.glob('{}/**/{}*{}*{}'.format(folder_bucket, prefix,
                                                      search_str, end_str),
                              recursive=True)
            if path in files:
                files.remove(path)

        else:
            files = glob.glob('{}/{}*{}*{}'.format(folder_bucket, prefix,
                                                   search_str, end_str))

    else:

        # for buckets, use storage.Client().list_blobs; see
        # https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix#storage_list_files_with_prefix-python

        # setting delimiter to '/' restricts the results to only the
        # files in a given folder
        if recursive:
            delimiter = None
        else:
            delimiter = '/'


        # bucket name and prefix (e.g. gs://) to add to output files
        bucket_name, bucket_file = get_bucket_name(path)
        bucket_prefix = path.split(bucket_name)[0]


        if False:
            log.info ('folder_bucket: {}'.format(folder_bucket))
            log.info ('prefix: {}'.format(prefix))
            log.info ('path: {}'.format(path))
            log.info ('bucket_name: {}'.format(bucket_name))
            log.info ('bucket_file: {}'.format(bucket_file))


        # get the blobs
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, prefix=bucket_file,
                                          delimiter=delimiter)

        # collect blobs' names in list of files
        files = []
        for blob in blobs:

            filename = blob.name

            # check for search string; if search string is empty,
            # following if statement will be False
            if search_str not in filename:
                continue

            # check if filename ends with [end_str]
            len_ext = len(end_str)
            if len_ext > 0 and filename[-len_ext:] != end_str:
                # if not, continue with next blob
                continue

            # after surviving above checks, append filename including
            # the bucket prefix and name
            files.append('{}{}/{}'.format(bucket_prefix, bucket_name, filename))


    #log.info ('files returned by [list_files]: {}'.format(files))
    log.info ('number of files returned by [list_files]: {}'.format(len(files)))


    return files


################################################################################

def copy_file (src_file, dest, move=False, verbose=True):

    """function to copy or move a file [src_file] to [dest], which may
       be a file or folder; [src_file] and/or [dest] may be part of
       the usual filesystem or in a google cloud bucket; in the latter
       case the argument(s) should start with gs://[bucket_name]

    """

    if verbose:
        if move:
            label = 'moving'
        else:
            label = 'copying'

        log.info('{} {} to {}'.format(label, src_file, dest))


    # if not dealing with google cloud buckets, use shutil.copy2 or shutil.move
    if not (src_file[0:5] == 'gs://' or dest[0:5] == 'gs://'):

        if not move:
            shutil.copy2(src_file, dest)
        else:
            shutil.move(src_file, dest)

    else:

        # this could be done in python, but much easier with gsutil
        # from the shell
        if move:
            cp_cmd = 'mv'
        else:
            cp_cmd = 'cp'


        cmd = ['gsutil', '-q', cp_cmd, src_file, dest]
        result = subprocess.run(cmd)
        #result = subprocess.run(cmd, capture_output=True)
        #log.info(result.stdout.decode('UTF-8'))


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
            shutil.rmtree(path)

        # do not check if directory exists, just make it; changed this
        # after racing condition occurred on the ilifu Slurm cluster
        # when reducing flatfields, where different tasks need to make
        # the same directory
        os.makedirs(path, exist_ok=True)


    return


################################################################################

def main ():

    """Wrapper allowing [run_blackbox_slurm] to be run from the command line"""

    parser = argparse.ArgumentParser(description='Run BlackBOX on Google Slurm '
                                     'cluster')
    parser.add_argument('--date', type=str, default=None,
                        help='date to process (yyyymmdd, yyyy-mm-dd, yyyy/mm/dd '
                        'or yyyy.mm.dd); default=today (date change is at local '
                        'noon)')
    parser.add_argument('--telescope', type=str, default='BG', help='telescope')
    parser.add_argument('--mode', choices=['day', 'night'], default='night',
                        help='processing mode; night mode will also process new '
                        'incoming images')
    parser.add_argument('--runtime', type=str, default='6:00:00',
                        help='runtime requested per image; default=6:00:00')
    args = parser.parse_args()


    run_blackbox_slurm (args.date, args.telescope, args.mode, args.runtime)


################################################################################

if __name__ == "__main__":
    main()
