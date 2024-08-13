import os
import set_zogy

#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 1
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthreads = 2

#===============================================================================
# Reduction steps
#===============================================================================

# switch on/off different parts
img_reduce = False
cat_extract = False
trans_extract = False

# force re-processing of new image, only for above parts that are
# switched on
force_reproc_new = False

# switch to create the reference image/folder from the image being
# processed in case it does not exist yet for that combination of
# OBJECT/FieldID and filter
create_ref = False
# create master file if it does not exist yet; if False the master
# frame closest in time will be used
create_master = False

# subtract master bias
subtract_mbias = {'ML1': False, 'BG': True}
# perform satellite detection
detect_sats = True
# perform non-linearity correction
correct_nonlin = False
# create master dark
create_mdark = False


# time window [days] within which to include individual biases/darks/flats
# in master (0=only same night, 1=including previous and next night, etc.)
cal_window = {'bias': 3, 'dark': 3, 'flat': 7}
# maximum number of individual bias/dark/flat frames to combine
ncal_max = {'bias': 20, 'dark': 20, 'flat': 15}

# degree polynomial fit to vertical overscan clipped means
voscan_poldeg = 3

#===============================================================================
# Directory structure and files to keep
#===============================================================================

# switch to keep tmp directories (True) or not (False)
keep_tmp = False

# name of telescope data base directory with the different
# subdirectories defined further below
run_dir_base = {'ML1': '/idia/projects/meerlicht', 'BG': os.environ['HOME']}

# temporary directory where data is reduced; ideally this is on a disk
# with fast read/write speed; for ML1, this is the same as run_dir_base
tmp_dir_base = {'ML1': run_dir_base['ML1']}
# for BlackGEM, it depends on whether tmp data is kept for inspection
# or not
if keep_tmp:
    tmp_dir_base['BG'] = '{}/RunBlackBOX/tmp'.format(run_dir_base['BG'])
else:
    tmp_dir_base['BG'] = '/tmp'


# the loop below creates dictionaries with keys ['ML1', 'BG2', 'BG3',
# 'BG4'] for the different paths to the raw, red, log, ref and tmp
# directories, which can be used in [blackbox] to extract the correct
# path for a given telescope
run_dir = {}; raw_dir={}; red_dir={}; log_dir={}; ref_dir={}; tmp_dir={}
master_dir = {}; hdrtables_dir = {}


# ML/BG processing environment: 'test', 'staging', 'production'
proc_env = 'production'


# MeerLICHT directory structure
# -----------------------------
proc_env_dict = {'test': '/test_paulv',
                 'staging': '/staging_paulv',
                 'production': ''}
proc_env_subdir = proc_env_dict[proc_env]


for tel in ['ML1']:
    # raw folder does not depend on the processing environment
    run_dir[tel] = '{}/{}'.format(run_dir_base[tel], tel)
    raw_dir[tel] = '{}/raw'.format(run_dir[tel])
    # the following folders do depend on processing environment
    red_dir[tel] = '{}{}/red'.format(run_dir[tel], proc_env_subdir)
    log_dir[tel] = '{}{}/log'.format(run_dir[tel], proc_env_subdir)
    ref_dir[tel] = '{}{}/ref'.format(run_dir[tel], proc_env_subdir)
    tmp_dir[tel] = '{}/{}{}/tmp'.format(tmp_dir_base[tel], tel, proc_env_subdir)
    hdrtables_dir[tel] = '{}/Headers{}'.format(run_dir_base[tel],
                                               proc_env_subdir)
    if proc_env == 'test':
        # for the test environment, use the existing masters
        master_dir[tel] = '{}/red'.format(run_dir[tel])
    else:
        master_dir[tel] = red_dir[tel]


# BlackGEM buckets at Google cloud
# --------------------------------
proc_env_dict = {'test': 'gs://blackgem-test-env/',
                 'staging': 'gs://blackgem-staging-env/',
                 'production': 'gs://'}
proc_env_base = proc_env_dict[proc_env]

# BlackGEM directory structure
for tel in ['BG2', 'BG3', 'BG4', 'BG']:
    # raw bucket does not depend on the processing environment
    raw_dir[tel] = 'gs://blackgem-raw/{}'.format(tel)
    # neither does the tmp folder
    tmp_dir[tel] = '{}/{}'.format(tmp_dir_base['BG'], tel)

    # reduced, ref, masters and hdrtables
    red_dir[tel] = '{}blackgem-red/{}'.format(proc_env_base, tel)
    #red_dir[tel] = '/home/sa_105685508700717199458/BBtest/{}'.format(tel)
    ref_dir[tel] = '{}blackgem-ref'.format(proc_env_base)
    hdrtables_dir[tel] = '{}blackgem-hdrtables/{}'.format(proc_env_base, tel)

    if proc_env == 'test':
        # for the test environment, use the existing masters
        master_dir[tel] = 'gs://blackgem-masters/{}'.format(tel)
    else:
        master_dir[tel] = '{}blackgem-masters/{}'.format(proc_env_base, tel)



# name endings of files to keep for the reference and new images
img_reduce_exts = ['_red.fits', '_mask.fits', '_red_hdr.fits', '_mini.fits',
                   '_red_limmag.fits', '_red.log', '_red_objmask.fits']
cat_extract_exts = ['_red_cat.fits', '_psf.fits', '_psfex.cat', '_cat_hdr.fits']
trans_extract_exts = ['_D.fits', '_Scorr.fits', '_trans_limmag.fits',
                      '_trans.fits', '_trans_hdr.fits', '_trans_light.fits',
                      '_Fpsf.fits', '_trans_sso.fits', '_sso_predict.fits']
ref_2keep = img_reduce_exts + cat_extract_exts
all_2keep = ref_2keep + trans_extract_exts

#===============================================================================
# Calibration files
#===============================================================================

# all calibration files are in the calibration folder defined in the
# zogy settings file set_zogy
cal_dir = set_zogy.cal_dir

# name of Xtalk file created by Kerry
#crosstalk_file = {'ML1': '{}/crosstalk_20180620.txt'.format(cal_dir)}
# new ones created in August 2024
crosstalk_file = {}
for tel in ['ML1', 'BG2', 'BG3', 'BG4']:
    crosstalk_file[tel] = ('{}/{}_crosstalk_20240807.dat'
                           .format(cal_dir, tel))

# name of initial bad pixel mask; filter dependence is added in
# blackbox, e.g. ML1_bpm_r_0p2_20200727.fits.fz, instead of making
# these dictionaries with the filters as keys.
bad_pixel_mask = {'ML1': '{}/BPM/ML1/ML1_bpm_0p2_20200727.fits.fz'.format(cal_dir),
                  'BG2': '{}/BPM/BG2/BG2_bpm_0p2_20240312.fits.fz'.format(cal_dir),
                  'BG3': '{}/BPM/BG3/BG3_bpm_0p2_20230531.fits.fz'.format(cal_dir),
                  'BG4': '{}/BPM/BG4/BG4_bpm_0p2_20230531.fits.fz'.format(cal_dir)}

# name of ML/BG field definition file
mlbg_fieldIDs = '{}/MLBG_FieldIDs_Feb2022_nGaia.fits'.format(cal_dir)

# name of file with non-linearity correcting spline
nonlin_corr_file = '{}/nonlin_splines_20200501.pkl'.format(cal_dir)

#===============================================================================
# Cosmic ray and satellite trail detection
#===============================================================================

# values adopted for these LA Cosmic's parameters used in
# astroscrappy; play with these values and see what works best; Kerry
# had sigclip=6 and objlim=10
sigclip = {'ML1': 15, 'BG': 20}
sigfrac = 0.01
objlim = 3
niter = 3
# use separable median filter instead of the full median filter;
# [sepmed]=True is significantly faster (factor ~3), but can lead to
# bright stars being masked and corrected as if they are cosmics
sepmed = False

# use Fiore's ASTA code to detect satellites; if set to False,
# acstools will be applied
use_asta = True

# path to ASTA model file
asta_model = '/Software/ASTA/model-best.h5'

# binning used for satellite trail detection
sat_bin = 1

#===============================================================================
# CCD settings and definition of channel/data/overscan/normalisation sections
#===============================================================================

# channel gains, where indices correspond to the channels as follows:
#
# [ 8, 9, 10, 11, 12, 13, 14, 15]
# [ 0, 1,  2,  3,  4,  5,  6,  7]
#
# channel gains:
# from z-band domeflats with x=1, y=10 binning from 2021-01-18:
gain = {'ML1': [2.112, 2.125, 2.130, 2.137, 2.156, 2.158, 2.163, 2.164,
                2.109, 2.124, 2.126, 2.132, 2.136, 2.154, 2.155, 2.157],
        # (np.array([2.11, 2.11, 2.14, 2.15, 2.16, 2.16, 2.20, 2.13,
        #            2.06, 2.13, 2.12, 2.13, 2.14, 2.15, 2.17, 2.17]) *
        # fine-tuned using flat_20210115_q.fits.fz
        # np.array([1.001, 1.007, 0.995, 0.994, 0.998, 0.999, 0.983, 1.016,
        #           1.024, 0.997, 1.003, 1.001, 0.998, 1.002, 0.993, 0.994])),
        # from STA/Archon test report:
        #'ML1': [2.1022, 2.1274, 2.1338, 2.1487, 2.1699, 2.1659, 2.1817, 2.1237,
        #        2.0904, 2.1186, 2.1202, 2.1407, 2.1476, 2.1483, 2.1683, 2.1518],


        # for BGs: starting gain from the STA/Archon test reports (Mode 2, 1MHz),
        #          fine-tuned by gain correction factors from master flats

        'BG2': [2.694, 2.685, 2.691, 2.661, 2.655, 2.673, 2.695, 2.659,
                2.654, 2.748, 2.712, 2.717, 2.714, 2.702, 2.673, 2.743],
        #'BG2': [2.6615, 2.6922, 2.6976, 2.6733, 2.6650, 2.6897, 2.7162, 2.6904,
        #        2.6018, 2.7345, 2.7181, 2.7034, 2.7185, 2.7063, 2.6797, 2.7589],
        # fine-tuned using median header GAINCF?? values from q-band
        # master flats taken from 10-14 March 2024
        #np.array([1.012, 0.997, 0.996, 0.998, 1.005, 0.994, 0.992, 0.988,
        #          1.020, 1.005, 0.998, 1.005, 0.998, 0.998, 0.997, 0.984])

        'BG3': [2.614, 2.609, 2.634, 2.647, 2.600, 2.616, 2.683, 2.649,
                2.680, 2.679, 2.644, 2.604, 2.615, 2.633, 2.615, 2.714],
        # startin gain
        # np.array([2.6547, 2.6541, 2.6378, 2.6249, 2.6058, 2.6068, 2.6335, 2.6166,
        #           2.7061, 2.6962, 2.6424, 2.6171, 2.6228, 2.6219, 2.6076, 2.6878])
        # fine-tuned using median header GAINCF?? values from i-band
        # master master flats taken during the 2nd half of May
        # np.array([0.987, 0.986, 1.001, 1.004, 0.997, 1.003, 1.014, 1.005,
        #           0.985, 0.988, 0.995, 1.005, 1.017, 1.012, 1.000, 1.000])
        # but had to re-adjust correction factors; see also calc_gain_cf.py


        # N.B.: BG4 channels 11 and 12 have exactly the same value;
        # typo in report?  That should be ok after modifying gains
        # with channel gain correction factors.
        'BG4': [2.415, 2.393, 2.365, 2.333, 2.340, 2.320, 2.348, 2.389,
                2.395, 2.403, 2.381, 2.350, 2.362, 2.369, 2.391, 2.430]}
        # starting gain
        # array([2.3747, 2.3645, 2.3559, 2.3736, 2.3637, 2.3916, 2.3694, 2.3845,
        #        2.4095, 2.3719, 2.3644, 2.3644, 2.3574, 2.3785, 2.3835, 2.3755])
        # fine-tuned using median header GAINCF?? values from q-band
        # master master flats taken between May 20 and ~30
        # array([1.017, 1.012, 1.004, 0.983, 0.99 , 0.97 , 0.991, 1.002,
        #        0.994, 1.013, 1.007, 0.994, 1.002, 0.996, 1.003, 1.023])


# assumed saturation level in ADU of raw images; should this be
# changed to electrons, such that single satlevel can be used for all
# telescopes? As potential well should be roughly the same for all
# chips, ~110k e-. After measuring the channel satlevels and
# multiplying by the gains, this potential well varies with ~10% from
# channel to channel, and is still much higher for ML1 (122ke-) than
# for BG3 (95ke-) and BG4 (104ke-), so just stick with separate
# satlevels in ADU
#
#satlevel = {'ML1': 55e3, 'BG2': 40e3, 'BG3': 35e3, 'BG4': 39e3}
# determined from many images using get_satlevel_chans.py
satlevel = {
    'ML1': [5.89e4, 5.94e4, 5.82e4, 5.59e4, 5.60e4, 5.63e4, 5.60e4, 5.75e4,
            5.88e4, 5.81e4, 5.71e4, 5.65e4, 5.59e4, 5.60e4, 5.59e4, 5.65e4],
    'BG2': [3.84e4, 3.77e4, 3.75e4, 3.79e4, 3.79e4, 3.80e4, 3.75e4, 3.93e4,
            4.50e4, 4.08e4, 4.08e4, 4.09e4, 4.07e4, 3.95e4, 4.15e4, 4.37e4],
    'BG3': [3.96e4, 3.83e4, 3.79e4, 3.77e4, 3.81e4, 3.83e4, 3.74e4, 3.94e4,
            4.00e4, 3.98e4, 4.13e4, 4.29e4, 4.29e4, 4.22e4, 4.13e4, 4.38e4],
    'BG4': [4.11e4, 4.09e4, 4.16e4, 4.29e4, 4.32e4, 4.29e4, 4.23e4, 4.41e4,
            4.66e4, 4.60e4, 4.53e4, 4.67e4, 4.66e4, 4.65e4, 4.64e4, 4.66e4]
    }

# ML1: [63, 62, 61, 59, 58, 58, 59, 60,
#       62, 60, 59, 58, 57, 58, 58, 58]
# BG2: [40, 40, 40, 40, 40, 40, 39, 42,
#       45, 46,
# BG3: [39, 38, 37, 36, 40, 37, 39, 39,
#       45, 44, 40, 42, 44, 41, 43, 41]
# BG4: [41, 40, 40, 41, 42, 42, 42, 42,
#       46, 45, 44, 44, 45, 44, 44, 44]


# reduced image data section used for flat normalisation
flat_norm_sec = {'ML1': tuple([slice(6600,9240), slice(5280,7920)]),
                 'BG2': tuple([slice(500,2000),  slice(1320,6600)]),
                 'BG3': tuple([slice(300,1200),  slice(5280,10000)]),
                 'BG4': tuple([slice(2640,5280), slice(3960,7920)])}


# reject evening flats?
flat_reject_eve = {'ML': False, 'BG': True}


# define number of channels in x and y
ny, nx = 2, 8
# and size of data section in each channel
ysize_chan, xsize_chan = 5280, 1320

#===============
# Email settings
#===============
# for ML: sender apparently needs to contain <@astro.ru.nl> for emails
# to actually arrive at Radboud; not relevant for BG/Google Cloud
sender = {'ML1': '<paul.vreeswijk@blackgem.org>',
          'BG': '<paul.vreeswijk@blackgem.org>'}
# comma-separated email addresses of recipients
recipients = {'ML1': 'ml-nightreports@blackgem.org',
              'BG': 'bg-nightreports@blackgem.org'}
reply_to = 'paul.vreeswijk@blackgem.org'
smtp_server = 'smtp-relay.gmail.com'
port = 465
use_SSL = True
