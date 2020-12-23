import os
import numpy as np

#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 1
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthreads = 1

#===============================================================================
# Reduction steps
#===============================================================================

# switch on/off different parts
img_reduce = False
cat_extract = True
trans_extract = False

# force re-processing of new image, only for above parts that are
# switched on
force_reproc_new = True

# subtract master bias
subtract_mbias = False
# perform satellite detection
detect_sats = True
# perform non-linearity correction
correct_nonlin = False


# time window [days] within which to include biases in master bias 
# (0=only same night, 1=including previous and next night, etc.)
bias_window = 3
nbias_max = 20
# time window [days] within which to include flats in master flat 
# (0=only same night, 1=including previous and next night, etc.)
flat_window = 5
nflat_max = 15

# degree polynomial fit to vertical overscan clipped means
voscan_poldeg = 3

#===============================================================================
# Directory structure and files to keep
#===============================================================================

# directory name where [blackbox] is run and the default subdirectories
run_dir_base = os.environ['DATAHOME']

# temporary directory where data is reduced; ideally this is on a disk
# with fast read/write speed - for the moment it is the same as
# [run_dir_base], but could be anywhere
tmp_dir_base = run_dir_base
# switch to keep tmp directories (True) or not (False)
keep_tmp = True

# the loop below creates dictionaries with keys ['ML1', 'BG2', 'BG3',
# 'BG4'] for the different paths to the raw, red, log, ref and tmp
# directories, which can be used in [blackbox] to extract the correct
# path for a given telescope
run_dir = {}; raw_dir={}; red_dir={}; log_dir={}; ref_dir={}; tmp_dir={}
for tel in ['ML1', 'BG2', 'BG3', 'BG4']:
    run_dir[tel] = '{}/{}'.format(run_dir_base, tel)
    raw_dir[tel] = '{}/raw'.format(run_dir[tel])
    red_dir[tel] = '{}/red'.format(run_dir[tel])
    log_dir[tel] = '{}/log'.format(run_dir[tel])
    ref_dir[tel] = '{}/ref'.format(run_dir[tel])
    tmp_dir[tel] = '{}/{}/tmp'.format(tmp_dir_base, tel)

# name endings of files to keep for the reference and new images
img_reduce_exts = ['_red.fits', '_mask.fits', '_hdr.fits', '_mini.fits',
                   '_red_limmag.fits', '_red.log']
cat_extract_exts = ['_cat.fits', '_psf.fits', '_psfex.cat']
trans_extract_exts = ['_D.fits', '_Scorr.fits', '_trans_limmag.fits',
                      '_trans.fits']
all_2keep = img_reduce_exts + cat_extract_exts
ref_2keep = all_2keep
new_2keep = trans_extract_exts + all_2keep

#===============================================================================
# Calibration files
#===============================================================================

# name of Xtalk file created by Kerry
bb_home = os.environ['BLACKBOXHOME']
crosstalk_file = {'ML1': '{}/CalFiles/crosstalk_20180620.txt'.format(bb_home),
                  'BG':  '{}/CalFiles/crosstalk_20180620.txt'.format(bb_home)}

# name of initial bad pixel mask; filter dependence is added in
# blackbox, instead of making these dictionaries with the filters as
# keys.
bad_pixel_mask = {'ML1': '{}/CalFiles/bpm_0p2_20200727.fits.fz'.format(bb_home),
                  'BG':  '{}/CalFiles/bpm_0p2_20200727.fits.fz'.format(bb_home)}

# name of ML/BG field definition file
mlbg_fieldIDs = ('{}/CalFiles/MLBG_FieldIDs_Mar2019.dat'.format(bb_home))

# name of file with non-linearity correcting spline
nonlin_corr_file = {'ML1': '{}/CalFiles/nonlin_splines_20200501.pkl'
                    .format(bb_home),
                    'BG':  '{}/CalFiles/nonlin_splines_20200501.pkl'
                    .format(bb_home)}

#===============================================================================
# Cosmic ray and satellite trail detection
#===============================================================================

# values adopted for these LA Cosmic's parameters used in
# astroscrappy; play with these values and see what works best; Kerry
# had sigclip=6 and objlim=10
sigclip = {'ML1': 8.0, 'BG': 20}
sigfrac = 0.01
objlim = 1.0
niter = 2
# use separable median filter instead of the full median filter;
# [sepmed]=True is significantly faster (factor ~3), but can lead to
# bright stars being masked and corrected as if they are cosmics
sepmed = False

# binning used for satellite trail detection
sat_bin = 2

#===============================================================================
# CCD settings and definition of channel/data/overscan/normalisation sections
#===============================================================================

# channel gains, where indices correspond to the channels as follows:
#
# [ 8, 9, 10, 11, 12, 13, 14, 15]
# [ 0, 1,  2,  3,  4,  5,  6,  7]
#
# channel gains:
# defined by Kerry:
#gain = [2.29,2.31,2.30,2.32,2.37,2.36,2.37,2.35,2.28,2.31,2.31,2.35,2.35,2.35,2.35,2.36]
#gain[10] = 2.38
# from domeflats 2019-01-10:
gain = {'ML1': (np.array([2.31, 2.39, 2.40, 2.44, 2.43, 2.42, 2.47, 2.40,
                          2.32, 2.43, 2.38, 2.39, 2.43, 2.46, 2.51, 2.51]) *
                # ML correction factor from q-band master flat 3 June 2019, with flat_window=1
                np.array([1.044, 1.016, 1.014, 1.000, 1.012, 1.017, 0.998, 1.027,
                          1.038, 0.998, 1.021, 1.017, 1.002, 0.998, 0.978, 0.980])),
        'BG':  (np.array([2.60, 2.60, 2.60, 2.60, 2.60, 2.60, 2.60, 2.60,
                          2.60, 2.60, 2.60, 2.60, 2.60, 2.60, 2.60, 2.60]))}

# from STA test report SN22838
#gain = {'ML1': (np.array([2.1022, 2.1274, 2.1338, 2.1487, 2.1699, 2.1659, 2.1817, 2.1237,
#                          2.0904, 2.1186, 2.1202, 2.1407, 2.1476, 2.1483, 2.1683, 2.1518]))}
# determined from 20170707-1MHz-1s10pPTC-Scaled test data by PMV (20200501):
#gain = {'ML1': (np.array([2.16, 2.17, 2.20, 2.20, 2.24, 2.22, 2.26, 2.17,
#                          2.15, 2.16, 2.14, 2.19, 2.18, 2.25, 2.20, 2.20]))


# assumed saturation level in ADU of raw images
satlevel = 55000

# reduced image data section used for flat normalisation
flat_norm_sec = {'ML1': tuple([slice(6600,9240), slice(5280,7920)])}

# define number of channels in x and y
ny, nx = 2, 8
# and size of data section in each channel
ysize_chan, xsize_chan = 5280, 1320

#===============
# Email settings
#===============
# for ML: sender apparently needs to contain <@astro.ru.nl> for emails
# to actually arrive at Radboud; not relevant for BG/Google Cloud
sender = {'ML1': 'MeerLICHT night log <p.vreeswijk@astro.ru.nl>',
          'BG': 'BlackGEM night log'}
# comma-separated email addresses of recipients
recipients = 'p.vreeswijk@astro.ru.nl'
reply_to = 'p.vreeswijk@astro.ru.nl'
smtp_server = {'ML1': 'localhost', 'BG': 'smtp-relay.gmail.com'}
port = {'ML1': 0, 'BG': 465}
use_SSL = {'ML1': False, 'BG': True}
