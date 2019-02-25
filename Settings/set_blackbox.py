import os

__version__ = '0.8'

#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 2
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthread = 2

#===============================================================================
# Reduction switches
#===============================================================================

# subtract master bias
subtract_mbias = False

#===============================================================================
# Directory structure and files to keep
#===============================================================================

# directory name where [blackbox] is run and the default subdirectories
run_dir_base = os.environ['DATA_HOME']

# temporary directory where data is reduced; ideally this is on a disk
# with fast read/write speed - for the moment it is the same as
# [run_dir_base], but could be anywhere
tmp_dir_base = run_dir_base
# switch to keep tmp directories (True) or not (False)
keep_tmp = False

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
all_2keep = ['_red.fits', '_mask.fits', '_cat.fits', '_mini.fits', '_red.log']
ref_2keep = ['_ldac.fits', '_psf.fits', '_psfex.cat'] + all_2keep
new_2keep = ['_D.fits', '_Scorr.fits', '_Fpsf.fits','_Fpsferr.fits',
             '_trans.fits'] + all_2keep

#===============================================================================
# Calibration files
#===============================================================================

# name of Xtalk file created by Kerry
crosstalk_file = {'ML1': os.environ['BLACKBOX_HOME']+'/CalFiles/crosstalk_20180620.txt'}

# name of initial bad pixel mask
bad_pixel_mask = {'ML1': os.environ['BLACKBOX_HOME']+'/CalFiles/bpm_u_0p05.fits'}

#===============================================================================
# Cosmic ray and satellite trail detection
#===============================================================================

# values adopted for these LA Cosmic's parameters used in
# astroscrappy; play with these values and see what works best; Kerry
# had sigclip=6 and objlim=10
sigclip = 6.0
sigfrac = 0.3
objlim = 10.0
niter = 3

# binning used for satellite trail detection
sat_bin = 2

#===============================================================================
# CCD settings and definition of channel/data/overscan/normalisation sections
#===============================================================================

# check if different channels in [set_blackbox.gain] correspond to the
# correct channels; currently indices of gain correspond to the
# channels as follows:
#
# [ 8, 9, 10, 11, 12, 13, 14, 15]
# [ 0, 1,  2,  3,  4,  5,  6,  7]
#
# which are the same indices for the sections defined below

# channel gains:
# defined by Kerry:
#gain = [2.29,2.31,2.30,2.32,2.37,2.36,2.37,2.35,2.28,2.31,2.31,2.35,2.35,2.35,2.35,2.36]
#gain[10] = 2.38
# from domeflats 2019-01-10:
gain = {'ML1': [2.31,2.39,2.40,2.44,2.43,2.42,2.47,2.40,2.32,2.43,2.38,2.39,2.43,2.46,2.51,2.51]}

# assumed saturation level in ADU of raw images
satlevel = 55000

# reduced image data section used for flat normalisation
flat_norm_sec = {'ML1': tuple([slice(5300,6300), slice(4100,5100)])}

# define channel, data, overscan and normalisation sections
ysize, ny, os_ysize = 10600, 2,  20; dy = int(ysize/ny)
xsize, nx, os_xsize = 12000, 8, 180; dx = int(xsize/nx)

# the sections below are defined such that e.g. chan_sec[0] refers to
# all pixels of the first channel, where the channel indices are
# currently defined to be located on the CCD as follows:
#
# [ 8, 9, 10, 11, 12, 13, 14, 15]
# [ 0, 1,  2,  3,  4,  5,  6,  7]
        
# channel section slices including overscan; shape=(16,2)
chan_sec = tuple([(slice(y,y+dy), slice(x,x+dx))
                  for y in range(0,ysize,dy) for x in range(0,xsize,dx)])
# channel data section slices; shape=(16,2)
data_sec = tuple([(slice(y,y+dy-os_ysize), slice(x,x+dx-os_xsize))
                  for y in range(0,ysize,dy+os_ysize) for x in range(0,xsize,dx)])
# channel vertical overscan section slices; shape=(16,2)
os_sec_vert = tuple([(slice(y,y+dy), slice(x+dx-os_xsize,x+dx-1))
                     for y in range(0,ysize,dy) for x in range(0,xsize,dx)])
# channel horizontal overscan sections; shape=(16,2)
# cut off 5 pixels from os_ysize
os_sec_hori = tuple([(slice(y,y+15), slice(x,x+dx-os_xsize))
                     for y in range(dy-15,dy+15,15) for x in range(0,xsize,dx)])
# channel reduced data section slices; shape=(16,2)
data_sec_red = tuple([(slice(y,y+dy-os_ysize), slice(x,x+dx-os_xsize))
                      for y in range(0,ysize-ny*os_ysize,dy-os_ysize)
                      for x in range(0,xsize-nx*os_xsize,dx-os_xsize)])
