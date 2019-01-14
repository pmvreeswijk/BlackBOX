import os

__version__ = '0.7'

#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 2
# maximum number of threads for each process
nthread = 2

#===============================================================================
# Directory structure and files to keep
#===============================================================================

# directory name where [bb] is run and the default subdirectories
run_dir = '/media/data/pmv/Test_BGreduce_subpipe_zogy'
raw_dir = '{}/raw'.format(run_dir)
red_dir = '{}/red'.format(run_dir)
log_dir = '{}/log'.format(run_dir)
ref_dir = '{}/ref'.format(run_dir)
# directory where data is reduced; ideally this is on a disk with fast
# read/write speed
tmp_dir = '{}/tmp'.format(run_dir)

# switch to keep tmp directories (True) or not (False)
keep_tmp = True

# name endings of files to keep for the reference and new images
all_2keep = ['_red.fits', '_mask.fits', '_cat.fits', '_mini.fits', '_red.log']
ref_2keep = ['_ldac.fits', '_psf.fits', '_psfex.cat'] + all_2keep
new_2keep = ['_D.fits', '_Scorr.fits', '_Fpsf.fits','_Fpsferr.fits',
             '_trans.fits'] + all_2keep


#===============================================================================
# Calibration files
#===============================================================================

# name of Xtalk file created by Kerry
crosstalk_file = os.environ['ZOGYHOME']+'/CalFiles/crosstalk_20180620.txt'

# name of initial bad pixel mask
bad_pixel_mask = os.environ['ZOGYHOME']+'/CalFiles/bpm_u_0p05.fits'
        
#===============================================================================
# Cosmic ray and satellite trail detection
#===============================================================================

# values adopted for these LA Cosmic's parameters used in
# astroscrappy; play with these values and see what works best; Kerry
# had sigclip=6 and objlim=10
sigclip = 4.5
sigfrac = 0.3
objlim = 10.0
niter = 3

# binning used for satellite trail detection
sat_bin = 2

#===============================================================================
# CCD settings and definition of channel/data/overscan/normalisation sections
#===============================================================================

# check if different channels in [set_bb.gain] correspond to the
# correct channels; currently indices of gain correspond to the
# channels as follows:
#
# [ 8, 9, 10, 11, 12, 13, 14, 15]
# [ 0, 1,  2,  3,  4,  5,  6,  7]
#
# which are the same indices for the sections defined below

# channel gains:
gain = [2.29,2.31,2.30,2.32,2.37,2.36,2.37,2.35,2.28,2.31,2.31,2.35,2.35,2.35,2.35,2.36]
gain[10] = 2.38

# assumed saturation level in ADU of raw images
satlevel = 55000.

# reduced image data section used for flat normalisation
flat_norm_sec = tuple([slice(5300,6300), slice(4100,5100)])

# define channel, data, overscan and normalisation sections
ysize, ny, os_ysize = 10600, 2,  20; dy = int(ysize/ny)
xsize, nx, os_xsize = 12000, 8, 180; dx = int(xsize/nx)

# the sections below are defined such that e.g. chan_sec[0] refers to
# all pixels of the first channel, where the channels are currently
# defined to be located on the CCD as follows:
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

