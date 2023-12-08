
#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 1
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthreads = 2

#===============================================================================
# Directory structure and files to keep
#===============================================================================

# directory structure and files to keep for the reference building are
# the same as defined in set_blackbox, including the path for the
# temporary directories, with the name of the reference image as
# additional directory; the switch below determines whether to keep
# the reference tmp directory (True) or delete it right after the
# reference image has been built
keep_tmp = False

#===============================================================================
# Imcombine settings
#===============================================================================

# method of image combination; options:
# MEDIAN Take the median of pixel values
# AVERAGE Take the average
# MIN Take the minimum
# MAX Take the maximum
# WEIGHTED Take the weighted average
# CHI2 Take the weighted, quadratic sum
# SUM Take the sum
# not in latest manual v2.21 (code is at v2.38):
# CLIPPED, CHI-OLD, CHI-MODE, CHI-MEAN, WEIGHTED_WEIGHT, MEDIAN_WEIGHT,
# AND, NAND, OR or NOR
combine_type = 'clipped'

# maximum spread in seeing values (using PSF-SEE) allowed when
# selecting images to combine in 'clipped' mode to avoid clipping of
# bright stars; abs(highest seeing / lowest seeing - 1) <=
# max_spread_seeing
max_spread_seeing = 0.3
# range in SWarp noise amplification fraction to explore for images to
# combine in 'clipped' mode to avoid clipping of bright stars
A_range = [0.3, 5.1, 0.1]
# clipping threshold range used in SWarp
nsigma_range = [2.5, 3.6, 0.5]
# minimum number of selected images required in clipped mode;
# resorting to weighted average of all images if not reached
nmin_4clipping = 3


# sum of mask type integers (bad=1,..) to discard
masktype_discard = 63 # i.e. discard bad (1) + cosmic (2) + satellite (16)
                      # + edge (32)

# centering method; options: 'grid', 'median_field' or 'median_filter'
center_type = 'grid'

# output image size method; options: 'input', 'all_field' or
# 'all_filter'; if [center_type] = 'median_filter' then imagesize_type
# cannot be 'all_field'
imagesize_type = 'all_field'

# pixelscale type; options same as SWarp's PIXELSCALE_TYPE options:
# median, min, max, manual, fit
pixscale_type = 'manual'
# fixed pixel scale of output image; only used in case [pixscale_type]
# is set to 'manual'
pixscale_out = 0.5642


# background subtraction method; options: 'blackbox', 'auto',
# 'manual', 'constant', 'none'; N.B.: the background boxsize and
# filtersize are taken from the ZOGY settings file!
back_type = 'blackbox'


#===============================================================================
# Select subset to combine in case of many images of the same field/filter
#===============================================================================

# use absolute target limiting magnitudes suggested by PaulG; see his
# email from 2021-01-06
limmag_target = {'u': 21.0, 'g': 22.3, 'q': 22.5, 'r': 22.0, 'i': 21.3, 'z': 20.5}
# 0.5 mag deeper:
#limmag_target = {'u': 21.5, 'g': 22.8, 'q': 23.0, 'r': 22.5, 'i': 21.8, 'z': 21.0}
# use all images
#limmag_target = {'u': 30.0, 'g': 30.0, 'q': 30.0, 'r': 30.0, 'i': 30.0, 'z': 30.0}
# do not use less than [nmin] images if available
nimages_min = 15
