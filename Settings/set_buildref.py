
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

# maximum spread in seeing values (using S-SEEING) allowed when
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


# scale channel zeropoints to full-image zeropoint
scale_chan_zps = True


#===============================================================================
# Select subset to combine in case of many images of the same field/filter
#===============================================================================

# use absolute target limiting magnitudes suggested by PaulG; see his
# email from 2021-01-06; if input parameter deep is True, this target
# limmag is not considered
limmag_target = {'ML1': {'u': 21.0, 'g': 22.3, 'q': 22.5, 'r': 22.0, 'i': 21.3, 'z': 20.5},
# 1st reference images for Blackgem (March 2024): ~deepest image taken + 1 mag
                 'BG': {'u': 21.1, 'g': 22.0, 'q': 22.8, 'r': 21.8, 'i': 21.8, 'z': 20.3}}
# use all images
#limmag_target = {'u': 30.0, 'g': 30.0, 'q': 30.0, 'r': 30.0, 'i': 30.0, 'z': 30.0}


# require at least this number of images, after the
# date/qc-flag/seeing cuts, to create a co-added image
nimages_min = 3

# maximum number of images used, after the date/qc-flag/seeing cuts
# and sorting the list by LIMMAG; if input parameter deep is True,
# this maximum is disregarded
nimages_max = 25

# minimum limiting magnitude improvement - with respect to possibly
# existing reference image in the same filter - required for new ref
# image to be created; if set to None, this requirement is dropped
dlimmag_min = 0.5
