
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
combine_type = 'median'

# sum of mask type integers (bad=1,..) to discard'
masktype_discard = 49 # i.e. discard bad (1) + satellite (16) + edge (32)

# centering method; options: 'grid', 'median_field' or 'median_filter'
center_type = 'median_filter'

# background subtraction method; options: 'blackbox', 'auto',
# 'manual', 'constant', 'none'
back_type = 'blackbox'

# N.B.: the background boxsize and filtersize are taken from the ZOGY
# settings file!

# pixelscale type; options same as SWarp's PIXELSCALE_TYPE options:
# median, min, max, manual, fit
pixscale_type = 'manual'
# fixed pixel scale of output image; only used in case [pixscale_type]
# is set to 'manual'
pixscale_out = 0.5642


#===============================================================================
# Select subset to combine in case of many images of the same field/filter
#===============================================================================

# use absolute target limiting magnitudes suggested by PaulG; see his
# email from 2021-01-06
#use_abslimits = True
limmag_target = {'u': 21.0, 'g': 22.3, 'q': 22.5, 'r': 22.0, 'i': 21.3, 'z': 20.5}
# do not use less than [nmin] images if available 
nimages_min = 15
