
#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 8
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthread = 1

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

# sum of mask type integers (bad=1,..) to discard'
masktype_discard = 49 # i.e. discard bad (1) + satellite (16) + edge (32)

# centering method; options: 'grid', 'first', 'last', 'mean', 'median'
center_type = 'mean'

# background subtraction method; options: 'blackbox', 'auto',
# 'manual', 'constant', 'none'
back_type = 'new'

# N.B.: the background boxsize and filtersize are taken from the ZOGY
# settings file!

#===============================================================================
# Select subset to combine in case of many images of the same field/filter
#===============================================================================

# maximum number of images to combine
subset_nmax = 20
# header keyword on which to sort if number of images exceeds
# [subset_max]; N.B.: keyword value needs to be an integer or float in
# order to be able to sort
subset_key = 'LIMMAG'
# include low end (True) or high end (False) of keyword values?; True for
# e.g. S-SEEING, but False for e.g. LIMMAG
subset_lowend = False
