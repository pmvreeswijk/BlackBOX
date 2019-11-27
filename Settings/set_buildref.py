
#===============================================================================
# Number of processes and threads
#===============================================================================

# number of processes to run in parallel
nproc = 1
# maximum number of threads for each process (this parameter
# cannot be made telescope dependent through a dictionary!)
nthread = 2

#===============================================================================
# Gain correction
#===============================================================================

# apply gain fine-tuning inferred from master flat
tune_gain = False

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
combine_type = 'weighted'

# sum of mask type integers (bad=1,..) to discard'
masktype_discard = 49 # i.e. discard bad (1) + satellite (16) + edge (32)

# centering method; options: 'grid', 'first', 'last', 'mean', 'median'
center_type = 'grid'

# background subtraction method; options: 'blackbox', 'auto'
back_type = 'blackbox'

# background mesh size (pixels) used in case back_type is set to 'auto'
back_size = 120

# size (in background meshes) of the background-filtering mask
back_filtersize = 3
