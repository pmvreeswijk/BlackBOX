
"""Dictionary of allowed ranges of header keyword values to be used
   for the automatic data quality control (QC) of MeerLICHT and
   BlackGEM data.  This dictionary is used by the [qc_check] function
   in the [qc] module.

   The value in the qc_range dictionary is also a dictionary with
   the keys:
         1) 'default': the default value that will be used in case
            the keyword is not present in the input header and a dummy 
            catalog is being created
         2) 'val_type': providing the type of range provided
         3) 'val_range': a list of tuples, each containing either one
            (for type='bool') or two values (for the other types).
         4) 'cat_type': catalog type ('new', 'ref', 'trans' or 'all')
         5) 'comment': header comment / brief description of keyword

   Depending on 'val_type', these values are interpreted differently
         1) 'min_max': (C1, C2) such that C1 <= value <= C2
         2) 'bool': (C) such that value==C or value==C2
         3) 'sigma': (E, STD) such that abs(value-E) <= n*STD,
                     where n is a list of predefined factors
                     corresponding to the accepted ranges.
                     currently: n_std = [2, 3, 7]
         4) 'exp_abs': (E, C) such that abs(value-E) <= C
         5) 'exp_frac': (E, f) such that abs((value-E)/E) <= f 

   The value of each keyword is first checked against the first
   element of the 'range' key. If the value within this range, the key
   gets a 'green' flag. If it is not within this range, it checks the
   value with the 2nd range provided.  If there is no 2nd range
   provided, the key gets a 'red' flag.  If the value is within the
   2nd range, the corresponding flag will be 'yellow'. The 3rd range
   corresponds to the 'orange' flag, and if the keyword value is not
   within any of the ranges provided, it gets a 'red' flag.

   For the value type 'sigma' only the expected value (E) and a
   standard deviation can be provided, and these are expanded to three
   ranges using: n_std = [2, 3, 7]. So if a keyword value is within
   n_std * STD, its color flag will be 'green', 'yellow' and 'orange',
   respectively. If outside of this, it will be flagged 'red'.

"""

qc_range = {
    'ML1': {

        # 'raw' image header keywords
        'GPS-SHUT': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.88, 0.03) ],            'cat_type': 'all', 'comment': '[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME'},
        
        # Main processing steps
        'XTALK-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'corrected for crosstalk?'},
        'NONLIN-P': {'default': False, 'val_type': 'bool', 'val_range': [ True, False ],             'cat_type': 'all', 'comment': 'corrected for non-linearity?'},
        'GAIN-P'  : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'corrected for gain?'},
        'OS-P'    : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'corrected for overscan?'},
        'MBIAS-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True, False ],             'cat_type': 'all', 'comment': 'corrected for master bias?'},
        'MBIAS-F' : {'default': None,  'val_type': 'skip', 'val_range': None,                        'cat_type': 'all', 'comment': 'name of master bias applied'},
        'MFLAT-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'corrected for master flat?'},
        'MFLAT-F' : {'default': None,  'val_type': 'skip', 'val_range': None,                        'cat_type': 'all', 'comment': 'name of master flat applied'},
        'MFRING-P': {'default': False, 'val_type': 'bool', 'val_range': [ True, False ],             'cat_type': 'all', 'comment': 'corrected for master fringe map?'},
        'MFRING-F': {'default': None,  'val_type': 'skip', 'val_range': None,                        'cat_type': 'all', 'comment': 'name of master fringe map applied'},
        'COSMIC-P': {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'corrected for cosmics rays?'},
        'SAT-P'   : {'default': False, 'val_type': 'bool', 'val_range': [ True, False ],             'cat_type': 'all', 'comment': 'processed for satellite trails?'},
        'S-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'successfully processed by SExtractor?'},
        'A-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'successfully processed by Astrometry.net?'},
        'PSF-P'   : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'successfully processed by PSFEx?'},
        'PC-P'    : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'all', 'comment': 'successfully processed by phot. calibration?'},
        'SWARP-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'trans', 'comment': 'reference image successfully SWarped?'},
        'Z-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'cat_type': 'trans', 'comment': 'successfully processed by ZOGY?'},

        # Channel bias levels [e-]
        # 2019 values
        'BIASMEAN': {'default': None, 'val_type': 'sigma', 'val_range': [ (  7333.520, 3*30.891) ], 'cat_type': 'all', 'comment': 'average all channel means vertical overscan'},
        # for the moment, skip the value range check on the individual channels' bias levels ('sigma' replaced with 'skip')
        'BIASM1'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  6933.564,   32.281) ], 'cat_type': 'all', 'comment': 'channel 1 mean vertical overscan'},
        'BIASM2'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7199.254,   34.481) ], 'cat_type': 'all', 'comment': 'channel 2 mean vertical overscan'},
        'BIASM3'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7291.843,   31.315) ], 'cat_type': 'all', 'comment': 'channel 3 mean vertical overscan'},
        'BIASM4'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7384.878,   30.259) ], 'cat_type': 'all', 'comment': 'channel 4 mean vertical overscan'},
        'BIASM5'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7262.722,   29.910) ], 'cat_type': 'all', 'comment': 'channel 5 mean vertical overscan'},
        'BIASM6'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7275.950,   30.754) ], 'cat_type': 'all', 'comment': 'channel 6 mean vertical overscan'},
        'BIASM7'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7447.558,   31.199) ], 'cat_type': 'all', 'comment': 'channel 7 mean vertical overscan'},
        'BIASM8'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7169.434,   28.927) ], 'cat_type': 'all', 'comment': 'channel 8 mean vertical overscan'},
        'BIASM9'  : {'default': None, 'val_type': 'skip', 'val_range': [ (  7011.460,   31.531) ], 'cat_type': 'all', 'comment': 'channel 9 mean vertical overscan'},
        'BIASM10' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7500.022,   32.602) ], 'cat_type': 'all', 'comment': 'channel 10 mean vertical overscan'},
        'BIASM11' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7307.696,   29.695) ], 'cat_type': 'all', 'comment': 'channel 11 mean vertical overscan'},
        'BIASM12' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7334.698,   32.213) ], 'cat_type': 'all', 'comment': 'channel 12 mean vertical overscan'},
        'BIASM13' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7460.912,   27.949) ], 'cat_type': 'all', 'comment': 'channel 13 mean vertical overscan'},
        'BIASM14' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7591.438,   26.561) ], 'cat_type': 'all', 'comment': 'channel 14 mean vertical overscan'},
        'BIASM15' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7567.986,   31.364) ], 'cat_type': 'all', 'comment': 'channel 15 mean vertical overscan'},
        'BIASM16' : {'default': None, 'val_type': 'skip', 'val_range': [ (  7600.082,   34.135) ], 'cat_type': 'all', 'comment': 'channel 16 mean vertical overscan'},
        # 2017 values
        #'BIASMEAN': {'val_type': 'sigma', 'val_range': [ (  7101.809,   29.768) ], 'cat_type': 'all', 'comment': 'average all channel means vertical overscan'},
        #'BIASM1'  : {'val_type': 'sigma', 'val_range': [ (  6915.888,   33.458) ], 'cat_type': 'all', 'comment': 'channel 1 mean vertical overscan'},
        #'BIASM2'  : {'val_type': 'sigma', 'val_range': [ (  7012.454,   34.318) ], 'cat_type': 'all', 'comment': 'channel 2 mean vertical overscan'},
        #'BIASM3'  : {'val_type': 'sigma', 'val_range': [ (  7023.386,   32.152) ], 'cat_type': 'all', 'comment': 'channel 3 mean vertical overscan'},
        #'BIASM4'  : {'val_type': 'sigma', 'val_range': [ (  7052.668,   31.735) ], 'cat_type': 'all', 'comment': 'channel 4 mean vertical overscan'},
        #'BIASM5'  : {'val_type': 'sigma', 'val_range': [ (  7125.328,   33.090) ], 'cat_type': 'all', 'comment': 'channel 5 mean vertical overscan'},
        #'BIASM6'  : {'val_type': 'sigma', 'val_range': [ (  7142.417,   33.233) ], 'cat_type': 'all', 'comment': 'channel 6 mean vertical overscan'},
        #'BIASM7'  : {'val_type': 'sigma', 'val_range': [ (  7169.577,   30.164) ], 'cat_type': 'all', 'comment': 'channel 7 mean vertical overscan'},
        #'BIASM8'  : {'val_type': 'sigma', 'val_range': [ (  7077.317,   30.216) ], 'cat_type': 'all', 'comment': 'channel 8 mean vertical overscan'},
        #'BIASM9'  : {'val_type': 'sigma', 'val_range': [ (  6905.847,   31.534) ], 'cat_type': 'all', 'comment': 'channel 9 mean vertical overscan'},
        #'BIASM10' : {'val_type': 'sigma', 'val_range': [ (  7129.008,   29.231) ], 'cat_type': 'all', 'comment': 'channel 10 mean vertical overscan'},
        #'BIASM11' : {'val_type': 'sigma', 'val_range': [ (  7120.504,   27.446) ], 'cat_type': 'all', 'comment': 'channel 11 mean vertical overscan'},
        #'BIASM12' : {'val_type': 'sigma', 'val_range': [ (  7213.597,   28.497) ], 'cat_type': 'all', 'comment': 'channel 12 mean vertical overscan'},
        #'BIASM13' : {'val_type': 'sigma', 'val_range': [ (  7226.578,   28.769) ], 'cat_type': 'all', 'comment': 'channel 13 mean vertical overscan'},
        #'BIASM14' : {'val_type': 'sigma', 'val_range': [ (  7275.009,   25.343) ], 'cat_type': 'all', 'comment': 'channel 14 mean vertical overscan'},
        #'BIASM15' : {'val_type': 'sigma', 'val_range': [ (  7101.251,   28.866) ], 'cat_type': 'all', 'comment': 'channel 15 mean vertical overscan'},
        #'BIASM16' : {'val_type': 'sigma', 'val_range': [ (  7154.681,   30.773) ], 'cat_type': 'all', 'comment': 'channel 16 mean vertical overscan'},

        # Channel read noise [e-]
        # 2019 values
        'RDNOISE' : {'default': None, 'val_type': 'sigma', 'val_range': [ (    11.258,   3*0.181) ], 'cat_type': 'all', 'comment': 'average all channel sigmas vertical overscan'},
        # for the moment, skip the value range check on the individual channels' readnoise ('sigma' replaced with 'skip')
        'RDN1'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    13.222,    0.230) ], 'cat_type': 'all', 'comment': 'channel 1 sigma (STD) vertical overscan'},
        'RDN2'    : {'default': None, 'val_type': 'skip', 'val_range': [ (     7.853,    0.144) ], 'cat_type': 'all', 'comment': 'channel 2 sigma (STD) vertical overscan'},
        'RDN3'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    13.436,    0.202) ], 'cat_type': 'all', 'comment': 'channel 3 sigma (STD) vertical overscan'},
        'RDN4'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    12.608,    0.190) ], 'cat_type': 'all', 'comment': 'channel 4 sigma (STD) vertical overscan'},
        'RDN5'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    12.566,    0.199) ], 'cat_type': 'all', 'comment': 'channel 5 sigma (STD) vertical overscan'},
        'RDN6'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    12.234,    0.253) ], 'cat_type': 'all', 'comment': 'channel 6 sigma (STD) vertical overscan'},
        'RDN7'    : {'default': None, 'val_type': 'skip', 'val_range': [ (     7.816,    0.166) ], 'cat_type': 'all', 'comment': 'channel 7 sigma (STD) vertical overscan'},
        'RDN8'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    14.106,    0.254) ], 'cat_type': 'all', 'comment': 'channel 8 sigma (STD) vertical overscan'},
        'RDN9'    : {'default': None, 'val_type': 'skip', 'val_range': [ (    11.921,    0.178) ], 'cat_type': 'all', 'comment': 'channel 9 sigma (STD) vertical overscan'},
        'RDN10'   : {'default': None, 'val_type': 'skip', 'val_range': [ (     7.998,    0.159) ], 'cat_type': 'all', 'comment': 'channel 10 sigma (STD) vertical overscan'},
        'RDN11'   : {'default': None, 'val_type': 'skip', 'val_range': [ (    10.896,    0.195) ], 'cat_type': 'all', 'comment': 'channel 11 sigma (STD) vertical overscan'},
        'RDN12'   : {'default': None, 'val_type': 'skip', 'val_range': [ (     9.342,    0.171) ], 'cat_type': 'all', 'comment': 'channel 12 sigma (STD) vertical overscan'},
        'RDN13'   : {'default': None, 'val_type': 'skip', 'val_range': [ (    14.306,    0.246) ], 'cat_type': 'all', 'comment': 'channel 13 sigma (STD) vertical overscan'},
        'RDN14'   : {'default': None, 'val_type': 'skip', 'val_range': [ (    14.110,    0.261) ], 'cat_type': 'all', 'comment': 'channel 14 sigma (STD) vertical overscan'},
        'RDN15'   : {'default': None, 'val_type': 'skip', 'val_range': [ (     9.419,    0.184) ], 'cat_type': 'all', 'comment': 'channel 15 sigma (STD) vertical overscan'},
        'RDN16'   : {'default': None, 'val_type': 'skip', 'val_range': [ (     8.231,    0.158) ], 'cat_type': 'all', 'comment': 'channel 16 sigma (STD) vertical overscan'},     
        ## 2017 values
        #'RDNOISE' : {'val_type': 'sigma', 'val_range': [ (    10.454,    0.730) ], 'cat_type': 'all', 'comment': 'average all channel sigmas vertical overscan'},
        #'RDN1'    : {'val_type': 'sigma', 'val_range': [ (    11.643,    0.859) ], 'cat_type': 'all', 'comment': 'channel 1 sigma (STD) vertical overscan'},
        #'RDN2'    : {'val_type': 'sigma', 'val_range': [ (     8.943,    0.327) ], 'cat_type': 'all', 'comment': 'channel 2 sigma (STD) vertical overscan'},
        #'RDN3'    : {'val_type': 'sigma', 'val_range': [ (    11.440,    0.816) ], 'cat_type': 'all', 'comment': 'channel 3 sigma (STD) vertical overscan'},
        #'RDN4'    : {'val_type': 'sigma', 'val_range': [ (    11.696,    0.666) ], 'cat_type': 'all', 'comment': 'channel 4 sigma (STD) vertical overscan'},
        #'RDN5'    : {'val_type': 'sigma', 'val_range': [ (    11.629,    0.653) ], 'cat_type': 'all', 'comment': 'channel 5 sigma (STD) vertical overscan'},
        #'RDN6'    : {'val_type': 'sigma', 'val_range': [ (    11.581,    0.578) ], 'cat_type': 'all', 'comment': 'channel 6 sigma (STD) vertical overscan'},
        #'RDN7'    : {'val_type': 'sigma', 'val_range': [ (     9.269,    0.398) ], 'cat_type': 'all', 'comment': 'channel 7 sigma (STD) vertical overscan'},
        #'RDN8'    : {'val_type': 'sigma', 'val_range': [ (    12.045,    0.963) ], 'cat_type': 'all', 'comment': 'channel 8 sigma (STD) vertical overscan'},
        #'RDN9'    : {'val_type': 'sigma', 'val_range': [ (     9.206,    0.391) ], 'cat_type': 'all', 'comment': 'channel 9 sigma (STD) vertical overscan'},
        #'RDN10'   : {'val_type': 'sigma', 'val_range': [ (     9.038,    0.361) ], 'cat_type': 'all', 'comment': 'channel 10 sigma (STD) vertical overscan'},
        #'RDN11'   : {'val_type': 'sigma', 'val_range': [ (    10.094,    0.636) ], 'cat_type': 'all', 'comment': 'channel 11 sigma (STD) vertical overscan'},
        #'RDN12'   : {'val_type': 'sigma', 'val_range': [ (     9.479,    0.417) ], 'cat_type': 'all', 'comment': 'channel 12 sigma (STD) vertical overscan'},
        #'RDN13'   : {'val_type': 'sigma', 'val_range': [ (    10.818,    0.861) ], 'cat_type': 'all', 'comment': 'channel 13 sigma (STD) vertical overscan'},
        #'RDN14'   : {'val_type': 'sigma', 'val_range': [ (    11.382,    0.978) ], 'cat_type': 'all', 'comment': 'channel 14 sigma (STD) vertical overscan'},
        #'RDN15'   : {'val_type': 'sigma', 'val_range': [ (     9.258,    0.412) ], 'cat_type': 'all', 'comment': 'channel 15 sigma (STD) vertical overscan'},
        #'RDN16'   : {'val_type': 'sigma', 'val_range': [ (     9.044,    0.373) ], 'cat_type': 'all', 'comment': 'channel 16 sigma (STD) vertical overscan'},

        # general
        'AIRMASS' : {'default': None, 'val_type': 'min_max', 'val_range': [ (1,2), (2,2.5), (2.5, 2.95) ], 'cat_type': 'all', 'comment': 'Airmass (based on RA, DEC, DATE-OBS)'},
        'N-INFNAN': {'default': None, 'val_type': 'min_max', 'val_range': [ (0,0), (1,10), (11,1e6) ], 'cat_type': 'all', 'comment': 'number of pixels with infinite/nan values'},

        # master bias (these keywords should not end up in dummy catalogs)
        'NBIAS'   : {'default': None, 'val_type': 'min_max', 'val_range': [ (10,50), (7,9), (5,6) ], 'cat_type': None, 'comment': 'number of bias frames combined'},
        'MBMEAN'  : {'default': None, 'val_type': 'sigma', 'val_range':   [ (0, 5) ],               'cat_type': None, 'comment': '[e-] mean master bias'},
        'MBRDN'   : {'default': None, 'val_type': 'sigma', 'val_range':   [ (0, 5) ],               'cat_type': None, 'comment': '[e-] sigma (STD) master bias'},

        # individual flats (this keyword should not end up in dummy catalogs)
        'MEDSEC'  : {'default': None, 'val_type': 'min_max', 'val_range': [ (2.4*20e3, 2.4*30e3), (2.4*15e3, 2.4*35e3), (2.4*10e3, 2.4*40e3) ], 'cat_type': None, 'comment': '[e-] median flat over STATSEC (bias-subtracted)'},
        'RSTDSEC' : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.01) ],              'cat_type': None, 'comment': 'relative sigma (STD) flat over STATSEC'},
        
        # master flat (these keywords should not end up in dummy catalogs)
        'NFLAT'   : {'default': None, 'val_type': 'min_max', 'val_range': [ (6,50), (4,5), (3,3) ], 'cat_type': None, 'comment': 'number of flat frames combined'},
        'MFMEDSEC': {'default': None, 'val_type': 'sigma', 'val_range': [ (         1,    0.001) ], 'cat_type': None, 'comment': 'median master flat over STATSEC'},
        'MFSTDSEC': {'default': None, 'val_type': 'sigma', 'val_range': [ (         0,     0.01) ], 'cat_type': None, 'comment': 'sigma (STD) master flat over STATSEC'},
        'FLATDITH': {'default': None, 'val_type': 'bool', 'val_range': [ True, False ],             'cat_type': None, 'comment': 'majority of flats were dithered'},

        'NCOSMICS': {'default': None, 'val_type': 'min_max', 'val_range': [ (3,50), (2.5,100), (2,200) ], 'cat_type': 'all', 'comment': '[/s] number of cosmic rays identified'},
        'NSATS'   : {'default': None, 'val_type': 'min_max', 'val_range': [ (0,1), (2,3), (4,5) ],  'cat_type': 'all', 'comment': 'number of satellite trails identified'},

        # SExtractor
        'S-NOBJ'  : {'default': None, 'val_type': 'min_max', 'val_range': [ (5e3,5e4), (3e3,2e5), (1e3,1e6) ], 'cat_type': 'all', 'comment': 'number of objects detected by SExtractor'},
        'S-SEEING': {'default': None, 'val_type': 'min_max', 'val_range': [ (2,4), (1,5), (0.5,7) ],'cat_type': 'all', 'comment': '[arcsec] SExtractor seeing estimate'},
        'S-SEESTD': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.1,0.1) ],              'cat_type': 'all', 'comment': '[arcsec] sigma (STD) SExtractor seeing'},
        'S-ELONG' : {'default': None, 'val_type': 'sigma', 'val_range': [ (1.1,0.1) ],              'cat_type': 'all', 'comment': 'SExtractor ELONGATION (A/B) estimate'},
        'S-ELOSTD': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.05,0.05) ],            'cat_type': 'all', 'comment': 'sigma (STD) SExtractor ELONGATION (A/B'},
        'S-BKG'   : {'default': None, 'val_type': 'min_max', 'val_range': [ (50,2e3), (10,1e4), (0,5e4) ], 'cat_type': 'all', 'comment': '[e-] median background full image'},
        'S-BKGSTD': {'default': None, 'val_type': 'min_max', 'val_range': [  (0., 200.)],           'cat_type': 'all', 'comment': '[e-] sigma (STD) background full image'},

        # Astrometry.net        
        'A-PSCALE': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.563, 0.0005) ],        'cat_type': 'all', 'comment': '[arcsec/pix] pixel scale WCS solution'},
        'A-ROT'   : {'default': None, 'val_type': 'sigma', 'val_range': [ (-90, 0.5) ],             'cat_type': 'all', 'comment': '[deg] rotation WCS solution'},

        'A-CAT-F' : {'default': None, 'val_type': 'skip',  'val_range': None,                       'cat_type': 'all', 'comment': 'astrometric catalog'},
        'A-NAST'  : {'default': None, 'val_type': 'min_max', 'val_range': [ (1e3,1e4), (100, 3e4), (50, 1e5) ], 'cat_type': 'all', 'comment': 'number of brightest stars used for WCS'},
        'A-DRA'   : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.01)],               'cat_type': 'all', 'comment': '[arcsec] dRA median offset to astrom. catalog'},
        'A-DDEC'  : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.01)],               'cat_type': 'all', 'comment': '[arcsec] dDEC median offset to astrom. catalog'},
        'A-DRASTD': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.03, 0.02) ],           'cat_type': 'all', 'comment': '[arcsec] dRA sigma (STD) offset'},
        'A-DDESTD': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.03, 0.02) ],           'cat_type': 'all', 'comment': '[arcsec] dDEC sigma (STD) offset'},

        # PSFEx
        'PSF-NOBJ': {'default': None, 'val_type': 'min_max', 'val_range': [ (500,2e4), (100,5e4), (10,1e5) ], 'cat_type': 'all', 'comment': 'number of accepted PSF stars'},
        'PSF-CHI2': {'default': None, 'val_type': 'sigma', 'val_range': [ (1, 0.1) ],               'cat_type': 'all', 'comment': 'final reduced chi-squared PSFEx fit'},
        'PSF-FWHM': {'default': None, 'val_type': 'sigma', 'val_range': [ (6, 1) ],                 'cat_type': 'all', 'comment': '[arcsec] image FWHM inferred by PSFEx'},

        # photometric calibration (PC)
        'PC-CAT-F': {'default': None, 'val_type': 'skip',  'val_range': None,                       'cat_type': 'all', 'comment': 'photometric catalog'},
        'PC-NCAL' : {'default': None, 'val_type': 'min_max', 'val_range': [ (10, 1e4) ],            'cat_type': 'all', 'comment': 'number of brightest photcal stars used'},
        'PC-ZP'   : {'default': None, 'val_type': 'sigma', 'val_range': {'u': [ (22.4, 0.1) ],
                                                                         'g': [ (23.4, 0.1) ],
                                                                         'q': [ (23.9, 0.1) ],
                                                                         'r': [ (22.9, 0.1) ],
                                                                         'i': [ (22.4, 0.1) ],
                                                                         'z': [ (21.4, 0.1) ]},     'cat_type': 'all', 'comment': '[mag] zeropoint=m_AB+2.5*log10(flux[e-/s])+A*k'},
        'PC-ZPSTD': {'default': None, 'val_type': 'sigma', 'val_range': {'u': [ (0.06, 0.02) ],
                                                                         'g': [ (0.03, 0.01) ],
                                                                         'q': [ (0.02, 0.01) ],
                                                                         'r': [ (0.02, 0.01) ],
                                                                         'i': [ (0.02, 0.01) ],
                                                                         'z': [ (0.02, 0.01) ]},    'cat_type': 'all', 'comment': '[mag] sigma (STD) zeropoint sigma'},
        # needed if PC-ZP is already being checked? 
        # N.B.: these limmags below are assuming 5 sigma, as set by source_nsigma in ZOGY settings file
        # if that 5 sigma changes, these number need updating with correction: -2.5*log10(nsigma/5)!
        'LIMMAG'  : {'default': None, 'val_type': 'sigma', 'val_range': {'u': [ (19.1, 0.4) ],
                                                                         'g': [ (20.3, 0.4) ],
                                                                         'q': [ (20.6, 0.5) ],
                                                                         'r': [ (20.0, 0.2) ],
                                                                         'i': [ (19.4, 0.4) ],
                                                                         'z': [ (18.3, 0.3) ]},     'cat_type': 'all', 'comment': '[mag] full-frame 5-sigma limiting magnitude'},
        
        # Transients
        'Z-DX'    : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.02) ],              'cat_type': 'trans', 'comment': '[pix] dx median offset full image'},
        'Z-DY'    : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.02) ],              'cat_type': 'trans', 'comment': '[pix] dy median offset full image'},
        'Z-DXSTD' : {'default': None, 'val_type': 'sigma', 'val_range': [ (0.1, 0.04) ],            'cat_type': 'trans', 'comment': '[pix] dx sigma (STD) offset full image'},
        'Z-DYSTD' : {'default': None, 'val_type': 'sigma', 'val_range': [ (0.1, 0.04) ],            'cat_type': 'trans', 'comment': '[pix] dy sigma (STD) offset full image'},
        'Z-FNR'   : {'default': None, 'val_type': 'min_max', 'val_range': [ (0.2, 5), (0.1, 10), (0.05, 20) ], 'cat_type': 'trans', 'comment': 'median flux ratio (Fnew/Fref) full image'},
        'Z-FNRSTD': {'default': None, 'val_type': 'sigma', 'val_range': [ (0.05, 0.05) ],           'cat_type': 'trans', 'comment': 'sigma (STD) flux ratio (Fnew/Fref) full image'},
        'Z-SCMED' : {'default': None, 'val_type': 'sigma', 'val_range': [ (0, 0.1) ],               'cat_type': 'trans', 'comment': 'median Scorr full image'},
        'Z-SCSTD' : {'default': None, 'val_type': 'sigma', 'val_range': [ (1.2, 0.1) ],             'cat_type': 'trans', 'comment': 'sigma (STD) Scorr full image'},
        'T-NTRANS': {'default': None, 'val_type': 'sigma', 'val_range': [ (100, 200) ],             'cat_type': 'trans', 'comment': 'no. of significant transients (pre-vetting)'},
        'T-LMAG' :  {'default': None, 'val_type': 'sigma', 'val_range': {'u': [ (18.8, 0.3) ],
                                                                         'g': [ (20.0, 0.3) ],
                                                                         'q': [ (20.1, 0.4) ],
                                                                         'r': [ (19.7, 0.2) ],
                                                                         'i': [ (19.1, 0.4) ],
                                                                         'z': [ (17.9, 0.3) ]},     'cat_type': 'trans', 'comment': '[mag] full-frame transient [T-NSIGMA]-sigma limiting mag'},
        #
        # some additional ones to make sure these are listed in the dummy output catalogs
        'CCD-TEMP': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Current CCD temperature'},
        'RA-REF':   {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Requested right ascension'},
        'DEC-REF':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Requested declination'},
        'FLIPSTAT': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Telescope side of the pier'},
        'CL-BASE':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[m] Reinhardt cloud base altitude'},
        'RH-MAST':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Vaisala RH mast'},
        'PRESSURE': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[hPa] Vaisala pressure mast'},
        'T-ROOF':   {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Reinhardt temperature roof'},
        'T-MAST':   {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Vaisala temperature mast'},
        'T-STRUT':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature carbon strut between M1 and M2'},
        'T-CRING':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature main carbon ring around M1'},
        'T-SPIDER': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature carbon spider above M2'},
        'T-FWN':    {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature filter wheel housing North'},
        'T-FWS':    {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature filter wheel housing South'},
        'T-M2HOLD': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature aluminium M2 holder'},
        'T-GUICAM': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature guide camera'},
        'T-M1':     {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[C] Temperature backside M1'},
        'WINDAVE':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[km/h] Vaisala wind speed mast'},
        'WINDGUST': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[km/h] Vaisala wind gust mast'},
        'WINDDIR':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[deg] Vaisala wind direction mast'},

        'SITELAT':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[deg] Site latitude'},
        'SITELONG': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[deg] Site longitude'},
        'ELEVATIO': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[m] Site elevation'},
        'CCD-ID':   {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'CCD camera ID'},
        'CONTROLL': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'CCD controller'},
        'DETSPEED': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[kHz] Detector read speed'},
        'INSTRUME': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Instrument name'},
        'FOCUSPOS': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[micron] Focuser position'},
        
        'OBSERVER': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Robotic observations software and PC ID'},
        'ABOTVER':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'ABOT version'},
        'PROGNAME': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Program name'},
        'PROGID':   {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Program ID'},
        'GUIDERST': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Guider status'},
        'GUIDERFQ': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[Hz] Guide loop frequency'},
        'TRAKTIME': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[s] Autoguider exposure time during imaging'},
        'ADCX':     {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[mm] Position offset ADC lens in x'},
        'ADCY':     {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[mm] Position offset ADC lens in y'}, 
        
        'REDFILE':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'BlackBOX reduced image name'},
        'MASKFILE': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'BlackBOX mask image name'},
                
        'PSF-SIZE': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[pix] size PSF image'},
        'PSF-CFGS': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'size PSF config. image (= PSF-SIZE / PSF-SAMP)'},
        
        'PC-EXTCO': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[mag] filter extinction coefficient (k) used'},
        'AIRMASSC': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'Airmass at image center'},
        'RA-CNTR':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'RA (ICRS) at image center (astrometry.net)'},
        'DEC-CNTR': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'DEC (ICRS) at image center (astrometry.net)'},
        'LIMFLUX':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': '[e-/s] full-frame n-sigma limiting flux'},

        'DUMMYCAT': {'default': False, 'val_type': 'skip', 'val_range': None, 'cat_type': 'all',   'comment': 'is this a dummy catalog without actual sources?'},
        'QC-FLAG':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'all',   'comment': 'QC flag color (green|yellow|orange|red)'},

        'Z-FPEMED': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'trans', 'comment': '[e-/s] median Fpsferr full image'},
        'Z-FPESTD': {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'trans', 'comment': '[e-/s] sigma (STD) Fpsferr full image'},
        'T-LFLUX':  {'default': None, 'val_type': 'skip',  'val_range': None, 'cat_type': 'trans', 'comment': '[e-/s] full-frame transient [T-NSIGMA]-sigma limit. flux'},
        #
        
    }
}

