
"""Dictionary of allowed ranges of header keyword values to be used
   for the automatic data quality control (QC) of MeerLICHT and
   BlackGEM data. This dictionary is used by the [qc_check] function
   in the [qc] module, which will assign quality flags to a list of
   header keywords or a full header.

   Each telescope (ML1, BG2, etc.) can have its own set of QC ranges,
   and the telescope is set with the initial dictionary key.  The
   value corresponding to the header key in the telescope qc_range
   dictionary is also a dictionary with the keys:

         1) 'default': the default value that will be used in case
            the keyword is not present in the input header and a dummy 
            catalog is being created
         2) 'val_type': providing the type of range provided
         3) 'val_range': a list of tuples, each containing either one
            (for type='bool') or two values (for the other types).
            The filter-specific ranges can be set by making 'val_range'
            a dictionary with the filters as keys.
         4) 'key_type': makes distinction between the keywords related
            to the full-source catalog ('full') and the transient 
            catalog ('trans').
         5) 'comment': header comment / brief description of keyword


   Depending on 'val_type', these values are interpreted differently:

         1) 'min_max': (C1, C2) such that C1 <= value <= C2
         2) 'bool': (C) such that value==C or value==C2
         3) 'sigma': (E, STD) such that abs(value-E) <= n*STD,
                     where n is a list of predefined factors
                     corresponding to the accepted ranges.
                     currently: n_std = [2, 4, 7]
         4) 'exp_abs': (E, C) such that abs(value-E) <= C
         5) 'exp_frac': (E, f) such that abs((value-E)/E) <= f 
         6) 'skip': keyword is not considered

   The value of each keyword is first checked against the first
   element of the 'val_range' key. If the value is within this range,
   the key gets a 'green' flag. If it is not within this range, it
   checks the value against the 2nd range provided.  If there is no
   2nd range provided, the key gets a 'red' flag.  If the value is
   within the 2nd range, the corresponding flag will be 'yellow'. The
   3rd range corresponds to the 'orange' flag, and if the keyword
   value is not within any of the ranges provided, it gets a 'red'
   flag.

   For the value type 'sigma' only the expected value (E) and a
   standard deviation needs to be provided, and these are expanded to
   three ranges using: n_std = [2, 4, 7]. So if a keyword value is
   within n_std * STD, its color flag will be 'green', 'yellow' and
   'orange', respectively. If outside of this, it will be flagged
   'red'.

"""

qc_range = {
    'ML1': {

        # 'raw' image header keywords
        'GPS-SHUT': {'default':'None', 'val_type': 'min_max','val_range': [ (0.85,0.89), (0.8,0.94), (-1e3,1e3) ],'key_type': 'full', 'pos': False, 'comment': '[s] Shutter time:(GPSEND-GPSSTART)-EXPTIME'},

        # Main processing steps
        'XTALK-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'corrected for crosstalk?'},
        'NONLIN-P': {'default': False, 'val_type': 'bool', 'val_range': [ False ],                   'key_type': 'full', 'pos': False, 'comment': 'corrected for non-linearity?'},
        'GAIN-P'  : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'corrected for gain?'},
        'OS-P'    : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'corrected for overscan?'},
        'MBIAS-P' : {'default': False, 'val_type': 'bool', 'val_range': [ False ],                   'key_type': 'full', 'pos': False, 'comment': 'corrected for master bias?'},
        'MBIAS-F' : {'default':'None', 'val_type': 'skip', 'val_range': None,                        'key_type': 'full', 'pos': False, 'comment': 'name of master bias applied'},
        'MFLAT-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'corrected for master flat?'},
        'MFLAT-F' : {'default':'None', 'val_type': 'skip', 'val_range': None,                        'key_type': 'full', 'pos': False, 'comment': 'name of master flat applied'},
        'MFRING-P': {'default': False, 'val_type': 'bool', 'val_range': {'u': [ False ],
                                                                         'g': [ False ],
                                                                         'q': [ False ],
                                                                         'r': [ False ],
                                                                         'i': [ False ],
                                                                         'z': [ True, False ]},      'key_type': 'full', 'pos': False, 'comment': 'corrected for master fringe map?'},        
        'MFRING-F': {'default':'None', 'val_type': 'skip', 'val_range': None,                        'key_type': 'full', 'pos': False, 'comment': 'name of master fringe map applied'},
        'COSMIC-P': {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'corrected for cosmics rays?'},
        'SAT-P'   : {'default': False, 'val_type': 'bool', 'val_range': [ True, False ],             'key_type': 'full', 'pos': False, 'comment': 'processed for satellite trails?'},
        'S-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'successfully processed by SExtractor?'},
        'A-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'successfully processed by Astrometry.net?'},
        'PSF-P'   : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'successfully processed by PSFEx?'},
        'PC-P'    : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'full', 'pos': False, 'comment': 'successfully processed by phot. calibration?'},
        'SWARP-P' : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'trans', 'pos': False, 'comment': 'reference image successfully SWarped?'},
        'Z-P'     : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'trans', 'pos': False, 'comment': 'successfully processed by ZOGY?'},
        'MC-P'    : {'default': False, 'val_type': 'bool', 'val_range': [ True ],                    'key_type': 'trans', 'pos': False, 'comment': 'successfully processed by MeerCRAB?'},


        # Channel bias levels [e-]
        # 2019 values
        #'BIASMEAN': {'default':'None', 'val_type': 'sigma','val_range': [ (  7370, 30) ],            'key_type': 'full', 'pos': True , 'comment': 'average all channel means vertical overscan'},
        # 2021 values; multiplied above values by gain_new / gain_old, or 0.88:
        'BIASMEAN': {'default':'None', 'val_type': 'sigma','val_range': [ (  6450, 50) ],            'key_type': 'full', 'pos': True , 'comment': 'average all channel means vertical overscan'},
        

        # for the moment, skip the value range check on the individual channels' bias levels ('sigma' replaced with 'skip')
        'BIASM1'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  6933.564,   32.281) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 1 mean vertical overscan'},
        'BIASM2'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7199.254,   34.481) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 2 mean vertical overscan'},
        'BIASM3'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7291.843,   31.315) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 3 mean vertical overscan'},
        'BIASM4'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7384.878,   30.259) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 4 mean vertical overscan'},
        'BIASM5'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7262.722,   29.910) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 5 mean vertical overscan'},
        'BIASM6'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7275.950,   30.754) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 6 mean vertical overscan'},
        'BIASM7'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7447.558,   31.199) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 7 mean vertical overscan'},
        'BIASM8'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7169.434,   28.927) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 8 mean vertical overscan'},
        'BIASM9'  : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7011.460,   31.531) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 9 mean vertical overscan'},
        'BIASM10' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7500.022,   32.602) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 10 mean vertical overscan'},
        'BIASM11' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7307.696,   29.695) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 11 mean vertical overscan'},
        'BIASM12' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7334.698,   32.213) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 12 mean vertical overscan'},
        'BIASM13' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7460.912,   27.949) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 13 mean vertical overscan'},
        'BIASM14' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7591.438,   26.561) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 14 mean vertical overscan'},
        'BIASM15' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7567.986,   31.364) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 15 mean vertical overscan'},
        'BIASM16' : {'default':'None', 'val_type': 'skip', 'val_range': [ (  7600.082,   34.135) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 16 mean vertical overscan'},
        # 2017 values
        #'BIASMEAN': {'val_type': 'sigma', 'val_range': [ (  7101.809,   29.768) ], 'key_type': 'full', 'pos': True , 'comment': 'average all channel means vertical overscan'},
        #'BIASM1'  : {'val_type': 'sigma', 'val_range': [ (  6915.888,   33.458) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 1 mean vertical overscan'},
        #'BIASM2'  : {'val_type': 'sigma', 'val_range': [ (  7012.454,   34.318) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 2 mean vertical overscan'},
        #'BIASM3'  : {'val_type': 'sigma', 'val_range': [ (  7023.386,   32.152) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 3 mean vertical overscan'},
        #'BIASM4'  : {'val_type': 'sigma', 'val_range': [ (  7052.668,   31.735) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 4 mean vertical overscan'},
        #'BIASM5'  : {'val_type': 'sigma', 'val_range': [ (  7125.328,   33.090) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 5 mean vertical overscan'},
        #'BIASM6'  : {'val_type': 'sigma', 'val_range': [ (  7142.417,   33.233) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 6 mean vertical overscan'},
        #'BIASM7'  : {'val_type': 'sigma', 'val_range': [ (  7169.577,   30.164) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 7 mean vertical overscan'},
        #'BIASM8'  : {'val_type': 'sigma', 'val_range': [ (  7077.317,   30.216) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 8 mean vertical overscan'},
        #'BIASM9'  : {'val_type': 'sigma', 'val_range': [ (  6905.847,   31.534) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 9 mean vertical overscan'},
        #'BIASM10' : {'val_type': 'sigma', 'val_range': [ (  7129.008,   29.231) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 10 mean vertical overscan'},
        #'BIASM11' : {'val_type': 'sigma', 'val_range': [ (  7120.504,   27.446) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 11 mean vertical overscan'},
        #'BIASM12' : {'val_type': 'sigma', 'val_range': [ (  7213.597,   28.497) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 12 mean vertical overscan'},
        #'BIASM13' : {'val_type': 'sigma', 'val_range': [ (  7226.578,   28.769) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 13 mean vertical overscan'},
        #'BIASM14' : {'val_type': 'sigma', 'val_range': [ (  7275.009,   25.343) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 14 mean vertical overscan'},
        #'BIASM15' : {'val_type': 'sigma', 'val_range': [ (  7101.251,   28.866) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 15 mean vertical overscan'},
        #'BIASM16' : {'val_type': 'sigma', 'val_range': [ (  7154.681,   30.773) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 16 mean vertical overscan'},

        # Channel read noise [e-]
        # 2019 values
        'RDNOISE' : {'default':'None', 'val_type': 'min_max','val_range': [ (5,11), (5,13), (5,15) ], 'key_type': 'full', 'pos': True , 'comment': 'average all channel sigmas vertical overscan'},
        # for the moment, skip the value range check on the individual channels' readnoise ('sigma' replaced with 'skip')
        'RDN1'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    13.222,    0.230) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 1 sigma (STD) vertical overscan'},
        'RDN2'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (     7.853,    0.144) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 2 sigma (STD) vertical overscan'},
        'RDN3'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    13.436,    0.202) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 3 sigma (STD) vertical overscan'},
        'RDN4'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    12.608,    0.190) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 4 sigma (STD) vertical overscan'},
        'RDN5'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    12.566,    0.199) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 5 sigma (STD) vertical overscan'},
        'RDN6'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    12.234,    0.253) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 6 sigma (STD) vertical overscan'},
        'RDN7'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (     7.816,    0.166) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 7 sigma (STD) vertical overscan'},
        'RDN8'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    14.106,    0.254) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 8 sigma (STD) vertical overscan'},
        'RDN9'    : {'default':'None', 'val_type': 'skip', 'val_range': [ (    11.921,    0.178) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 9 sigma (STD) vertical overscan'},
        'RDN10'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (     7.998,    0.159) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 10 sigma (STD) vertical overscan'},
        'RDN11'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (    10.896,    0.195) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 11 sigma (STD) vertical overscan'},
        'RDN12'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (     9.342,    0.171) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 12 sigma (STD) vertical overscan'},
        'RDN13'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (    14.306,    0.246) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 13 sigma (STD) vertical overscan'},
        'RDN14'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (    14.110,    0.261) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 14 sigma (STD) vertical overscan'},
        'RDN15'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (     9.419,    0.184) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 15 sigma (STD) vertical overscan'},
        'RDN16'   : {'default':'None', 'val_type': 'skip', 'val_range': [ (     8.231,    0.158) ],  'key_type': 'full', 'pos': True , 'comment': 'channel 16 sigma (STD) vertical overscan'},     
        ## 2017 values
        #'RDNOISE' : {'val_type': 'sigma', 'val_range': [ (    10.454,    0.730) ], 'key_type': 'full', 'pos': True , 'comment': 'average all channel sigmas vertical overscan'},
        #'RDN1'    : {'val_type': 'sigma', 'val_range': [ (    11.643,    0.859) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 1 sigma (STD) vertical overscan'},
        #'RDN2'    : {'val_type': 'sigma', 'val_range': [ (     8.943,    0.327) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 2 sigma (STD) vertical overscan'},
        #'RDN3'    : {'val_type': 'sigma', 'val_range': [ (    11.440,    0.816) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 3 sigma (STD) vertical overscan'},
        #'RDN4'    : {'val_type': 'sigma', 'val_range': [ (    11.696,    0.666) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 4 sigma (STD) vertical overscan'},
        #'RDN5'    : {'val_type': 'sigma', 'val_range': [ (    11.629,    0.653) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 5 sigma (STD) vertical overscan'},
        #'RDN6'    : {'val_type': 'sigma', 'val_range': [ (    11.581,    0.578) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 6 sigma (STD) vertical overscan'},
        #'RDN7'    : {'val_type': 'sigma', 'val_range': [ (     9.269,    0.398) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 7 sigma (STD) vertical overscan'},
        #'RDN8'    : {'val_type': 'sigma', 'val_range': [ (    12.045,    0.963) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 8 sigma (STD) vertical overscan'},
        #'RDN9'    : {'val_type': 'sigma', 'val_range': [ (     9.206,    0.391) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 9 sigma (STD) vertical overscan'},
        #'RDN10'   : {'val_type': 'sigma', 'val_range': [ (     9.038,    0.361) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 10 sigma (STD) vertical overscan'},
        #'RDN11'   : {'val_type': 'sigma', 'val_range': [ (    10.094,    0.636) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 11 sigma (STD) vertical overscan'},
        #'RDN12'   : {'val_type': 'sigma', 'val_range': [ (     9.479,    0.417) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 12 sigma (STD) vertical overscan'},
        #'RDN13'   : {'val_type': 'sigma', 'val_range': [ (    10.818,    0.861) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 13 sigma (STD) vertical overscan'},
        #'RDN14'   : {'val_type': 'sigma', 'val_range': [ (    11.382,    0.978) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 14 sigma (STD) vertical overscan'},
        #'RDN15'   : {'val_type': 'sigma', 'val_range': [ (     9.258,    0.412) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 15 sigma (STD) vertical overscan'},
        #'RDN16'   : {'val_type': 'sigma', 'val_range': [ (     9.044,    0.373) ], 'key_type': 'full', 'pos': True , 'comment': 'channel 16 sigma (STD) vertical overscan'},

        # master bias (these keywords should not end up in dummy catalogs: keytype should not be equal to 'full' or 'trans')
        'NBIAS'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (10,50), (7,9), (5,6) ], 'key_type': 'mbias', 'pos': True , 'comment': 'number of bias frames combined'},
        'MBMEAN'  : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 5) ],                'key_type': 'mbias', 'pos': False, 'comment': '[e-] mean master bias'},
        'MBRDN'   : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 5) ],                'key_type': 'mbias', 'pos': True , 'comment': '[e-] sigma (STD) master bias'},

        # individual flats (these keywords should not end up in dummy catalogs: keytype should not be equal to 'full' or 'trans')
        'MEDSEC'  : {'default':'None', 'val_type': 'min_max', 'val_range': [ (2.15*20e3, 2.15*30e3), (2.15*15e3, 2.15*35e3), (2.15*10e3, 2.15*40e3) ], 'key_type': 'flat', 'pos': True , 'comment': '[e-] median flat over STATSEC (bias-subtracted)'},
        #'RSTDSEC' : {'default':'None', 'val_type': 'sigma', 'val_range': [ (0, 0.01) ],              'key_type': None, 'pos': True , 'comment': 'relative sigma (STD) flat over STATSEC'},
        #'FLATRSTD': {'default':'None', 'val_type': 'sigma', 'val_range': [ (0,0.025),(0,0.026),(0,0.027)], 'key_type': None, 'pos': True , 'comment': 'relative sigma (STD) flat'},

        'RDIF-MAX': {'default':'None', 'val_type': 'min_max', 'val_range': {'u': [ (0, 0.028), (0, 0.029), (0, 0.030) ],
                                                                            'g': [ (0, 0.012), (0, 0.013), (0, 0.014) ],
                                                                            'q': [ (0, 0.013), (0, 0.014), (0, 0.015) ],
                                                                            'r': [ (0, 0.013), (0, 0.014), (0, 0.015) ],
                                                                            'i': [ (0, 0.015), (0, 0.0165),(0, 0.018) ],
                                                                            'z': [ (0, 0.025), (0, 0.026), (0, 0.027) ]}, 'key_type': 'flat', 'pos': True , 'comment': '(max(subs)-min(subs)) / (max(subs)+min(subs))'},

        'RSTD-MAX': {'default':'None', 'val_type': 'min_max', 'val_range': {'u': [ (0, 0.078) ],
                                                                            'g': [ (0, 0.066) ],
                                                                            'q': [ (0, 0.058) ],
                                                                            'r': [ (0, 0.047) ],
                                                                            'i': [ (0, 0.028) ],
                                                                            'z': [ (0, 0.024) ]},                'key_type': 'flat', 'pos': True , 'comment': 'max. relative sigma (STD) of subimages'},

        # master flat (these keywords should not end up in dummy catalogs: keytype should not be equal to 'full' or 'trans')
        'NFLAT'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (6,50), (4,5), (3,3) ],            'key_type': 'mflat', 'pos': True , 'comment': 'number of flat frames combined'},
        'MFMEDSEC': {'default':'None', 'val_type': 'sigma',   'val_range': [ (         1,  0.001) ],            'key_type': 'mflat', 'pos': False, 'comment': 'median master flat over STATSEC'},
        'MFSTDSEC': {'default':'None', 'val_type': 'sigma',   'val_range': [ (         0,   0.01) ],            'key_type': 'mflat', 'pos': True , 'comment': 'sigma (STD) master flat over STATSEC'},
        'FLATDITH': {'default':'None', 'val_type': 'bool',    'val_range': [ True ],                            'key_type': 'mflat', 'pos': False, 'comment': 'majority of flats were dithered'},

        # general
        'AIRMASS' : {'default':'None', 'val_type': 'min_max', 'val_range': [ (1,2), (2,2.5), (2.5, 2.95) ],     'key_type': 'full', 'pos': True , 'comment': 'Airmass (based on RA, DEC, DATE-OBS)'},
        'N-INFNAN': {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,0), (1,10), (11,1e6) ],         'key_type': 'full', 'pos': True , 'comment': 'number of pixels with infinite/nan values'},

        # cosmics/satellites
        'NCOSMICS': {'default':'None', 'val_type': 'min_max', 'val_range': [ (3,50), (2,100), (0,500) ],        'key_type': 'full', 'pos': True , 'comment': '[/s] number of cosmic rays identified'},
        'NSATS'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,10), (10,20), (20,100) ],         'key_type': 'full', 'pos': True , 'comment': 'number of satellite trails identified'},

        # SExtractor
        'S-NOBJ'  : {'default':'None', 'val_type': 'skip',    'val_range': [ (4e3,1e4), (3e3,2e5), (1e3,1e7) ], 'key_type': 'full', 'pos': True , 'comment': 'number of objects detected by SExtractor'},
        'NOBJECTS': {'default':'None', 'val_type': 'min_max', 'val_range': [ (4e3,1e5), (1e3,3e5), (1e2,1e7) ], 'key_type': 'full', 'pos': True , 'comment': 'number of >= [NSIGMA]-sigma objects'},
        'S-SEEING': {'default':'None', 'val_type': 'min_max', 'val_range': [ (1.5,4), (1,5), (0.5,7) ],         'key_type': 'full', 'pos': True , 'comment': '[arcsec] SExtractor seeing estimate'},
        'S-SEESTD': {'default':'None', 'val_type': 'skip',    'val_range':  {'u': [ (0.1,0.3) ],
                                                                             'g': [ (0.1,0.1) ],
                                                                             'q': [ (0.1,0.1) ],
                                                                             'r': [ (0.1,0.1) ],
                                                                             'i': [ (0.1,0.1) ],
                                                                             'z': [ (0.1,0.1) ]},               'key_type': 'full', 'pos': True , 'comment': '[arcsec] sigma (STD) SExtractor seeing'},
        'S-ELONG' : {'default':'None', 'val_type': 'sigma',   'val_range': [ (1.1,0.2) ],                       'key_type': 'full', 'pos': True , 'comment': 'SExtractor ELONGATION (A/B) estimate'},
        'S-ELOSTD': {'default':'None', 'val_type': 'skip',    'val_range': [ (0.04,0.04) ],                     'key_type': 'full', 'pos': True , 'comment': 'sigma (STD) SExtractor ELONGATION (A/B)'},
        'S-BKG'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,5e2), (0,5e3), (0,5e4) ],       'key_type': 'full', 'pos': False, 'comment': '[e-] median background full image'},
        'S-BKGSTD': {'default':'None', 'val_type': 'skip',    'val_range': [ (15,10) ],                         'key_type': 'full', 'pos': True , 'comment': '[e-] sigma (STD) background full image'},

        # Astrometry.net        
        'A-PSCALE': {'default':'None', 'val_type': 'sigma',   'val_range': [ (0.5642, 0.0001) ],                'key_type': 'full', 'pos': True , 'comment': '[arcsec/pix] pixel scale WCS solution'},
        'A-ROT'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (-91,-89), (-93,-87), (-180,180) ], 'key_type': 'full', 'pos': False, 'comment': '[deg] rotation WCS solution (E of N for "up")'},

        'A-CAT-F' : {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full', 'pos': False, 'comment': 'astrometric catalog'},
        'A-NAST'  : {'default':'None', 'val_type': 'min_max', 'val_range': [ (5e2,1e4), (100, 3e4), (20, 1e5) ],'key_type': 'full', 'pos': True , 'comment': 'number of brightest stars used for WCS'},
        'A-DRA'   : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.02)],                       'key_type': 'full', 'pos': False, 'comment': '[arcsec] dRA median offset to astrom. catalog'},
        'A-DDEC'  : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.02)],                       'key_type': 'full', 'pos': False, 'comment': '[arcsec] dDEC median offset to astrom. catalog'},
        'A-DRASTD': {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (0.06, 0.04) ], 
                                                                            'g': [ (0.03, 0.02) ], 
                                                                            'q': [ (0.03, 0.02) ], 
                                                                            'r': [ (0.03, 0.02) ], 
                                                                            'i': [ (0.03, 0.02) ], 
                                                                            'z': [ (0.03, 0.02) ]},             'key_type': 'full', 'pos': True , 'comment': '[arcsec] dRA sigma (STD) offset'},

        'A-DDESTD': {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (0.06, 0.04) ],
                                                                            'g': [ (0.03, 0.02) ], 
                                                                            'q': [ (0.03, 0.02) ], 
                                                                            'r': [ (0.03, 0.02) ], 
                                                                            'i': [ (0.03, 0.02) ], 
                                                                            'z': [ (0.03, 0.02) ]},             'key_type': 'full', 'pos': True , 'comment': '[arcsec] dDEC sigma (STD) offset'},

        # PSFEx
        'PSF-NOBJ': {'default':'None', 'val_type': 'min_max', 'val_range': [ (500,2e4), (100,5e4), (10,2e5) ],  'key_type': 'full', 'pos': True , 'comment': 'number of accepted PSF stars'},
        'PSF-CHI2': {'default':'None', 'val_type': 'sigma',   'val_range': [ (1, 0.1) ],                        'key_type': 'full', 'pos': True , 'comment': 'final reduced chi-squared PSFEx fit'},
        'PSF-SEE' : {'default':'None', 'val_type': 'min_max', 'val_range': [ (1.5,4), (1,5), (0.5,7) ],         'key_type': 'full', 'pos': True , 'comment': '[arcsec] image seeing inferred by PSFEx'},

        # photometric calibration (PC)
        'PC-CAT-F': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full', 'pos': False, 'comment': 'photometric catalog'},
        'PC-NCAL' : {'default':'None', 'val_type': 'min_max', 'val_range': [ (50, 1e3), (20, 1e4), (5,1e5) ],   'key_type': 'full', 'pos': True , 'comment': 'number of brightest photcal stars used'},

        'PC-ZP'   : {'default':'None', 'val_type': 'min_max', 'val_range': {'u': [ (22.1, 22.7), (21.4, 23.4), (0, 30) ],
                                                                            'g': [ (23.0, 23.6), (22.3, 24.3), (0, 30) ],
                                                                            'q': [ (23.5, 24.1), (22.8, 24.8), (0, 30) ],
                                                                            'r': [ (22.6, 23.2), (21.9, 23.9), (0, 30) ],
                                                                            'i': [ (22.0, 22.6), (21.3, 23.3), (0, 30) ],
                                                                            'z': [ (21.1, 21.7), (20.4, 22.4), (0, 30) ]}, 'key_type': 'full', 'pos': True , 'comment': '[mag] zeropoint=m_AB+2.5*log10(flux[e-/s])+A*k'},
        # previously using sigma method
        #'PC-ZP'   : {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (22.4, 0.15) ],
        #                                                                    'g': [ (23.3, 0.15) ],
        #                                                                    'q': [ (23.8, 0.15) ],
        #                                                                    'r': [ (22.9, 0.15) ],
        #                                                                    'i': [ (22.3, 0.15) ],
        #                                                                    'z': [ (21.4, 0.15) ]},              'key_type': 'full', 'pos': True , 'comment': '[mag] zeropoint=m_AB+2.5*log10(flux[e-/s])+A*k'},

        'PC-ZPSTD': {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (0.07, 0.03) ], 
                                                                            'g': [ (0.03, 0.03) ], 
                                                                            'q': [ (0.02, 0.03) ], 
                                                                            'r': [ (0.02, 0.03) ], 
                                                                            'i': [ (0.02, 0.03) ], 
                                                                            'z': [ (0.03, 0.03) ]},             'key_type': 'full', 'pos': True , 'comment': '[mag] sigma (STD) zeropoint sigma'},

        # updated PC-MZPD values and also ZPSTD above to estimates based on uqi data set from Simon on GW190814; increased because of variation in crowded fields such as SMC (16000)
        'PC-MZPD' : {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (0.07, 0.09) ],
                                                                            'g': [ (0.03, 0.09) ],
                                                                            'q': [ (0.02, 0.09) ],
                                                                            'r': [ (0.02, 0.09) ],
                                                                            'i': [ (0.02, 0.09) ],
                                                                            'z': [ (0.03, 0.09) ]},             'key_type': 'full', 'pos': True , 'comment': '[mag] maximum zeropoint difference between subimages'},

        'PC-MZPS' : {'default':'None', 'val_type': 'skip',    'val_range': {'u': [ (0.01, 0.02) ],
                                                                            'g': [ (0.01, 0.02) ],
                                                                            'q': [ (0.01, 0.02) ],
                                                                            'r': [ (0.01, 0.02) ],
                                                                            'i': [ (0.01, 0.02) ],
                                                                            'z': [ (0.01, 0.02) ]},             'key_type': 'full', 'pos': True , 'comment': '[mag] maximum zeropoint sigma (STD) of subimages'},

        # N.B.: these limmags below are assuming 5 sigma, as set by source_nsigma in ZOGY settings file
        # if that 5 sigma changes, these number need updating with correction: -2.5*log10(nsigma/5)!
        'LIMMAG'  : {'default':'None', 'val_type': 'min_max', 'val_range': {'u': [ (18.9, 22.2), (18.2, 22.2), (0, 30) ],
                                                                            'g': [ (20.0, 23.3), (19.3, 23.3), (0, 30) ],
                                                                            'q': [ (20.5, 23.9), (19.8, 23.9), (0, 30) ],
                                                                            'r': [ (19.8, 23.1), (19.1, 23.1), (0, 30) ],
                                                                            'i': [ (19.2, 22.5), (18.5, 22.5), (0, 30) ],
                                                                            'z': [ (18.0, 21.3), (17.3, 21.3), (0, 30) ]}, 'key_type': 'full', 'pos': True , 'comment': '[mag] full-frame 5-sigma limiting mag'},
        # previously using sigma method
        #'LIMMAG'  : {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (19.2, 0.15) ],
        #                                                                    'g': [ (20.3, 0.15) ],
        #                                                                    'q': [ (20.8, 0.15) ],
        #                                                                    'r': [ (20.1, 0.15) ],
        #                                                                    'i': [ (19.5, 0.15) ],
        #                                                                    'z': [ (18.3, 0.15) ]},              'key_type': 'full', 'pos': True , 'comment': '[mag] full-frame 5-sigma limiting magnitude'},


        # check on offset between RA-CNTR, DEC-CNTR and the RA, DEC corresponding to the ML/BG field definition for a particular OBJECT or field ID
        'RADECOFF': {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.15) ],                       'key_type': 'full',   'pos': True , 'comment': '[deg] offset RA,DEC-CNTR wrt ML/BG field grid'},

        
        
        # Transients
        'Z-DX'    : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.04) ],                       'key_type': 'trans', 'pos': False, 'comment': '[pix] dx median offset full image'},
        'Z-DY'    : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.04) ],                       'key_type': 'trans', 'pos': False, 'comment': '[pix] dy median offset full image'},
        'Z-DXSTD' : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0.1, 0.1) ],                      'key_type': 'trans', 'pos': True , 'comment': '[pix] dx sigma (STD) offset full image'},
        'Z-DYSTD' : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0.1, 0.1) ],                      'key_type': 'trans', 'pos': True , 'comment': '[pix] dy sigma (STD) offset full image'},
        'Z-FNR'   : {'default':'None', 'val_type': 'min_max', 'val_range': [ (0.7, 1.3), (0.4, 2.5), (0.06, 15) ],'key_type': 'trans', 'pos': True , 'comment': 'median flux ratio (Fnew/Fref) full image'},
        'Z-FNRSTD': {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (0.06, 0.03) ], 
                                                                            'g': [ (0.03, 0.03) ], 
                                                                            'q': [ (0.03, 0.03) ], 
                                                                            'r': [ (0.03, 0.03) ], 
                                                                            'i': [ (0.03, 0.03) ], 
                                                                            'z': [ (0.03, 0.03) ]},             'key_type': 'trans', 'pos': True , 'comment': 'sigma (STD) flux ratio (Fnew/Fref) full image'},

        'Z-SCMED' : {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.30) ],                       'key_type': 'trans', 'pos': False, 'comment': 'median Scorr full image'},
        'Z-SCSTD' : {'default':'None', 'val_type': 'sigma',   'val_range': [ (1, 0.15) ],                       'key_type': 'trans', 'pos': True , 'comment': 'sigma (STD) Scorr full image'},
        'T-NTRANS': {'default':'None', 'val_type': 'skip',    'val_range': [ (100, 200)],                       'key_type': 'trans', 'pos': True , 'comment': 'number of >= [T-NSIGMA]-sigma transients (pre-vetting)'},
        'T-FTRANS': {'default':'None', 'val_type': 'sigma',   'val_range': [ (0, 0.03) ],                       'key_type': 'trans', 'pos': True , 'comment': 'transient fraction: T-NTRANS / NOBJECTS'},

        # N.B.: these limmags below are assuming 6 sigma, as set by transient_nsigma in ZOGY settings file        
        # if that 6 sigma changes, these number need updating with correction: -2.5*log10(nsigma/6)!
        'T-LMAG' :  {'default':'None', 'val_type': 'min_max', 'val_range': {'u': [ (18.7, 22.0), (18.0, 22.0), (0, 30) ],
                                                                            'g': [ (19.8, 23.1), (19.1, 23.1), (0, 30) ],
                                                                            'q': [ (20.3, 23.6), (19.6, 23.6), (0, 30) ],
                                                                            'r': [ (19.6, 22.9), (18.9, 22.9), (0, 30) ],
                                                                            'i': [ (19.0, 22.3), (18.3, 22.3), (0, 30) ],
                                                                            'z': [ (17.9, 21.2), (17.2, 21.2), (0, 30) ]}, 'key_type': 'trans', 'pos': True , 'comment': '[mag] full-frame transient [T-NSIGMA]-sigma lim. mag'},
        # previously using sigma method:
        #'T-LMAG' :  {'default':'None', 'val_type': 'sigma',   'val_range': {'u': [ (19.0, 0.15) ],
        #                                                                    'g': [ (20.1, 0.15) ],
        #                                                                    'q': [ (20.6, 0.15) ],
        #                                                                    'r': [ (19.9, 0.15) ],
        #                                                                    'i': [ (19.3, 0.15) ],
        #                                                                    'z': [ (18.2, 0.15) ]},              'key_type': 'trans', 'pos': True , 'comment': '[mag] full-frame transient [T-NSIGMA]-sigma limiting mag'},

        
        # some additional ones to make sure these are listed in the dummy output catalogs
        'REDFILE':  {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'BlackBOX reduced image name'},
        'MASKFILE': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'BlackBOX mask image name'},
                
        'PSF-SIZE': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': True , 'comment': '[pix] size PSF image for optimal subtraction'},
        'PSF-CFGS': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': True , 'comment': '[config. pix] size PSF configuration image'},
        'PC-EXTCO': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': True , 'comment': '[mag] filter extinction coefficient (k) used'},
        'AIRMASSC': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': True , 'comment': 'Airmass at image center'},
        'RA-CNTR':  {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'RA (ICRS) at image center (astrometry.net)'},
        'DEC-CNTR': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'DEC (ICRS) at image center (astrometry.net)'},
        
        'NSIGMA':   {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': True , 'comment': '[sigma] input source detection threshold'},

        'DUMCAT':   {'default': False, 'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'dummy catalog without sources?'},
        'TDUMCAT':  {'default': False, 'val_type': 'skip',    'val_range': None,                                'key_type': 'trans',  'pos': False, 'comment': 'dummy transient catalog without sources?'},
        'QC-FLAG':  {'default':'red',  'val_type': 'skip',    'val_range': None,                                'key_type': 'full',   'pos': False, 'comment': 'QC flag color (green|yellow|orange|red)'},
        'TQC-FLAG': {'default':'red',  'val_type': 'skip',    'val_range': None,                                'key_type': 'trans',  'pos': False, 'comment': 'transient QC flag (green|yellow|orange|red)'},

        'T-NSIGMA': {'default':'None', 'val_type': 'skip',    'val_range': None,                                'key_type': 'trans',  'pos': True , 'comment': '[sigma] input transient detection threshold'},

        #
        
    },

    'BG2': {
        # general
        'N-INFNAN': {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,0), (1,10), (11,1e6) ],         'key_type': 'full', 'pos': True , 'comment': 'number of pixels with infinite/nan values'},
    },

    'BG3': {
        # general
        'N-INFNAN': {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,0), (1,10), (11,1e6) ],         'key_type': 'full', 'pos': True , 'comment': 'number of pixels with infinite/nan values'},
    },

    'BG4': {
        # general
        'N-INFNAN': {'default':'None', 'val_type': 'min_max', 'val_range': [ (0,0), (1,10), (11,1e6) ],         'key_type': 'full', 'pos': True , 'comment': 'number of pixels with infinite/nan values'},
    }

}

