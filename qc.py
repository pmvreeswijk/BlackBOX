
import argparse
import numpy as np
import astropy.io.fits as fits
from zogy import format_cat, get_par

import set_qc
import set_zogy

__version__ = '0.2'

def qc_check (header, telescope='ML1', keywords=None, cat_type=None,
              cat_dummy=None, return_range_comment=False, 
              hide_greens=True, hide_warnings=True, log=None):
    
    """Function to determine whether the value of a given keyword is
       within an acceptable range or not. The input parameters are
       [header]: a FITS header as extracted using astropy or a python
       dictionary, and, optionally [keywords]: a list of one or more
       keywords to check.  If [keywords] is not provided, all keywords
       in [header] are checked and if any of them matches with a key
       in the [qc_range] dictionary, it will get processed.

       The [header] keywords' values are compared to a dictionary
       'qc_range' (defined in [set_qc]) in which the quality control
       ranges are defined for a number of keys.

       The value in the qc_range dictionary is also a dictionary with
       the keys:
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

       The value of each keyword is first checked against the 
       first element of the 'range' key. If the value within this
       range, the key gets a 'green' flag. If it is not within
       this range, it checks the value with the 2nd range provided.
       If there is no 2nd range provided, the key gets a 'red' flag.
       If the value is within the 2nd range, the corresponding
       flag will be 'yellow'. The 3rd range corresponds to the
       'orange' flag, and if the keyword value is not within any
       of the ranges provided, it gets a 'red' flag.

       For the value type 'sigma' only the expected value (E) and a
       standard deviation can be provided, and these are expanded to
       three ranges using: n_std = [2, 4, 7]. So if a keyword value is
       within n_std * STD, its color flag will be 'green', 'yellow'
       and 'orange', respectively. If outside of this, it will be
       flagged 'red'.
    
       The input [cat_type] can be 'new', 'ref' or 'trans'. For
       the 'new' and 'ref' cases, all set_qc keywords with key_type
       equals 'full' will be checked, while for cat_type 'trans',
       only the keywords with key_type 'trans' will be checked.

       If any keyword is flagged red and [cat_dummy] is defined, the
       function [format_cat] in zogy will be used to create a zero
       entry binary fits table with the column definitions determined
       with [cat_type], which can be 'new', 'ref' or 'trans'. For
       'new' or 'ref', the dummy catalog will contain the header
       keywords with set_qc key_type 'full', while for the 'trans'
       it will contain both 'full' and 'trans' key_types.

       The function returns: 
       - the list of keywords that were checked and found to be
         non-green. If [hide_greens] is set to False, all
         keywords that were checked are returned in a list.
       - the list of color flags corresponding to the above
         output keywords
       - if [return_range_comment] is True (default=False), two
         additional lists are returned: (1) the allowed range of
         values corresponding to the color flag, and (2) the header
         comment field of the keyword.
       - if [hide_warnings] is True (default), no warnings
         are provided
       - in case of a red flag and [cat_dummy] is not None, an empty
         output catalog of type [cat_type] will be created.

    """

    # refer to qc_range for [telescope] defined in [set_qc] module
    qc_range = set_qc.qc_range[telescope]

    # if no keywords are provided, go through all keys in [qc_range]
    if keywords is None:
        keywords = list(qc_range.keys())

    # number of keywords in input [keywords] list
    nkeys = len(keywords)
    
    # initialize output color array to 'green':
    colors_out = ['green' for _ in range(nkeys)]
    
    # colors corresponding to the different ranges, following the KNMI
    # color codes to indicate the severeness of the weather in the
    # Netherlands
    colors = ['green', 'yellow', 'orange', 'red']

    # factors that will multiplied by the standard deviation provided
    # in the 'sigma' case for the qc_range 'val_type', that together
    # with the expected value determines the qc value ranges:
    # abs(value-E) <= n_std * std
    # For instance, values within n_std[0] are flagged green, values
    # between n_std[1] and n_std[2] will be flagged orange, and values
    # beyond n_std[2] are flagged red.
    n_std = [2, 4, 7]

    # determine filter if available
    if 'FILTER' in header.keys():
        filt = header['FILTER']
        
    # dictionary for allowed absolute range for given color
    dict_range_ok = {}
    dict_ranges = {}
    
    # loop input [keywords]
    for nkey, key in enumerate(keywords):

        # check if keyword is present in qc_range
        if key.upper() not in qc_range.keys():
            if not hide_warnings:
                print ('Warning: keyword {} not present in qc_range'
                       .format(key))
            # change color to empty string
            colors_out[nkey] = ''
            continue
        
        # check if keyword is present in the header
        if key.upper() not in header.keys():
            if not hide_warnings:
                print ('Warning: keyword {} not present in the input header'
                       .format(key))
            # change color to empty string
            colors_out[nkey] = ''
            continue
        
        # if qc_range[key] val_type is set to 'skip' then skip it
        val_type = qc_range[key]['val_type']
        if val_type == 'skip':
            colors_out[nkey] = ''
            continue
        
        # if input [cat_type] equals 'trans', only consider set_qc
        # keywords with key_type 'trans' for the determination of the
        # qc-flag, otherwise (also when [cat_type] is left to the
        # default None) consider set_qc keywords with key_type 'full.
        if cat_type == 'trans':
            if qc_range[key]['key_type'] != 'trans':
                # change color to empty string
                colors_out[nkey] = ''
                continue
        else:
            if qc_range[key]['key_type'] != 'full':
                # change color to empty string
                colors_out[nkey] = ''
                continue


        val_range = qc_range[key]['val_range']
        # check if value range is specified per filter (e.g. for zeropoint)
        try:
            filt in val_range.keys()
        except:
            pass
        else:
            val_range = val_range[filt]


        # if keyword value equals 'None', then also skip it
        header_val = header[key]
        if header_val == 'None':
            print('Warning: {}=\'None\'. Skipping quality check.'.format(key))
            colors_out[nkey] = ''
            continue

        
        # update string keywords that are supposed to be booleans to
        # proper booleans (remnant of BGreduce)
        if val_type == 'bool':
            if type(header_val)==str:
                if header_val.strip() == 'T':
                    header_val = True
                else:
                    header_val = False
        

        if val_type=='sigma':
            # expand [val_range] to three ranges
            val_range = np.array(3*val_range)
            # and multiply 2nd column with [n_std]
            val_range[:,1] *= n_std
            

        nranges = np.shape(val_range)[0]
        for i in range(nranges):


            if val_type == 'exp_abs' or val_type=='sigma':
                bool_temp = np.abs(header_val-val_range[i][0]) <= val_range[i][1]
                range_ok = (val_range[i][0]-val_range[i][1], 
                            val_range[i][0]+val_range[i][1])
                
                
            elif val_type == 'exp_frac':

                bool_temp = (np.abs((header_val-val_range[i][0])/val_range[i][0])
                             <= val_range[i][1])
                range_ok = (val_range[i][0]*(1.-val_range[i][1]), 
                            val_range[i][0]*(1.+val_range[i][1]))


            elif val_type == 'min_max':

                bool_temp = (header_val >= val_range[i][0] and
                             header_val <= val_range[i][1])
                range_ok = (val_range[i][0], val_range[i][1])

                
            elif val_type == 'bool':

                bool_temp = (header_val == val_range[i])
                if i==0:
                    range_ok = val_range[i]
                else:
                    range_ok = (range_ok, val_range[i])
                    
            else:

                print ('Error: [val_type] not one of "exp_abs", "exp_frac", '
                       '"min_max", "bool" or "sigma"')
                raise SystemExit
            
                    
            if bool_temp:
                # if within current range, assign i color to this key and leave loop
                colors_out[nkey] = colors[i]
                # if first iteration, assign green range to [dict_range_ok]
                if i==0:
                    if type(range_ok) != bool:
                        dict_range_ok[key] = '{:g},{:g}'.format(*range_ok)
                    else:
                        dict_range_ok[key] = '{}'.format(range_ok)                
                break
            else:
                # if False
                if i<nranges-1:
                    # assign i+1 color to this key
                    colors_out[nkey] = colors[i+1]
                else:
                    # if this is the last iteration, assigned the last color
                    colors_out[nkey] = colors[-1]


            # this will effectively assign range_ok of previous color
            if type(range_ok) != bool:
                dict_range_ok[key] = '{:g},{:g}'.format(*range_ok)
            else:
                dict_range_ok[key] = '{}'.format(range_ok)



    colors_out = np.array(colors_out)
    mask = (colors_out != '')

    if hide_greens:
        mask = ((colors_out != 'green') & mask)

    # determine qc_flag color 
    qc_flag = 'green'
    for col in colors:
        if col in colors_out[mask]:
            qc_flag = col
    

    # the block below is only relevant in case cat_type is specified,
    # i.e. not needed for the checks on e.g. the bias or flat images.
    # Note that the science image header is updated with the final
    # new+zogy header just before exiting [blackbox_reduce], so it
    # will also including the header keywords below.
    if cat_type is not None:

        # if [cat_dummy] is provided then make the dummy catalog
        make_dumcat = (cat_dummy is not None)


        if cat_type == 'trans':
            
            header['TQC-FLAG'] = (qc_flag, 'transient QC flag '
                                  '(green|yellow|orange|red)')
            header['TDUMCAT'] = (make_dumcat, 'dummy transient catalog without '
                                 'sources?')

            # in case the QC-FLAG is worse than TQC-FLAG, make TQC-FLAG equal
            # to QC-FLAG and add the keyword 'TQC[flag color]1' with the value
            # 'QC-FLAG' to make it clear the color was inherited from QC-FLAG
            if colors.index(qc_flag) < colors.index(header['QC-FLAG']):
                header['TQC-FLAG'] = header['QC-FLAG']
                header.set('TQC{}1'.format(header['QC-FLAG'][0:3].upper()),
                           'QC-FLAG', 'flag inherited from QC-FLAG',
                           after='TQC-FLAG')


        else:

            header['QC-FLAG'] = (qc_flag, 'QC flag (green|yellow|orange|red)')
            header['DUMCAT'] = (make_dumcat, 'dummy catalog without sources?')


        # for all non-green flags, list the keyword(s) that is (are)
        # responsible for the flag and list the QC range that was violated
        if qc_flag != 'green':

            # mask of the keywords that are flagged with the color of qc_flag
            mask_col = (colors_out == qc_flag)
            # the color one up from qc_flag is needed for the header comment
            prev_col = colors[colors.index(qc_flag)-1]
            # place the new header keywords after QC-FLAG
            if cat_type == 'trans':
                prev_key = 'TQC-FLAG'
            else:
                prev_key = 'QC-FLAG'

            for ncol, key_col in enumerate(np.array(keywords)[mask_col]):

                comment = '{} range: {}'.format(prev_col, dict_range_ok[key_col])

                if cat_type == 'trans':
                    key = 'TQC{}{}'.format(qc_flag[0:3].upper(), ncol+1)
                    #header[key] = (key_col, '{} range: {}'
                    #               .format(prev_col, dict_range_ok[key_col]))
                else:
                    key = 'QC-{}{}'.format(qc_flag[0:3].upper(), ncol+1)
                    #header[key] = (key_col, '{} range: {}'
                    #               .format(prev_col, dict_range_ok[key_col]))

                header.set(key, key_col, comment, after=prev_key)
                # record the key to place the next one after it
                prev_key = key



        if make_dumcat:

            # create header_dummy copy; if header is dictionary,
            # convert to fits header
            if type(header)==dict:
                header_fits = fits.Header()
                for key in header.keys():
                    header_fits[key] = header[key]
                header_dummy = header_fits
            else:
                header_dummy = header.copy()


            # need to make sure that all keys of [qc_range] are in the
            # header to be added to the dummy catalog; if not provide
            # default value
            for nkey, key in enumerate(qc_range.keys()):
                if key not in header_dummy:
                    if (qc_range[key]['key_type']==cat_type or 
                        qc_range[key]['key_type']=='full'):
                        header_dummy[key] = (qc_range[key]['default'],
                                             qc_range[key]['comment'])


            # create empty output catalog of type [cat_type] using
            # function [format_cat] in zogy.py
            if cat_type == 'trans' and get_par(set_zogy.save_thumbnails,telescope):
                # for transient catalog, also produce thumbnail definitions
                keys_thumbnails = ['THUMBNAIL_RED', 'THUMBNAIL_REF',
                                   'THUMBNAIL_D', 'THUMBNAIL_SCORR']
                size_thumbnails = get_par(set_zogy.size_thumbnails,telescope)
                result = format_cat(None, cat_dummy, cat_type=cat_type,
                                    header_toadd=header_dummy, 
                                    apphot_radii=get_par(
                                        set_zogy.apphot_radii,telescope),
                                    data_thumbnails=None,
                                    keys_thumbnails=keys_thumbnails,
                                    size_thumbnails=size_thumbnails,
                                    ML_calc_prob=get_par(
                                        set_zogy.ML_calc_prob,telescope),
                                    tel=telescope)
            else:
                result = format_cat(None, cat_dummy, cat_type=cat_type,
                                    header_toadd=header_dummy, 
                                    apphot_radii=get_par(
                                        set_zogy.apphot_radii,telescope),
                                    tel=telescope)



    keywords_out = np.array(keywords)[mask].tolist()
    colors_out = np.array(colors_out)[mask].tolist()

    if return_range_comment:
        list_range_ok = [dict_range_ok[key] for key in keywords_out]
        list_comment = [qc_range[key]['comment'] for key in keywords_out]
        return keywords_out, colors_out, list_range_ok, list_comment
    else:
        return keywords_out, colors_out
    

################################################################################

def run_qc_check (header, telescope, cat_type=None, cat_dummy=None, log=None):
    
    """Helper function to execute [qc_check] in BlackBOX and to return a
       single flag color - the most severe color - from the output
       [colors]. If 'red' then also add some info to the [log] if it
       is provided.

    """
    
    # check if the header keyword values are within specified range
    keys, colors, ranges, comments = qc_check(header, telescope=telescope,
                                              cat_type=cat_type,
                                              cat_dummy=cat_dummy,
                                              return_range_comment=True,
                                              log=log)
    qc_flag = 'green'
    for col in ['yellow', 'orange', 'red']:
        if col in colors:
            qc_flag = col

    if qc_flag == 'red':
        for nkey, key in enumerate(keys):
            logstr = ('{} flag for keyword: {}, value: {}, allowed range: {}, '
                      ' comment: {}'.format(colors[nkey], key, header[key], 
                                            ranges[nkey], comments[nkey]))
            if log is not None:
                if colors[nkey] == 'red':
                    log.error(logstr)

    return qc_flag


################################################################################

# some use examples
if False:

    ## reading the header of a MeerLICHT reduced fits file:
    filename = '/Volumes/SSD-Data/Data/ML1/red/2019/01/13/ML1_20190113_233133_red_trans.fits'
    with fits.open(filename) as hdulist:
        header_test = hdulist[1].header
    
    example = 4
    
    if example==1:
        
        # Examples: 1) create dictionary with a few keys, and check    
        dict = {'RDNOISE': 10.0, 'S-SEEING': 5.5, 'AIRMASS': 2.7, 'Z-P': True}
        print (qc_check(dict, keywords=['RDNOISE', 'S-SEEING', 'AIRMASS', 'Z-P'],
                        return_range_comment=True, hide_greens=False, hide_warnings=False))
        
    elif example==2:

        #2) check header of an image for specific keywords:
        print (qc_check(header_test, keywords=['RDNOISE', 'S-SEEING', 'AIRMASS', 'Z-P'],
                        return_range_comment=True, hide_greens=False, hide_warnings=False))
        
    elif example==3:
        
        # 3) check header of an image for all keywords in QC_range dictionary
        print (qc_check(header_test, hide_greens=False))
        
    elif example==4:
        
        # 4) same as 3 but also create dummy full-source catalog
        print (qc_check(header_test, cat_dummy='test_cat.fits', cat_type='new'))
        
    elif example==5:
        
        # 5) same as 3 but also create dummy transient catalog
        print (qc_check(header_test, cat_dummy='test_trans.fits', cat_type='trans'))
        
