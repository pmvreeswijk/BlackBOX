# set up log
import logging
log = logging.getLogger()

import argparse
import numpy as np
import astropy.io.fits as fits
from zogy import format_cat, get_par

import set_qc
import set_zogy

__version__ = '0.2'

def qc_check (header, telescope='ML1', keywords=None, check_key_type=None,
              cat_dummy=None, cat_type=None, return_range_comment=False,
              hide_greens=True, hide_warnings=True):

    """Function to determine whether the value of a given keyword is
       within an acceptable range or not.  The header keywords' values
       are compared to a dictionary 'qc_range' (defined in [set_qc])
       in which the quality control ranges are defined for a number of
       keys. The input parameters are:

       header (str): a FITS header as extracted using astropy or a
                     python dictionary

       telescope (str): determines the telescope sub-dictionary of
                        qc_range to use

       keywords (list): one or more keywords to check.  If not
                        provided, all keywords in [header] are checked
                        and if any of them matches with a key in the
                        [qc_range] dictionary, it will get processed.

       check_key_type (str): consider only those keywords whose
                             'key_type' value in the [qc_range]
                             dictionary corresponds to
                             [check_key_type]

       cat_dummy (str): name of dummy catalog to create

       cat_type (str): type of catalog to create - this determines the
                       columns included; possible values: 'new', 'ref'
                       or 'trans'

       return_range_comment (bool): if True, also return the keywords'
                                    ranges and comments

       hide_greens (bool): if True, only return results for non-green flags


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
            to the full-source catalog ('full'), the transient
            catalog ('trans'), flatfields ('flat'), etc.
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

       If [cat_dummy] is defined, the function [format_cat] in zogy
       will be used to create a zero entry binary fits table with the
       column definitions determined with [cat_type], which can be
       'new', 'ref' or 'trans'.

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
    try:
        qc_range = set_qc.qc_range[telescope]
    except:
        # for BlackGEM, all telescopes have the same quality control
        # (at least for the moment), with key 'BG'
        qc_range = set_qc.qc_range[telescope[0:2]]


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
                log.warning ('keyword {} not present in qc_range'
                             .format(key))
            # change color to empty string
            colors_out[nkey] = ''
            continue

        # check if keyword is present in the header
        if key.upper() not in header.keys():
            if not hide_warnings:
                log.warning ('keyword {} not present in the input header'
                             .format(key))
            # change color to empty string
            colors_out[nkey] = ''
            continue

        # if qc_range[key] val_type is set to 'skip' then skip it
        val_type = qc_range[key]['val_type']
        if val_type == 'skip':
            colors_out[nkey] = ''
            continue


        # if the key_type of the dictionary for this keyword does not
        # correspond to [check_key_type], skip it
        if check_key_type is not None:
            if qc_range[key]['key_type'] != check_key_type:
                # change color to empty string
                colors_out[nkey] = ''
                continue


        val_range = qc_range[key]['val_range']


        # if val_type is 'key', then val_range can contain a string
        # including a reference to another header keyword value that
        # needs to be evaluated
        cont = False
        if val_type=='key':
            # convert tuple to list, so that it can be modified below
            val_range = [list(item) for item in val_range]
            for j in range(len(val_range)):
                for i, val in enumerate(val_range[j]):
                    # check if item is a string
                    if isinstance(val, str):
                        try:
                            val_range[j][i] = eval(val)
                        except:
                            #if not hide_warnings:
                            log.warning ('could not evaluate {}; skipping '
                                         'quality check for {}'
                                         .format(val, key))
                            colors_out[nkey] = ''
                            cont = True



        # continue with next keyword if exception occurred in inner
        # loop of above block
        if cont:
            continue


        # check if value range is specified per filter (e.g. for zeropoint)
        try:
            filt in val_range.keys()
        except:
            pass
        else:
            val_range = val_range[filt]


        # if keyword value equals 'None' or None, then also skip it
        header_val = header[key]
        if header_val == 'None' or header_val is None:
            if not hide_warnings:
                log.warning ('{}=\'None\' or None; skipping quality check.'
                             .format(key))
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


        # for fields around the pole, manually increase val_range for
        # astrometric keywords due to unexplained increase in
        # astrometric scatter
        if (header['DEC'] <= -87 and
            key in ['A-DRA', 'A-DRASTD', 'A-DDEC', 'A-DDESTD']):
            val_range = np.array(3*val_range)



        nranges = np.shape(val_range)[0]
        for i in range(nranges):


            if val_type == 'exp_abs' or val_type=='sigma':
                bool_temp = np.abs(header_val-val_range[i][0]) <= val_range[i][1]
                range_ok = [val_range[i][0]-val_range[i][1],
                            val_range[i][0]+val_range[i][1]]


            elif val_type == 'exp_frac':

                bool_temp = (np.abs((header_val-val_range[i][0])/val_range[i][0])
                             <= val_range[i][1])
                range_ok = [val_range[i][0]*(1.-val_range[i][1]),
                            val_range[i][0]*(1.+val_range[i][1])]


            elif val_type == 'min_max' or val_type == 'key':

                bool_temp = (header_val >= val_range[i][0] and
                             header_val <= val_range[i][1])
                range_ok = [val_range[i][0], val_range[i][1]]


            elif val_type == 'bool':

                bool_temp = (header_val == val_range[i])
                if i==0:
                    range_ok = val_range[i]
                else:
                    range_ok = [range_ok, val_range[i]]

            else:

                log.error ('[val_type] not one of "exp_abs", "exp_frac", '
                           '"min_max", "bool", "sigma" or "key"')
                return


            # for non-boolean range: if 'pos' key in qc_range
            # dictionary is True, ensure that range_ok does not
            # contain negative values
            if qc_range[key]['pos'] and val_type != 'bool':
                range_ok[0] = max(0, range_ok[0])
                range_ok[1] = max(0, range_ok[1])


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


    # Note that the science image header is updated with the final
    # new+zogy header just before exiting [blackbox_reduce], so it
    # will also include the header keywords below.

    # boolean indicating if dummy catalog needs to be made, which is
    # if [cat_dummy] is provided as input - that the qc_flag is red or
    # not is checked before in blackbox
    make_dumcat = (cat_dummy is not None)

    # add a 'T' in front of 'QC-FLAG' and 'DUMCAT' in case
    # [check_key_type] is set to 'trans'
    if check_key_type == 'trans':
        prefix = 'T'
        label = 'transient '
    else:
        prefix = ''
        label = ''

    # place 'QC-FLAG' at the end of present header, but
    # would be good to place 'TQC-FLAG' right after it;
    # same for 'DUMCAT' and 'TDUMCAT'
    if prefix == 'T':
        prev_key = 'QC-FLAG'
    else:
        prev_key = None

    header.set ('{}QC-FLAG'.format(prefix), qc_flag, '{}QC flag (green|yellow|'
                'orange|red)'.format(label), after=prev_key)

    if prefix == 'T':
        prev_key = 'DUMCAT'
        comment = 'dummy transient catalog without sources?'
    else:
        prev_key = None
        comment = 'dummy catalog without sources?'

    #if make_dumcat:
    header.set ('{}DUMCAT'.format(prefix), make_dumcat, 'dummy {}catalog '
                'without sources?'.format(label), after=prev_key)


    # in case the QC-FLAG is worse than TQC-FLAG, make TQC-FLAG equal
    # to QC-FLAG and add the keyword 'TQC[flag color]1' with the value
    # 'QC-FLAG' to make it clear the color was inherited from QC-FLAG
    if 'QC-FLAG' in header and 'TQC-FLAG' in header:
        if colors.index(qc_flag) < colors.index(header['QC-FLAG']):
            header['TQC-FLAG'] = header['QC-FLAG']
            header.set('TQC{}1'.format(header['QC-FLAG'][0:3].upper()),
                       'QC-FLAG', 'flag inherited from QC-FLAG',
                       after='TQC-FLAG')


    # for all non-green flags, list the keyword(s) that is (are)
    # responsible for the flag and list the QC range that was violated

    # place the new header keywords after (T)QC-FLAG
    prev_key = '{}QC-FLAG'.format(prefix)

    for col in ['red', 'orange', 'yellow']:

        # mask of the keywords that are flagged with col
        mask_col = (colors_out == col)
        # the color one up from col is needed for the header comment
        prev_col = colors[colors.index(col)-1]

        for ncol, key_col in enumerate(np.array(keywords)[mask_col]):
            comment = '{} range: {}'.format(prev_col, dict_range_ok[key_col])
            key = '{}QC{}{}'.format(prefix, col[0:3].upper(), ncol+1)
            # if already present in header, replace it
            if key in header:
                header[key] = (key_col, comment)
            else:
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
            if key not in header_dummy.keys():
                if (qc_range[key]['key_type']==cat_type or
                    qc_range[key]['key_type']=='full'):
                    header_dummy[key] = (qc_range[key]['default'],
                                         qc_range[key]['comment'])


        # create empty output catalog of type [cat_type] using
        # function [format_cat] in zogy.py
        if cat_type == 'trans':

            # for transient catalog, produce thumbnail dictionary
            dict_thumbnails = {'THUMBNAIL_RED':   None,
                               'THUMBNAIL_REF':   None,
                               'THUMBNAIL_D':     None,
                               'THUMBNAIL_SCORR': None}
            save_thumbnails = get_par(set_zogy.save_thumbnails,telescope)
            size_thumbnails = get_par(set_zogy.size_thumbnails,telescope)

            result = format_cat(None, cat_dummy, cat_type=cat_type,
                                header_toadd=header_dummy,
                                apphot_radii=get_par(
                                    set_zogy.apphot_radii,telescope),
                                dict_thumbnails=dict_thumbnails,
                                save_thumbnails=save_thumbnails,
                                size_thumbnails=size_thumbnails,
                                ML_calc_prob=get_par(
                                    set_zogy.ML_calc_prob,telescope),
                                tel=telescope, set_zogy=set_zogy)

        else:
            result = format_cat(None, cat_dummy, cat_type=cat_type,
                                header_toadd=header_dummy,
                                apphot_radii=get_par(
                                    set_zogy.apphot_radii,telescope),
                                tel=telescope, set_zogy=set_zogy)



    keywords_out = np.array(keywords)[mask].tolist()
    colors_out = np.array(colors_out)[mask].tolist()

    if return_range_comment:
        list_range_ok = [dict_range_ok[key] for key in keywords_out]
        list_comment = [qc_range[key]['comment'] for key in keywords_out]
        return keywords_out, colors_out, list_range_ok, list_comment
    else:
        return keywords_out, colors_out


################################################################################

def run_qc_check (header, telescope, cat_type=None, cat_dummy=None,
                  check_key_type=None):

    """Helper function to execute [qc_check] in BlackBOX and to return a
       single flag color - the most severe color - from the output
       [colors]. If 'red' then also add some info to the [log] if it
       is provided.

    """

    # check if the header keyword values are within specified range
    keys, colors, ranges, comments = qc_check(header, telescope=telescope,
                                              cat_type=cat_type,
                                              cat_dummy=cat_dummy,
                                              check_key_type=check_key_type,
                                              return_range_comment=True)

    qc_flag = 'green'
    for col in ['yellow', 'orange', 'red']:
        if col in colors:
            qc_flag = col

    if qc_flag == 'red':
        for nkey, key in enumerate(keys):
            logstr = ('{} flag for keyword: {}, value: {}, allowed range: {}, '
                      ' comment: {}'.format(colors[nkey], key, header[key],
                                            ranges[nkey], comments[nkey]))
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
