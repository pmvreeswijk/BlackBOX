#!/bin/bash

# Bash script to help install BlackBOX and ZOGY automatically at IDIA.
#
# to run: download and execute "./install_blackbox_zogy_IDIA.sh"
#
# Still to do:
#
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading
#
#
# versions/settings
# ================================================================================

# python version
#v_python="3.7"
# zogy and blackbox; for latest version, leave these empty ("") or comment out
v_blackbox="0.9.1"
v_zogy="0.9.1"

# define home of zogy, data and blackbox
zogyhome=${PWD}/ZOGY
blackboxhome=${PWD}/BlackBOX

# exit script if zogyhome and/or blackboxhome already exist
if [ -d "${zogyhome}" ] || [ -d "${blackboxhome}" ]
then
    echo "${zogyhome} and/or ${blackboxhome} already exist(s); exiting script"
    exit
fi

# python and pip
# ================================================================================

#sudo ${packman} install python${v_python/./} py${v_python/./}-pip

# problem with using pipenv:
# WARNING: pipenv requires an #egg fragment for version controlled dependencies.
# Please install remote dependency in the form 
# git+git://github.com/pmvreeswijk/ZOGY#egg=<package-name>.
#sudo ${packman} install python${v_python/./} pipenv

# for now, use good old pip
#pip="python${v_python} -m pip"


# clone ZOGY and BlackBOX repositories in current directory
# ================================================================================

if [ ! -z ${v_zogy} ]
then
    zogy_branch="--branch v${v_zogy}"
    v_zogy_git="@v${v_zogy}"
fi
git clone ${zogy_branch} https://github.com/pmvreeswijk/ZOGY

if [ ! -z ${v_blackbox} ]
then
    blackbox_branch="--branch v${v_blackbox}"
    v_blackbox_git="@v${v_blackbox}"
fi
git clone ${blackbox_branch} https://github.com/pmvreeswijk/BlackBOX

# rsync BlackBOX content to ZOGY
rsync -av ${blackboxhome}/* ${zogyhome}
# remove BlackBOX to avoid confusion
rm -rf ${blackboxhome}


# install ZOGY and BlackBOX repositories
# ================================================================================

#sudo -H ${pip} install git+git://github.com/pmvreeswijk/ZOGY${v_zogy_git}
#sudo -H ${pip} install git+git://github.com/pmvreeswijk/BlackBOX${v_blackbox_git}

