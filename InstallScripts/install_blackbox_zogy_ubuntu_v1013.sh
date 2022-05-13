#!/bin/bash

# Bash script to help install BlackBOX and ZOGY automatically on an
# Ubuntu machine. It has been tested using a Google cloud VM instance
# with a fresh install of Ubuntu 18.04 LTS.
#
# to run: download and execute "./install_blackbox_zogy_ubuntu.sh"
#
# Still to do:
#
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading
#
#
# versions/settings
# ================================================================================

# python version
v_python="3"
# zogy, blackbox and meercrab; for latest version, leave these empty ("") or comment out
v_blackbox=""
v_zogy=""
v_meercrab=""

# define home of zogy, blackbox and meercrab
zogyhome=${PWD}/ZOGY
blackboxhome=${PWD}/BlackBOX
meercrabhome=${PWD}/meercrab
# define data home (here defined for MeerLICHT):
datahome=/idia/projects/meerlicht

# exit script if zogyhome and/or blackboxhome and/or meercrabhome already exist
if [ -d "${zogyhome}" ] || [ -d "${blackboxhome}" ] || [ -d "${meercrabhome}" ]
then
    echo "${zogyhome} and/or ${blackboxhome} and/or ${meercrabhome} already exist(s); exiting script"
    exit
fi


# check Linux version, update/upgrade it and set package manager
# ================================================================================

uname_str=$(uname -a)
if [[ ${uname_str,,} == *"ubuntu"* ]]
then
    packman="apt-get"
    # update
    sudo ${packman} -y update
fi
# upgrade
sudo ${packman} -y upgrade


# python, pip and git
# ================================================================================

echo "installing python, pip and git"
sudo ${packman} -y install python${v_python}
sudo ${packman} -y install python${v_python}-dev

if [ ${v_python} \< "3" ]
then
    sudo ${packman} -y install python-pip
else
    sudo ${packman} -y install python3-pip
fi
pip="python${v_python} -m pip"

# git
sudo ${packman} -y install git git-lfs


# clone ZOGY and BlackBOX and MeerCRAB repositories in current directory
# ================================================================================

echo "cloning ZOGY and BlackBOX repository"
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

if [ ! -z ${v_meercrab} ]
then
    meercrab_branch="--branch v${v_meercrab}"
    v_meercrab_git="@v${v_meercrab}"
fi
echo "cloning meercrab repository"
git clone ${meercrab_branch} https://github.com/Zafiirah13/meercrab
#cd ${meercrabhome}
#git lfs install
#git lfs pull
#cd ${meercrabhome}/..


# install ZOGY, BlackBOX and MeerCRAB repositories
# ================================================================================

# need to install acstools first; if left until installation of
# blackbox as package in the setup file, or separately after the
# installation of the MeerCRAB packages, it leads to this error
# "Preparing wheel metadata: finished with status 'error'", starting
# from blackbox v1.0.12 (even though acstools nor python versions
# changed from previous installation - v1.0.11 - that went fine).
sudo -H ${pip} install acstools

echo "installing MeerCRAB packages"
# for MeerCRAB, not possible to use setup.py on git with latest python:
#sudo -H ${pip} install git+https://github.com/Zafiirah13/meercrab${v_meercrab_git}
# so install required packages manually:
#sudo -H ${pip} install pandas tensorflow imbalanced-learn matplotlib scipy keras Pillow scikit_learn numpy==1.19.2 astropy h5py==3.1.0 testresources
# require astropy==4.3.1 at the moment for aplpy not to break, which is does with v5.0
sudo -H ${pip} install pandas tensorflow imbalanced-learn matplotlib scipy keras Pillow scikit_learn numpy astropy==4.3.1 h5py testresources

echo "installing ZOGY and BlackBOX packages"
sudo -H ${pip} install git+https://github.com/pmvreeswijk/ZOGY${v_zogy_git}
sudo -H ${pip} install git+https://github.com/pmvreeswijk/BlackBOX${v_blackbox_git}


# packages used by ZOGY
# ================================================================================

# Astrometry.net
echo "installing astrometry.net"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install astrometry.net

# SExtractor (although it seems already installed automatically)
echo "installing sextractor"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install sextractor
# the executable for this installation is 'sextractor' while ZOGY
# versions starting from 0.9.2 expect 'source-extractor'; make a
# symbolic link; N.B.: since 2020-04-25 not needed anymore
#sudo ln -s /usr/bin/sextractor /usr/bin/source-extractor

# SWarp
echo "installing SWarp"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install swarp
# the executable for this installation is 'SWarp' while ZOGY expects
# 'swarp'; make a symbolic link
sudo ln -s /usr/bin/SWarp /usr/bin/swarp

# PSFEx - this basic install does not allow multi-threading
echo "installing PSFEx"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install psfex

# ds9; add environment DEBIAN_FRONTEND to avoid interaction with TZONE
echo "installing saods9"
DEBIAN_FRONTEND=noninteractive sudo ${packman} -q -y install saods9

# download and install packages used by match2SSO
# ================================================================================

mkdir match2SSO
cd match2SSO

echo "installing lunar"
git clone https://github.com/Bill-Gray/lunar
cd lunar
make
#make install
sudo make GLOBAL=Y install
cd ..

echo "installing jpl_eph"
git clone https://github.com/Bill-Gray/jpl_eph
cd jpl_eph
make
#make install
sudo make GLOBAL=Y install
cd ..

cd lunar
# integrat requires jpl_eph to be installed
make integrat
cd ..

wget https://www.minorplanetcenter.net/iau/lists/ObsCodes.html
#echo "downloading linux_m13000p17000.441"
#wget ftp://ssd.jpl.nasa.gov/pub/eph/planets/Linux/de441/linux_m13000p17000.441
cd ..

# download calibration catalog
# ================================================================================

url="https://storage.googleapis.com/blackbox-auxdata"

# N.B.: no need to download these when preparing singularity image, as
# it is copied over from local disk in .def file

# with Kurucz templates
#sudo wget -nc $url/photometry/ML_calcat_kur_allsky_ext1deg_20181115.fits.gz -P ${ZOGYHOME}/CalFiles/
# with Pickles templates
#sudo wget -nc $url/photometry/ML_calcat_pick_allsky_ext1deg_20181201.fits.gz -P ${ZOGYHOME}/CalFiles/
#echo "gunzipping calibration catalog(s) ..."
#sudo gunzip ${ZOGYHOME}/CalFiles/ML_calcat*.gz


# download astrometry.net index files
# ================================================================================

# make sure index files are saved in the right directory; on mlcontrol
# these are in /usr/local/astrometry/data/ (config file:
# /usr/local/astrometry/etc/astrometry.cfg) while on GCloud VM
# installation they are supposed to be in /usr/share/astrometry
# (config file: /etc/astrometry.cfg)
dir1="/usr/share/astrometry"
dir2="/usr/local/astrometry/data"
dir3="${HOME}/IndexFiles"
if [ -d "${dir1}" ]
then
    dir_save=${dir1}
elif [ -d "${dir2}" ]
then
    dir_save=${dir2}
else
    dir_save=${dir3}
    mkdir ${dir3}
fi
# N.B.: skipping download as this is done in .def file
#echo "downloading Astrometry.net index files to directory ${dir_save}"
#echo 
#sudo wget -nc $url/astrometry/index-500{4..6}-0{0..9}.fits -P ${dir_save}
#sudo wget -nc $url/astrometry/index-500{4..6}-1{0..1}.fits -P ${dir_save}


# set environent variables:
# ================================================================================

# let /usr/bin/python refer to version installed above
sudo ln -sf /usr/bin/python${v_python} /usr/bin/python  

echo
echo "======================================================================"
echo
echo "copy and paste the commands below to your shell startup script"
echo "(~/.bashrc, ~/.cshrc or ~/.zshrc) for these system variables"
echo "and python alias to be set when starting a new terminal, e.g.:"
echo
echo "# BlackBOX and ZOGY system variables"

if [[ ${SHELL} == *"bash"* ]] || [[ ${SHELL} == *"zsh"* ]]
then
    echo "export ZOGYHOME=${zogyhome}"
    echo "export BLACKBOXHOME=${blackboxhome}"
    echo "export MEERCRABHOME=${meercrabhome}"
    echo "# update DATAHOME to folder where the data are sitting"
    echo "export DATAHOME=${datahome}"
    echo "if [ -z \"\${PYTHONPATH}\" ]"
    echo "then"
    echo "    export PYTHONPATH=${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings:${meercrabhome}"
    echo "else"
    echo "    export PYTHONPATH=\${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings:${meercrabhome}"
    echo "fi"
fi

if [[ ${SHELL} == *"csh"* ]]
then
    echo "setenv ZOGYHOME ${zogyhome}"
    echo "setenv BLACKBOXHOME ${blackboxhome}"
    echo "setenv MEERCRABHOME ${meercrabhome}"
    echo "# update DATAHOME to folder where the data are sitting"
    echo "setenv DATAHOME ${datahome}"
    echo "if ( \$?PYTHONPATH ) then"
    echo "    setenv PYTHONPATH \${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings:${meercrabhome}"
    echo "else"
    echo "    setenv PYTHONPATH ${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings:${meercrabhome}"
    echo "endif"
fi

echo
echo "======================================================================"
echo
