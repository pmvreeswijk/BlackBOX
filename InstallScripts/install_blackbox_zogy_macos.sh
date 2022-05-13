#!/bin/bash

# Bash script to help install BlackBOX and ZOGY automatically on mac
# OS. It has been tested with a macbook running 10.14 (Mojave).
#
# It is assumed that these programs are already installed (potentially
# there are more required):
#
# - Xcode (see https://guide.macports.org/chunked/installing.html)
# - MacPorts (see https://guide.macports.org/chunked/installing.macports.html)
# - if you already have MacPorts installed, consider doing an update/upgrade:
#   % sudo port selfupdate
#   % sudo port upgrade outdated
#
# to run: download and execute "./install_blackbox_zogy_macos.sh"
#
# Still to do:
#
# - try to make apt-get install PSFEx (and SExtractor) with multi-threading
#
#
# versions/settings
# ================================================================================

# python version
v_python="3.7"
# zogy and blackbox; for latest version, leave these empty ("") or comment out
v_blackbox="0.9.1"
v_zogy="0.9.1"

# define home of zogy, data and blackbox
zogyhome=${PWD}/ZOGY
blackboxhome=${PWD}/BlackBOX
# define data home (here defined for MeerLICHT):
datahome=/ceph/meerlicht

# exit script if zogyhome and/or blackboxhome already exist
if [ -d "${zogyhome}" ] || [ -d "${blackboxhome}" ]
then
    echo "${zogyhome} and/or ${blackboxhome} already exist(s); exiting script"
    exit
fi

# check OS version, update/upgrade it and set package manager
# ================================================================================

uname_str=$(uname -a)
if [[ ${uname_str} == *"Darwin"* ]]
then
    packman="port"
fi


# python and pip
# ================================================================================

sudo ${packman} install python${v_python/./} py${v_python/./}-pip

# problem with using pipenv:
# WARNING: pipenv requires an #egg fragment for version controlled dependencies.
# Please install remote dependency in the form 
# git+https://github.com/pmvreeswijk/ZOGY#egg=<package-name>.
#sudo ${packman} install python${v_python/./} pipenv

# for now, use good old pip
pip="python${v_python} -m pip"


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


# install ZOGY and BlackBOX repositories
# ================================================================================

sudo -H ${pip} install git+https://github.com/pmvreeswijk/ZOGY${v_zogy_git}
sudo -H ${pip} install git+https://github.com/pmvreeswijk/BlackBOX${v_blackbox_git}


# packages used by ZOGY
# ================================================================================

# SExtractor
sudo ${packman} install source-extractor
# the executable for this installation is 'source-extractor' while
# BlackBOX versions before 0.9.2 expect 'sex'; make a symbolic link
ln -s /opt/local/bin/source-extractor /opt/local/bin/sex

# PSFEx
# gcc8 is needed for atlas
sudo ${packman} install gcc8
sudo ${packman} install psfex

# SWarp
sudo ${packman} install swarp

# ds9
sudo ${packman} install ds9


# Astrometry.net
# --------------
# following instructions at http://astrometry.net/doc/build.html

# check if not already present
eval which solve-field
if (( $? == 0 ))
then
    echo "solve-field executable exists; skipping Astrometry.net installation"
else
    # first the dependencies:
    sudo -H ${pip} install git+https://github.com/esheldon/fitsio
    sudo ${packman} install wget cairo netpbm swig cfitsio pkgconfig
    # following line needed to provide (unusual) path to cfitsio
    export PKG_CONFIG_PATH=/opt/local/lib/pkgconfig/cfitsio.pc

    # download, make and make install
    Anet="astrometry.net-0.77"
    wget http://astrometry.net/downloads/${Anet}.tar.gz
    tar -zxvf ${Anet}.tar.gz
    currentdir=${PWD}
    cd ${Anet}
    make
    sudo make install
    cd ${currentdir}
    rm -rf ${Anet}*
fi


# download calibration catalog
# ================================================================================

url="https://storage.googleapis.com/blackbox-auxdata"

# with Kurucz templates
sudo wget -nc $url/photometry/ML_calcat_kur_allsky_ext1deg_20181115.fits.gz -P ${ZOGYHOME}/CalFiles/
# with Pickles templates
#sudo wget -nc $url/photometry/ML_calcat_pick_allsky_ext1deg_20181201.fits.gz -P ${ZOGYHOME}/CalFiles/
sudo gunzip ${ZOGYHOME}/CalFiles/ML_calcat*.gz


# download astrometry.net index files
# ================================================================================

# make sure index files are saved in the right directory; on mlcontrol
# and on mac os these are in /usr/local/astrometry/data/ (config file:
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
echo "downloading Astrometry.net index files to directory ${dir_save}"
echo 
sudo wget -nc $url/astrometry/index-500{4..6}-0{0..9}.fits -P ${dir_save}
sudo wget -nc $url/astrometry/index-500{4..6}-1{0..1}.fits -P ${dir_save}


# set environent variables:
# ================================================================================

echo
echo "======================================================================"
echo 
echo "copy and paste the commands below to your shell startup script"
echo "(~/.bashrc, ~/.cshrc or ~/.zshrc) for these system variables"
echo "to be set when starting a new terminal, e.g.:"
echo
echo "# BlackBOX and ZOGY system variables"

if [[ ${SHELL} == *"bash"* ]] || [[ ${SHELL} == *"zsh"* ]]
then
    echo "export ZOGYHOME=${zogyhome}"
    echo "export BLACKBOXHOME=${blackboxhome}"
    echo "# update DATAHOME to folder where the data are sitting"
    echo "export DATAHOME=${datahome}"
    echo "if [ -z \"\${PYTHONPATH}\" ]"
    echo "then"
    echo "    export PYTHONPATH=${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings"
    echo "else"
    echo "    export PYTHONPATH=\${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings"
    echo "fi"
fi

if [[ ${SHELL} == *"csh"* ]]
then
    echo "setenv ZOGYHOME ${zogyhome}"
    echo "setenv BLACKBOXHOME ${blackboxhome}"
    echo "# update DATAHOME to folder where the data are sitting"
    echo "setenv DATAHOME ${datahome}"
    echo "if ( \$?PYTHONPATH ) then"
    echo "    setenv PYTHONPATH \${PYTHONPATH}:${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings"
    echo "else"
    echo "    setenv PYTHONPATH ${zogyhome}:${zogyhome}/Settings:${blackboxhome}:${blackboxhome}/Settings"
    echo "endif"
fi

echo "To make this the default Python or Python 3 (i.e., the version run by"
echo "the python or python3 commands), run one or both of:"
echo
echo "sudo port select --set python python${v_python/./}"
echo "sudo port select --set python3 python${v_python/./}"
echo
echo "======================================================================"
echo
