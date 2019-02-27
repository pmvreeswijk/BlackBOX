#!/bin/bash

# This is a bash script to help install BlackBOX and ZOGY
# automatically on a linux machine. It has been tested using a Google
# cloud VM instance with a fresh install of Ubuntu 18.04 LTS, and runs
# mostly fine.
#
# Still to do:
#
# - add the calibration binary fits catalog used by zogy
# - add the Astrometry.net index files
# - try to install PSFEx (and SExtractor) with multi-threading


# ubuntu update/upgrade
# ---------------------
sudo apt-get -y update
sudo apt-get -y upgrade


# python 2 or 3
# -------------
# meant to use python3.7, but this is giving problems in Ubuntu 18.04
# with pip being 3.6, so for the moment stick to python3 (whichs
# points to python3.6)
version=3
if (( $(echo "$version < 3.0" |bc -l) )); then
    sudo apt-get -y install python2.7 python2.7-dev
    echo alias python=python2.7 >> ${HOME}/.bash_aliases
    pip=pip;
else
    sudo apt-get -y install python3 python3-dev
    echo alias python=python3 >> ${HOME}/.bash_aliases
    pip=/usr/bin/pip3;
    echo alias pip=${pip} >> ${HOME}/.bash_aliases
fi
    

# git (already installed in non-minimal version of 18.04)
# -------------------------------------------------------
sudo apt-get -y install git


# clone ZOGY and BlackBOX repositories
# ------------------------------------
cd $HOME
git clone https://github.com/pmvreeswijk/ZOGY
git clone https://github.com/pmvreeswijk/BlackBOX


# define environent variables:
# ----------------------------
zogyhome=${HOME}/ZOGY
# for the moment add these to .bashrc such that
# next login they are available
echo export ZOGYHOME=${zogyhome} >> ${HOME}/.bashrc
echo export DATAHOME=${HOME} >> ${HOME}/.bashrc


# rsync BlackBOX content to ZOGY
# ------------------------------
rsync -av ${HOME}/BlackBOX/* ${zogyhome}


# update PYTHONPATH
# -----------------
echo export PYTHONPATH=${zogyhome}\:${zogyhome}/Settings >> ${HOME}/.bashrc


# install ZOGY and BlackBOX repositories
# --------------------------------------
sudo ${pip} install git+git://github.com/pmvreeswijk/ZOGY
sudo ${pip} install git+git://github.com/pmvreeswijk/BlackBOX


# download calibration catalog
# ----------------------------
# From where? To be put in ${ZOGYHOME}/CalFiles/


# packages used by ZOGY
# ---------------------

# Astrometry.net
sudo apt-get -y install astrometry.net
# make sure correct Index files are saved in correct directory on
# mlcontrol these are in /usr/local/astrometry/data/ but this
# directory does not exist on GCloud VM it is /usr/share/astrometry/,
# while config file is at /etc/astrometry.cfg

# SExtractor (although it seems already installed automatically)
sudo apt-get -y install sextractor
# the executable for this installation is 'sextractor' while ZOGY
# expects 'sex'; make a symbolic link
sudo ln -s /usr/bin/sextractor /usr/bin/sex

# SWarp
sudo apt-get -y install swarp
# the executable for this installation is 'SWarp' while ZOGY expects
# 'swarp'; make a symbolic link
sudo ln -s /usr/bin/SWarp /usr/bin/swarp

# PSFEx - this basic install does not allow multi-threading
sudo apt-get -y install psfex

# ds9
sudo apt-get -y install saods9

# dfits
sudo apt-get -y install qfits-tools
