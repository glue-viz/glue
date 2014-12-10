#!/bin/bash -x

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh

chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/anaconda/bin:/home/travis/miniconda/bin:home/travis/miniconda3/bin:$PATH
which conda
conda update --yes conda
