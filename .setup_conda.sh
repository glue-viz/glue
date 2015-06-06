#!/bin/bash -x

python_version=$1

if [ ${python_version:0:1} == "2" ]
then
    wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh -O miniconda.sh
else
    wget http://repo.continuum.io/miniconda/Miniconda3-3.6.0-Linux-x86_64.sh -O miniconda.sh
fi

chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/anaconda/bin:/home/travis/miniconda/bin:/home/travis/miniconda3/bin:$PATH
which conda
