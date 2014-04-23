#!/bin/bash -x

sudo apt-get install rubygems
rvmsudo gem install travis
gem environment
export PATH=`gem environment | grep "EXECUTABLE DIRECTORY" | cut -d":" -f2 | cut -c 2-`:$PATH

echo y | travis  # install shell completion tool
