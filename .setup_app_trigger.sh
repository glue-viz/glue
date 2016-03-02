#!/bin/bash -x

gem install travis -v 1.6.11
gem environment
export PATH=`gem environment | grep "EXECUTABLE DIRECTORY" | cut -d":" -f2 | cut -c 2-`:$PATH

echo y | travis branches -r glue-viz/Travis-MacGlue --skip-version-check # install shell completion tool
