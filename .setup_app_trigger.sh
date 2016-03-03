#!/bin/bash -x

# We have to include "yes |" since sometimes, gem asks about installing the
# shell completion tool.

yes | gem install travis -v 1.6.11
yes | gem environment

export PATH=`gem environment | grep "EXECUTABLE DIRECTORY" | cut -d":" -f2 | cut -c 2-`:$PATH

yes | travis branches -r glue-viz/Travis-MacGlue --skip-version-check