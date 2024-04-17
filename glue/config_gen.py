#!/usr/bin/env python
"""
Script used to create template config.py files for Glue
"""

import os
import sys
from shutil import copyfile

import glue


def get_clobber():
    result = None
    result = input("\nDestination file exists. Overwrite? [y/n] ")
    while result not in ['y', 'n']:
        print("\tPlease choose one of [y/n]")
        result = input("\nDestination file exists. Overwrite? [y/n] ")

    return result == 'y'


def main():

    # Import at runtime because some tests change this value. We also don't
    # just import the function directly otherwise it is cached.
    from glue import config
    dest = config.CFG_DIR

    if not os.path.exists(dest):
        print("Creating directory %s" % dest)
        os.makedirs(dest)

    infile = os.path.join(glue.__path__[0], 'default_config.py')
    outfile = os.path.join(dest, 'config.py')

    print("Creating file %s" % outfile)

    if os.path.exists(outfile):
        clobber = get_clobber()
        if not clobber:
            print("Exiting")
            sys.exit(1)

    copyfile(infile, outfile)


if __name__ == "__main__":
    main()
