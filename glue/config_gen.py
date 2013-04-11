#!/usr/bin/env python

import os, sys
from shutil import copyfile

import glue

def get_clobber():
    result = None
    result = raw_input("\nDestination file exists. Overwrite? [y/n] ")
    while result not in ['y', 'n']:
        print "\tPlease choose one of [y/n]"
        result = raw_input("\nDestination file exists. Overwrite? [y/n] ")

    return result == 'y'

def main():
    dest = os.path.expanduser('~/.glue/')
    if not os.path.exists(dest):
        print "Creating directory %s" % dest
        os.makedirs(dest)

    infile = os.path.join(glue.__path__[0], 'default_config.py')
    outfile = os.path.join(dest, 'config.py')

    print "Creating file %s" % outfile

    if os.path.exists(outfile):
        clobber = get_clobber()
        if not clobber:
            print "Exiting"
            sys.exit(1)

    copyfile(infile, outfile)

if __name__ == "__main__":
    main()