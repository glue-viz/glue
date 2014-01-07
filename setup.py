#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, Command, find_packages

import sys
import subprocess

try:
    import pypandoc
    LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    with open('README.md') as infile:
        LONG_DESCRIPTION=infile.read()

# read __version__
with open('glue/version.py') as infile:
    exec(infile)

cmdclass = {}


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call([sys.executable, 'runtests.py', 'glue'])
        raise SystemExit(errno)

cmdclass['test'] = PyTest


console_scripts = ['glue = glue.main:main',
                   'glue-config = glue.config_gen:main',
                   'glue-deps = glue._deps:main',
                   ]

setup(name='glueviz',
      version=__version__,
      description = 'Multidimensional data visualzation across files',
      long_description=LONG_DESCRIPTION,
      author='Chris Beaumont, Thomas Robitaille',
      author_email='glueviz@gmail.com',
      url='http://glueviz.org',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: BSD License'
          ],
      packages = find_packages(),
      entry_points={'console_scripts' : console_scripts},
      cmdclass=cmdclass,
      package_data={'': ['*.png', '*.ui']}
    )
