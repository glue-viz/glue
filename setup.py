#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, Command, find_packages

try:  # Python 3.x
    from setuptools.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2.x
    from setuptools.command.build_py import build_py

import sys
import platform
import subprocess

from setupext import (print_line, print_raw, print_status,
                      check_for_numpy, check_for_matplotlib,
                      check_for_qt4, check_for_ipython, check_for_scipy,
                      check_for_astropy, check_for_aplpy, check_for_pytest,
                      check_for_pyside, check_for_pil, check_for_mock
                      )

def is_windows():
    return platform.system() == 'Windows'

def print_sysinfo():
    """Print information about relevant dependencies"""
    #get version information
    for line in open('glue/version.py'):
        if (line.startswith('__version__')):
            exec(line.strip())

    #Print external package information
    print_line()
    print_raw("BUILDING GLUE")
    print_status('glue', __version__)
    print_status('python', sys.version)
    print_status('platform', sys.platform)
    if sys.platform == 'win32':
        print_status('Windows version', sys.getwindowsversion())

    print_raw("")
    print_raw("REQUIRED DEPENDENCIES")
    if not check_for_numpy('1.4'):
        sys.exit(1)
    check_for_matplotlib()
    check_for_qt4()
    check_for_pyside()

    print_raw("")
    print_raw("RECOMMENDED DEPENDENCIES")
    check_for_scipy()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : GENERAL")
    check_for_ipython()
    check_for_pil()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : ASTRONOMY")
    check_for_astropy()
    check_for_aplpy()

    print_raw("")
    print_raw("OPTIONAL DEPENDENCIES : TESTING")
    check_for_pytest()
    check_for_mock()
    print_line()

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

setup(name='Glue',
      version='0.1.0',
      description = 'Multidimensional data visualzation across files',
      author='Chris Beaumont, Thomas Robitaille',
      author_email='glueviz@gmail.com',
      url='http://glueviz.org',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Data Visualization',
          ],
      packages = find_packages(),
      entry_points={'console_scripts' : console_scripts},
      cmdclass=cmdclass,
      package_data={'': ['*.png', '*.ui']}
    )
