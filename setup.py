#!/usr/bin/env python
from __future__ import print_function

from setuptools import setup, Command, find_packages
from setuptools.command.test import test as TestCommand

import os
import sys
import subprocess

# Generate version.py

with open('glue/version.py') as infile:
    exec(infile.read())

# If the version is not stable, we can add a git hash to the __version__
if '.dev' in __version__:
    command = 'git rev-list --max-count=1 --abbrev-commit HEAD'
    try:
        commit_hash = subprocess.check_output(command, shell=True).decode('ascii').strip()
    except Exception:
        pass

# We write the git hash so that it gets frozen if installed
with open(os.path.join('glue', '_githash.py'), 'w') as f:
    f.write("__githash__ = \"{githash}\"".format(githash=commit_hash))
    
# We modify __version__ here too for commands such as egg_info
__version__ += commit_hash

try:
    import pypandoc
    LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    with open('README.md') as infile:
        LONG_DESCRIPTION=infile.read()

cmdclass = {}

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args + ['glue'])
        sys.exit(errno)

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
