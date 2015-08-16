#!/usr/bin/env python
from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import os
import sys
import subprocess

# Generate version.py

with open('glue/version.py') as infile:
    exec(infile.read())

# If the version is not stable, we can add a git hash to the __version__
if '.dev' in __version__:

    # Find hash for __githash__ and dev number for __version__ (can't use hash
    # as per PEP440)

    command_hash = 'git rev-list --max-count=1 --abbrev-commit HEAD'
    command_number = 'git rev-list --count HEAD'

    try:
        commit_hash = subprocess.check_output(command_hash, shell=True).decode('ascii').strip()
        commit_number = subprocess.check_output(command_number, shell=True).decode('ascii').strip()
    except Exception:
        pass
    else:
        # We write the git hash and value so that they gets frozen if installed
        with open(os.path.join('glue', '_githash.py'), 'w') as f:
            f.write("__githash__ = \"{githash}\"\n".format(githash=commit_hash))
            f.write("__dev_value__ = \"{dev_value}\"\n".format(dev_value=commit_number))

        # We modify __version__ here too for commands such as egg_info
        __version__ += commit_number

try:
    import pypandoc
    LONG_DESCRIPTION = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    with open('README.md') as infile:
        LONG_DESCRIPTION = infile.read()

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
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args + ['glue'])
        sys.exit(errno)

cmdclass['test'] = PyTest

# Define built-in plugins
entry_points = """
[glue.plugins]
ginga_viewer = glue.plugins.ginga_viewer:setup
export_d3po = glue.plugins.export_d3po:setup
export_plotly = glue.plugins.export_plotly:setup
pv_slicer = glue.plugins.tools.pv_slicer:setup
spectrum_tool = glue.plugins.tools.spectrum_tool:setup
coordinate_helpers = glue.plugins.coordinate_helpers:setup

[console_scripts]
glue = glue.main:main
glue-config = glue.config_gen:main
glue-deps = glue._deps:main
"""

setup(name='glueviz',
      version=__version__,
      description='Multidimensional data visualzation across files',
      long_description=LONG_DESCRIPTION,
      author='Chris Beaumont, Thomas Robitaille',
      author_email='glueviz@gmail.com',
      url='http://glueviz.org',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: BSD License'
          ],
      packages=find_packages(),
      entry_points=entry_points,
      cmdclass=cmdclass,
      package_data={'': ['*.png', '*.ui']}
      )
