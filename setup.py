#!/usr/bin/env python
from __future__ import print_function

from setuptools import setup, find_packages
from distutils.core import Command

import os
import re
import sys
import subprocess

# Generate version.py

with open('glue/version.py') as infile:
    exec(infile.read())

# If the version is not stable, we can add a git hash to the __version__
if '.dev' in __version__:  # noqa

    # Find hash for __githash__ and dev number for __version__ (can't use hash
    # as per PEP440)

    command_hash = 'git rev-list --max-count=1 --abbrev-commit HEAD'
    command_number = 'git rev-list --count HEAD'

    try:
        commit_hash = subprocess.check_output(
            command_hash, shell=True).decode('ascii').strip()
        commit_number = subprocess.check_output(
            command_number, shell=True).decode('ascii').strip()
    except Exception:
        pass
    else:
        # We write the git hash and value so that they gets frozen if installed
        with open(os.path.join('glue', '_githash.py'), 'w') as f:
            f.write("__githash__ = \"{githash}\"\n".format(
                githash=commit_hash))
            f.write("__dev_value__ = \"{dev_value}\"\n".format(
                dev_value=commit_number))

        # We modify __version__ here too for commands such as egg_info
        __version__ = re.sub('\.dev[^"]*', '.dev{0}'.format(commit_number),
                             __version__)  # noqa

with open('README.rst') as infile:
    LONG_DESCRIPTION = infile.read()

cmdclass = {}

# Define built-in plugins
entry_points = """
[glue.plugins]
export_d3po = glue.plugins.export_d3po:setup
export_plotly = glue.plugins.exporters.plotly:setup
pv_slicer = glue.plugins.tools.pv_slicer:setup
coordinate_helpers = glue.plugins.coordinate_helpers:setup
spectral_cube = glue.plugins.data_factories.spectral_cube:setup
dendro_viewer = glue.plugins.dendro_viewer:setup
image_viewer = glue.viewers.image:setup
scatter_viewer = glue.viewers.scatter:setup
histogram_viewer = glue.viewers.histogram:setup
profile_viewer = glue.viewers.profile:setup
table_viewer = glue.viewers.table:setup
data_exporters = glue.core.data_exporters:setup
fits_format = glue.io.formats.fits:setup
export_python = glue.plugins.tools:setup

[console_scripts]
glue-config = glue.config_gen:main
glue-deps = glue._deps:main

[gui_scripts]
glue = glue.main:main
"""

# Glue can work with PyQt5 and PySide2. We can't add them to install_requires
# because they aren't on PyPI but we can check here that one of them is
# installed.
try:
    import PyQt5  # noqa
except ImportError:
    try:
        import PySide2  # noqa
    except ImportError:
        print("Glue requires PyQt5 or PySide2 to be installed")
        sys.exit(1)

install_requires = ['numpy>=1.9',
                    'pandas>=0.14',
                    'astropy>=2.0',
                    'matplotlib>=2.0',
                    'qtpy>=1.2',
                    'setuptools>=1.0',
                    'ipython>=4.0',
                    'ipykernel',
                    'qtconsole',
                    'dill>=0.2',
                    'xlrd>=1.0',
                    'h5py>=2.4',
                    'bottleneck>=1.2',
                    'mpl-scatter-density>=0.3']

extras_require = {
    'recommended': ['scipy',
                    'scikit-image',
                    'plotly'],
    'astronomy': ['PyAVM',
                  'astrodendro',
                  'spectral-cube']
}

extras_require['all'] = (extras_require['recommended'] +
                         extras_require['astronomy'])

setup(name='glue-core',
      version=__version__,
      description='Multidimensional data visualzation across files',
      long_description=LONG_DESCRIPTION,
      author='Chris Beaumont, Thomas Robitaille',
      author_email='glueviz@gmail.com',
      url='http://glueviz.org',
      install_requires=install_requires,
      extras_require=extras_require,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: BSD License'
          ],
      packages=find_packages(),
      entry_points=entry_points,
      cmdclass=cmdclass,
      package_data={'': ['*.png', '*.ui', '*.glu', '*.hdf5', '*.fits',
                         '*.xlsx', '*.txt', '*.csv']}
      )
