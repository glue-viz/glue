#!/usr/bin/env python
from __future__ import print_function

import sys
from distutils.version import LooseVersion

try:
    import setuptools
    assert LooseVersion(setuptools.__version__) >= LooseVersion('30.3')
except (ImportError, AssertionError):
    sys.stderr.write("ERROR: setuptools 30.3 or later is required\n")
    sys.exit(1)

from setuptools import setup
from setuptools.config import read_configuration

# Glue can work with PyQt5 and PySide2. We first check if they are already
# installed before adding PyQt5 to install_requires, since the conda
# installation of PyQt5 is not recognized by install_requires.

conf = read_configuration('setup.cfg')
install_requires = conf['options']['install_requires']

try:
    import PyQt5  # noqa
except ImportError:
    try:
        import PySide2  # noqa
    except ImportError:
        install_requires.append('PyQt5')

setup(use_scm_version=True, install_requires=install_requires)
