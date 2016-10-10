#!/usr/bin/env python
"""
Guide users through installing Glue's dependencies
"""

from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

# Unfortunately, we can't rely on setuptools' install_requires
# keyword, because matplotlib doesn't properly install its dependencies
from subprocess import check_call, CalledProcessError
import sys
import importlib


class Dependency(object):

    def __init__(self, module, info, package=None, min_version=None):
        self.module = module
        self.info = info
        self.package = package or module
        self.min_version = min_version
        self.failed = False

    @property
    def installed(self):
        try:
            importlib.import_module(self.module)
            return True
        except ImportError:
            return False

    @property
    def version(self):
        try:
            module = __import__(self.module)
            return module.__version__
        except (ImportError, AttributeError):
            return 'unknown version'

    def install(self):
        if self.installed:
            return

        print("-> Installing {0} with pip".format(self.module))

        try:
            check_call(['pip', 'install', self.package])
        except CalledProcessError:
            self.failed = True

    def help(self):
        result = """
{module}:
******************

{info}

PIP package name:
{package}
""".format(module=self.module, info=self.info, package=self.package)
        return result

    def __str__(self):
        if self.installed:
            status = 'INSTALLED (%s)' % self.version
        elif self.failed:
            status = 'FAILED (%s)' % self.info
        else:
            status = 'MISSING (%s)' % self.info
        return "%20s:\t%s" % (self.module, status)


class QtDependency(Dependency):

    def install(self):
        print("-> Cannot install {0} automatically - skipping".format(self.module))

    def __str__(self):
        if self.installed:
            status = 'INSTALLED (%s)' % self.version
        else:
            status = 'NOT INSTALLED'
        return "%20s:\t%s" % (self.module, status)


class PyQt4(QtDependency):

    @property
    def version(self):
        try:
            from PyQt4 import Qt
            return "PyQt: {0} - Qt: {1}".format(Qt.PYQT_VERSION_STR, Qt.QT_VERSION_STR)
        except (ImportError, AttributeError):
            return 'unknown version'


class PyQt5(QtDependency):

    @property
    def version(self):
        try:
            from PyQt5 import Qt
            return "PyQt: {0} - Qt: {1}".format(Qt.PYQT_VERSION_STR, Qt.QT_VERSION_STR)
        except (ImportError, AttributeError):
            return 'unknown version'


class PySide(QtDependency):

    @property
    def version(self):
        try:
            import PySide
            from PySide import QtCore
            return "PySide: {0} - Qt: {1}".format(PySide.__version__, QtCore.__version__)
        except (ImportError, AttributeError):
            return 'unknown version'


# Add any dependencies here
# Make sure to add new categories to the categories tuple
gui_framework = (
    PyQt4('PyQt4', ''),
    PyQt5('PyQt5', ''),
    PySide('PySide', '')
)

required = (
    Dependency('qtpy', 'Required'),
    Dependency('setuptools', 'Required'),
    Dependency('numpy', 'Required', min_version='1.4'),
    Dependency('matplotlib', 'Required for plotting', min_version='1.1'),
    Dependency(
        'pandas', 'Adds support for Excel files and DataFrames', min_version='0.13.1'),
    Dependency('astropy', 'Used for FITS I/O, table reading, and WCS Parsing'))

general = (
    Dependency('dill', 'Used when saving Glue sessions'),
    Dependency('h5py', 'Used to support HDF5 files'),
    Dependency('xlrd', 'Used to support Excel files'),
    Dependency('scipy', 'Used for some image processing calculation'),
    Dependency('skimage',
               'Used to read popular image formats (jpeg, png, etc.)',
               'scikit-image'))


ipython = (
    Dependency('IPython', 'Needed for interactive IPython terminal'),
    Dependency('ipykernel', 'Needed for interactive IPython terminal'),
    Dependency('qtconsole', 'Needed for interactive IPython terminal'),
    Dependency('traitlets', 'Needed for interactive IPython terminal'),
    Dependency('pygments', 'Needed for interactive IPython terminal'),
    Dependency('zmq', 'Needed for interactive IPython terminal', 'pyzmq'))


astronomy = (
    Dependency('pyavm', 'Used to parse AVM metadata in image files', 'PyAVM'),
    Dependency('spectral_cube', 'Used to read in spectral cubes', 'spectral-cube'),
    Dependency('ginga', 'Adds a ginga viewer to glue', 'ginga'),
    Dependency('astrodendro', 'Used to read in and represent dendrograms', 'astrodendro'))


testing = (
    Dependency('mock', 'Used in test code'),
    Dependency('pytest', 'Used in test code'))

export = (
    Dependency('plotly', 'Used to explort plots to Plot.ly'),
)

categories = (('gui framework', gui_framework),
              ('required', required),
              ('general', general),
              ('ipython terminal', ipython),
              ('astronomy', astronomy),
              ('testing', testing),
              ('export', export))

dependencies = dict((d.module, d) for c in categories for d in c[1])


def get_status():
    s = ""
    for category, deps in categories:
        s += "%21s" % category.upper() + os.linesep
        for dep in deps:
            s += str(dep) + os.linesep
        s += os.linesep
    return s


def get_status_as_odict():
    status = OrderedDict()
    for category, deps in categories:
        for dep in deps:
            if dep.installed:
                status[dep.module] = dep.version
            else:
                status[dep.module] = "Not installed"
    return status


def show_status():
    print(get_status())


def install_all():
    for category, deps in categories:
        for dep in deps:
            dep.install()


def install_selected(modules):
    modules = set(m.lower() for m in modules)

    for category, deps in categories:
        for dep in deps:
            if dep.installed:
                continue
            if dep.module.lower() in modules or category.lower() in modules:
                dep.install()


def main(argv=None):
    argv = argv or sys.argv

    usage = """usage:
    #install all dependencies
    %s install

    #show all dependencies
    %s list

    #install a specific dependency or category
    %s install astropy
    %s install astronomy

    #display information about a dependency
    %s info astropy
""" % ('glue-deps', 'glue-deps', 'glue-deps', 'glue-deps', 'glue-deps')

    if len(argv) < 2 or argv[1] not in ['install', 'list', 'info']:
        sys.stderr.write(usage)
        sys.exit(1)

    if argv[1] == 'info':
        if len(argv) != 3:
            sys.stderr.write(usage)
            sys.stderr.write("Please specify a dependency\n")
            sys.exit(1)

        dep = dependencies.get(argv[2], None)

        if dep is None:
            sys.stderr.write("Unrecognized dependency: %s\n" % argv[2])
            sys.exit(1)

        print(dep.help())
        sys.exit(0)

    if argv[1] == 'list':
        show_status()
        sys.exit(0)

    # argv[1] == 'install'
    if len(argv) == 2:
        install_all()
        show_status()
        sys.exit(0)

    install_selected(argv[2:])
    show_status()


if __name__ == "__main__":
    main()
