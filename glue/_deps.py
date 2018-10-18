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

from glue._plugin_helpers import iter_plugin_entry_points


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
        except ImportError:
            return 'unknown version'
        except AttributeError:
            try:
                return module.__VERSION__
            except AttributeError:
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
        return "%20s:\t%s" % (self.package, status)


class Python(Dependency):

    def __init__(self):
        self.package = 'Python'

    @property
    def installed(self):
        return True

    @property
    def version(self):
        return sys.version.split()[0]


class QtDependency(Dependency):

    def install(self):
        print("-> Cannot install {0} automatically - skipping".format(self.module))

    def __str__(self):
        if self.installed:
            status = 'INSTALLED (%s)' % self.version
        else:
            status = 'NOT INSTALLED'
        return "%20s:\t%s" % (self.module, status)


class PyQt5(QtDependency):

    @property
    def version(self):
        try:
            from PyQt5 import Qt
            return "PyQt: {0} - Qt: {1}".format(Qt.PYQT_VERSION_STR, Qt.QT_VERSION_STR)
        except (ImportError, AttributeError):
            return 'unknown version'


class PySide2(QtDependency):

    @property
    def version(self):
        try:
            import PySide2
            from PySide2 import QtCore
            return "PySide2: {0} - Qt: {1}".format(PySide2.__version__, QtCore.__version__)
        except (ImportError, AttributeError):
            return 'unknown version'


# Add any dependencies here
# Make sure to add new categories to the categories tuple

python = (
    Python(),
)

gui_framework = (
    PyQt5('PyQt5', ''),
    PySide2('PySide2', '')
)

required = (
    Dependency('qtpy', 'Required', min_version='1.2'),
    Dependency('setuptools', 'Required', min_version='1.0'),
    Dependency('numpy', 'Required', min_version='1.9'),
    Dependency('bottleneck', 'Required', min_version='1.2'),
    Dependency('matplotlib', 'Required for plotting', min_version='2.0'),
    Dependency('pandas', 'Adds support for Excel files and DataFrames', min_version='0.14'),
    Dependency('astropy', 'Used for FITS I/O, table reading, and WCS Parsing', min_version='2.0'),
    Dependency('dill', 'Used when saving Glue sessions', min_version='0.2'),
    Dependency('h5py', 'Used to support HDF5 files', min_version='2.4'),
    Dependency('xlrd', 'Used to support Excel files', min_version='1.0'),
    Dependency('mpl_scatter_density', 'Used to make fast scatter density plots', 'mpl-scatter-density', min_version='0.3'),
)

general = (
    Dependency('scipy', 'Used for some image processing calculation'),
    Dependency('skimage',
               'Used to read popular image formats (jpeg, png, etc.)',
               'scikit-image'))


ipython = (
    Dependency('IPython', 'Needed for interactive IPython terminal', min_version='4'),
    Dependency('qtconsole', 'Needed for interactive IPython terminal'),
    Dependency('ipykernel', 'Needed for interactive IPython terminal'),
    Dependency('traitlets', 'Needed for interactive IPython terminal'),
    Dependency('pygments', 'Needed for interactive IPython terminal'),
    Dependency('zmq', 'Needed for interactive IPython terminal', 'pyzmq'))


astronomy = (
    Dependency('pyavm', 'Used to parse AVM metadata in image files', 'PyAVM'),
    Dependency('spectral_cube', 'Used to read in spectral cubes', 'spectral-cube'),
    Dependency('astrodendro', 'Used to read in and represent dendrograms', 'astrodendro'))


testing = (
    Dependency('mock', 'Used in test code'),
    Dependency('pytest', 'Used in test code'))

export = (
    Dependency('plotly', 'Used to explort plots to Plot.ly'),
)


def plugins():
    modules = []
    dependencies = []
    for entry_point in iter_plugin_entry_points():
        module_name = entry_point.module_name.split('.')[0]
        package = entry_point.dist.project_name
        modules.append((module_name, package))
    for module, package in sorted(set(modules)):
        dependencies.append(Dependency(module, '', package=package))
    return dependencies


categories = (('python', python),
              ('gui framework', gui_framework),
              ('required', required),
              ('plugins', plugins()),
              ('ipython terminal', ipython),
              ('general', general),
              ('astronomy', astronomy),
              ('testing', testing),
              ('export', export))


dependencies = dict((d.package, d) for c in categories for d in c[1])


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
                status[dep.package] = dep.version
            else:
                status[dep.package] = "Not installed"
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
