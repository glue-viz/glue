#!/usr/bin/env python
"""
Guide users through installing Glue's dependencies
"""

import os
from collections import OrderedDict

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

    def help(self):
        result = f"""
{self.module}:
******************

{self.info}

PIP package name:
{self.package}
"""
        return result

    def __str__(self):
        if self.installed:
            status = f'INSTALLED ({self.version})'
        elif self.failed:
            status = f'FAILED ({self.info})'
        else:
            status = f'MISSING ({self.info})'
        return f"{self.package:>20}:\t{status}"


class Python(Dependency):

    def __init__(self):
        self.module = 'Python'
        self.package = 'Python'
        self.info = 'Interpreter and core library'

    @property
    def installed(self):
        return True

    @property
    def version(self):
        return sys.version.split()[0]


class QtDependency(Dependency):

    def __str__(self):
        if self.installed:
            status = f'INSTALLED ({self.version})'
        else:
            status = 'NOT INSTALLED'
        return f"{self.module:>20}:\t{status}"


class PyQt5(QtDependency):

    @property
    def version(self):
        try:
            from PyQt5 import Qt
            return f"PyQt: {Qt.PYQT_VERSION_STR} - Qt: {Qt.QT_VERSION_STR}"
        except (ImportError, AttributeError):
            return 'unknown version'


class PyQt6(QtDependency):

    @property
    def version(self):
        try:
            from PyQt6 import QtCore
            return f"PyQt: {QtCore.PYQT_VERSION_STR} - Qt: {QtCore.QT_VERSION_STR}"
        except (ImportError, AttributeError):
            return 'unknown version'


class PySide2(QtDependency):

    @property
    def version(self):
        try:
            import PySide2
            from PySide2 import QtCore
            return f"PySide2: {PySide2.__version__} - Qt: {QtCore.__version__}"
        except (ImportError, AttributeError):
            return 'unknown version'


class PySide6(QtDependency):

    @property
    def version(self):
        try:
            import PySide6
            from PySide6 import QtCore
            return f"PySide6: {PySide6.__version__} - Qt: {QtCore.__version__}"
        except (ImportError, AttributeError):
            return 'unknown version'


class QtPy(Dependency):

    @property
    def installed(self):
        try:
            importlib.import_module(self.module)
            return True
        except Exception:
            # QtPy raises a PythonQtError in some cases, so we can't use
            # ImportError.
            return False


# Add any dependencies here
# Make sure to add new categories to the categories tuple

python = (
    Python(),
)

gui_framework = (
    PyQt5('PyQt5', 'Facultative QtPy backend'),
    PyQt6('PyQt6', 'Facultative QtPy backend'),
    PySide2('PySide2', 'Facultative QtPy backend'),
    PySide6('PySide6', 'Facultative QtPy backend')
)

required = (
    QtPy('qtpy', 'Required', min_version='1.9'),
    Dependency('setuptools', 'Required', min_version='30.3'),
    Dependency('echo', 'Required', min_version='0.5'),
    Dependency('numpy', 'Required', min_version='1.17'),
    Dependency('matplotlib', 'Required for plotting', min_version='3.2'),
    Dependency('pandas', 'Adds support for Excel files and DataFrames', min_version='1.2'),
    Dependency('astropy', 'Used for FITS I/O, table reading, and WCS Parsing', min_version='4.0'),
    Dependency('dill', 'Used when saving Glue sessions', min_version='0.2'),
    Dependency('h5py', 'Used to support HDF5 files', min_version='2.10'),
    Dependency('xlrd', 'Used to support Excel files', min_version='1.2'),
    Dependency('openpyxl', 'Used to support Excel files', min_version='3.0'),
    Dependency('mpl_scatter_density', 'Used to make fast scatter density plots', 'mpl-scatter-density', min_version='0.7'),
)

general = (
    Dependency('scipy', 'Used for some image processing calculation', min_version='1.1'),
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
        module_name, _, _ = entry_point.module.partition('.')
        package = entry_point.dist.name
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


USAGE = """usage:
#show all dependencies
glue-deps list

#display information about a dependency
glue-deps info astropy
"""


def main(argv=None):
    argv = argv or sys.argv

    if len(argv) < 2 or argv[1] not in ['list', 'info']:
        sys.stderr.write(USAGE)
        sys.exit(1)

    if argv[1] == 'info':
        if len(argv) != 3:
            sys.stderr.write(USAGE)
            sys.stderr.write("Please specify a dependency\n")
            sys.exit(1)

        dep = dependencies.get(argv[2], None)

        if dep is None:
            sys.stderr.write(f"Unrecognized dependency: {argv[2]:s}\n")
            sys.exit(1)

        print(dep.help())
        sys.exit(0)

    if argv[1] == 'list':
        show_status()
        sys.exit(0)


if __name__ == "__main__":
    main()
