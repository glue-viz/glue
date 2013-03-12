"""
Some helper functions taken from matplotlib
"""
from __future__ import print_function
from distutils import version


from textwrap import fill


def print_line(char='='):
    print(char * 76)


def print_status(package, status):
    initial_indent = "%22s: " % package
    indent = ' ' * 24
    print(fill(str(status), width=76,
               initial_indent=initial_indent,
               subsequent_indent=indent))


def print_message(message):
    indent = ' ' * 24 + "* "
    print(fill(str(message), width=76,
               initial_indent=indent,
               subsequent_indent=indent))


def print_raw(section):
    print(section)


def convert_qt_version(version):
    version = '%x' % version
    temp = []
    while len(version) > 0:
        version, chunk = version[:-2], version[-2:]
        temp.insert(0, str(int(chunk, 16)))
    return '.'.join(temp)


def check_for_qt4():
    try:
        from PyQt4 import pyqtconfig
    except ImportError:
        print_status("Qt4", "no")
        return False
    else:
        print_status("Qt4", "Qt: %s, PyQt4: %s" %
                     (convert_qt_version(
                         pyqtconfig.Configuration().qt_version),
                         pyqtconfig.Configuration().pyqt_version_str))
        return True


def check_for_pyside():
    try:
        from PySide import __version__
        from PySide import QtCore
    except ImportError:
        print_status("PySide", "no")
        return False
    else:
        print_status("PySide", "Qt: %s, PySide: %s" %
                     (QtCore.__version__, __version__))
        return True


def check_for_numpy(min_version):
    try:
        import numpy
    except ImportError:
        print_status("numpy", "no")
        print_message("You must install numpy %s or later to build glue." %
                      min_version)
        return False

    expected_version = version.LooseVersion(min_version)
    found_version = version.LooseVersion(numpy.__version__)
    if not found_version >= expected_version:
        print_message(
            'numpy %s or later is required; you have %s' %
            (min_version, numpy.__version__))
        return False

    print_status("numpy", numpy.__version__)
    return True


def check_warn_version(func):
    def do_check(min_version=None):

        #check if installed
        found_version = func()
        module = func.__name__.split('_')[-1]
        if found_version is None:
            print_status(module, 'no')
            return False
        print_status(module, found_version)

        #check if recent enough
        if min_version is None:
            return True

        expected = version.LooseVersion(min_version)
        found = version.LooseVersion(found_version)

        if not found >= expected:
            print_message("%s %s or later recommended: you have %s" %
                          (module, min_version, found_version))
            return False
        return True

    return do_check


def version_standard(mod):
    """Return version number of module by looking for module.__version__"""
    try:
        return __import__(mod).__version__
    except ImportError:
        return None


@check_warn_version
def check_for_scipy():
    return version_standard('scipy')


@check_warn_version
def check_for_matplotlib():
    return version_standard('matplotlib')


@check_warn_version
def check_for_astropy():
    return version_standard('astropy')


@check_warn_version
def check_for_atpy():
    return version_standard('atpy')


@check_warn_version
def check_for_aplpy():
    return version_standard('aplpy')


@check_warn_version
def check_for_ipython():
    return version_standard('IPython')


@check_warn_version
def check_for_pil():
    try:
        from PIL import Image
        return Image.VERSION
    except ImportError:
        return None


@check_warn_version
def check_for_pytest():
    return version_standard('pytest')


@check_warn_version
def check_for_mock():
    return version_standard('mock')
