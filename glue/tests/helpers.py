# Define decorators that can be used for pytest tests

import os
import zlib
import tempfile
from contextlib import contextmanager
from distutils.version import LooseVersion

import pytest


def make_skipper(module, label=None, version=None):
    label = label or module
    try:
        if label == 'PyQt5':  # PyQt5 does not use __version__
            from PyQt5 import QtCore
            version_installed = QtCore.PYQT_VERSION_STR
        else:
            mod = __import__(module)
            version_installed = mod.__version__
        if version:
            assert LooseVersion(version_installed) >= LooseVersion(version)
        installed = True
    except (ImportError, AssertionError):
        installed = False
    return installed, pytest.mark.skipif(str(not installed), reason='Requires %s' % label)


ASTROPY_INSTALLED, requires_astropy = make_skipper('astropy',
                                                   label='Astropy')

MATPLOTLIB_GE_22, requires_matplotlib_ge_22 = make_skipper('matplotlib', version='2.2')

ASTRODENDRO_INSTALLED, requires_astrodendro = make_skipper('astrodendro')

SCIPY_INSTALLED, requires_scipy = make_skipper('scipy',
                                               label='SciPy')

PIL_INSTALLED, requires_pil = make_skipper('PIL', label='PIL')

SKIMAGE_INSTALLED, requires_skimage = make_skipper('skimage',
                                                   label='scikit-image')

XLRD_INSTALLED, requires_xlrd = make_skipper('xlrd')

OPENPYXL_INSTALLED, requires_openpyxl = make_skipper('openpyxl')

PYAVM_INSTALLED, requires_pyavm = make_skipper('pyavm')

PLOTLY_INSTALLED, requires_plotly = make_skipper('plotly')

IPYTHON_INSTALLED, requires_ipython = make_skipper('IPython')

requires_pil_or_skimage = pytest.mark.skipif(str(not SKIMAGE_INSTALLED and not PIL_INSTALLED),
                                             reason='Requires PIL or scikit-image')

PLOTLY_INSTALLED, requires_plotly = make_skipper('plotly')

H5PY_INSTALLED, requires_h5py = make_skipper('h5py')

PYQT5_INSTALLED, requires_pyqt5 = make_skipper('PyQt5')
PYSIDE2_INSTALLED, requires_pyside2 = make_skipper('PySide2')

HYPOTHESIS_INSTALLED, requires_hypothesis = make_skipper('hypothesis')

QT_INSTALLED = PYQT5_INSTALLED or PYSIDE2_INSTALLED

SPECTRAL_CUBE_INSTALLED, requires_spectral_cube = make_skipper('spectral_cube',
                                                               label='spectral-cube')

requires_qt = pytest.mark.skipif(str(not QT_INSTALLED),
                                 reason='An installation of Qt is required')

PYQT_GT_59, _ = make_skipper('PyQt5', version='5.10')

requires_pyqt_gt_59_or_pyside2 = pytest.mark.skipif(str(not PYQT_GT_59 and not PYSIDE2_INSTALLED),
                                                    reason='Requires PyQt > 5.9 or PySide2')


@contextmanager
def make_file(contents, suffix, decompress=False):
    """Context manager to write data to a temporary file,
    and delete on exit

    :param contents: Data to write. string
    :param suffix: File suffix. string
    """
    if decompress:
        contents = zlib.decompress(contents)

    try:
        _, fname = tempfile.mkstemp(suffix=suffix)
        with open(fname, 'wb') as outfile:
            outfile.write(contents)
        yield fname
    finally:
        try:
            os.unlink(fname)
        except WindowsError:  # on Windows the unlink can fail
            pass
