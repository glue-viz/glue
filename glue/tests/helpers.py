# Define decorators that can be used for pytest tests

import pytest

from distutils.version import LooseVersion


def make_skipper(module, label=None, version=None):
    label = label or module
    try:
        mod = __import__(module)
        if version:
            assert LooseVersion(mod.__version__) >= LooseVersion(version)
        installed = True
    except ImportError:
        installed = False
    return installed, pytest.mark.skipif(str(not installed), reason='Requires %s' % label)


ASTROPY_INSTALLED, requires_astropy = make_skipper('astropy',
                                                   label='Astropy')

ASTROPY_GE_03_INSTALLED, requires_astropy_ge_03 = make_skipper('astropy',
                                                               label='Astropy >= 0.3',
                                                               version='0.3')

ASTRODENDRO_INSTALLED, requires_astrodendro = make_skipper('astrodendro')

SCIPY_INSTALLED, requires_scipy = make_skipper('scipy',
                                               label='SciPy')

PIL_INSTALLED, requires_pil = make_skipper('pil', label='PIL')

SKIMAGE_INSTALLED, requires_skimage = make_skipper('skimage',
                                                   label='scikit-image')

XLRD_INSTALLED, requires_xlrd = make_skipper('xlrd')

PLOTLY_INSTALLED, requires_plotly = make_skipper('plotly')

IPYTHON_GE_012_INSTALLED, requires_ipython_ge_012 = make_skipper('IPython',
                                                                 label='IPython >= 0.12',
                                                                 version='0.12')


requires_pil_or_skimage = pytest.mark.skipif(str(not SKIMAGE_INSTALLED and not PIL_INSTALLED),
                                             reason='Requires PIL or scikit-image')
