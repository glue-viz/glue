# Define decorators that can be used for pytest tests

import pytest
from distutils.version import LooseVersion


try:
    import astropy
    ASTROPY_INSTALLED = True
except:
    ASTROPY_INSTALLED = False
else:
    del astropy

requires_astropy = pytest.mark.skipif(str(not ASTROPY_INSTALLED), reason='Requires Astropy')

try:
    import astropy
    assert LooseVersion(astropy.__version__) >= LooseVersion('0.3')
    ASTROPY_GE_03_INSTALLED = True
except:
    ASTROPY_GE_03_INSTALLED = False
else:
    del astropy

requires_astropy_ge_03 = pytest.mark.skipif(str(not ASTROPY_GE_03_INSTALLED), reason='Requires Astropy >= 0.3')

try:
    import astrodendro
    ASTRODENDRO_INSTALLED = True
except ImportError:
    ASTRODENDRO_INSTALLED = False
else:
    del astrodendro

requires_astrodendro = pytest.mark.skipif(str(not ASTRODENDRO_INSTALLED), reason='Requires astrodendro')


try:
    import scipy
    SCIPY_INSTALLED = True
except ImportError:
    SCIPY_INSTALLED = False
else:
    del scipy

requires_scipy = pytest.mark.skipif(str(not SCIPY_INSTALLED), reason='Requires SciPy')


try:
    import PIL
    PIL_INSTALLED = True
except ImportError:
    PIL_INSTALLED = False
else:
    del PIL

requires_pil = pytest.mark.skipif(str(not PIL_INSTALLED), reason='Requires PIL')


try:
    import skimage
    SKIMAGE_INSTALLED = True
except ImportError:
    SKIMAGE_INSTALLED = False
else:
    del skimage

requires_skimage = pytest.mark.skipif(str(not SKIMAGE_INSTALLED), reason='Requires scikit-image')

requires_pil_or_skimage = pytest.mark.skipif(str(not SKIMAGE_INSTALLED and not PIL_INSTALLED), reason='Requires PIL or scikit-image')


try:
    import xlrd
    XLRD_INSTALLED = True
except ImportError:
    XLRD_INSTALLED = False
else:
    del xlrd
    
requires_xlrd = pytest.mark.skipif(str(not XLRD_INSTALLED), reason='Requires xlrd')
    
    
try:
    import plotly
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
else:
    del plotly

requires_plotly = pytest.mark.skipif(str(not PLOTLY_INSTALLED), reason='Requires plotly')


try:
    import IPython
    assert LooseVersion(IPython.__version__) <= LooseVersion('0.11')
    IPYTHON_GT_011_INSTALLED = True
except:
    IPYTHON_GT_011_INSTALLED = False
else:
    del IPython

requires_ipython_gt_011 = pytest.mark.skipif(str(not IPYTHON_GT_011_INSTALLED), reason='Requires IPython > 0.11')
