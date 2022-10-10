from glue.core.visual import VisualAttributes
from glue.utils.matplotlib import MATPLOTLIB_GE_36

if MATPLOTLIB_GE_36:
    from matplotlib import colormaps
else:
    from matplotlib.cm import get_cmap

import pytest


def test_VA_preferred_cmap():
    # Not a real CMAP array - errors
    with pytest.raises(TypeError, match="`preferred_cmap` must be a string or an instance of "
                       "a matplotlib.colors.Colormap"):
        VisualAttributes(preferred_cmap=1)

    # Not a valid string / known key [mpl 3.6+] for a CMAP - errors
    with pytest.raises(ValueError, match="not_a_cmap is not a valid colormap name."):
        VisualAttributes(preferred_cmap="not_a_cmap")

    viridis_cmap = colormaps["viridis"] if MATPLOTLIB_GE_36 else get_cmap("viridis")

    # get_cmap cmap name
    va = VisualAttributes(preferred_cmap="viridis")
    assert va.preferred_cmap == viridis_cmap
    # formal cmap name
    va = VisualAttributes(preferred_cmap="Viridis")
    assert va.preferred_cmap == viridis_cmap

    # Valid Colormap
    va = VisualAttributes(preferred_cmap=viridis_cmap)
    assert va.preferred_cmap == viridis_cmap

    # None is allowed - it is the default
    va = VisualAttributes(preferred_cmap=None)
    assert va.preferred_cmap is None
