from glue.core.visual import VisualAttributes
from matplotlib.cm import get_cmap


def test_VA_preferred_cmap():
    # Not a real CMAP array
    va = VisualAttributes(preferred_cmap=1)
    assert va.preferred_cmap == 1

    # Not a valid string for a CMAP
    va = VisualAttributes(preferred_cmap="not_a_cmap")
    assert va.preferred_cmap is None

    # get_cmap cmap name
    va = VisualAttributes(preferred_cmap="viridis")
    assert va.preferred_cmap == get_cmap("viridis")

    # formal cmap name
    va = VisualAttributes(preferred_cmap="Viridis")
    assert va.preferred_cmap == get_cmap("viridis")
