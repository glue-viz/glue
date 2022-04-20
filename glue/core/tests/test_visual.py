from glue.core.visual import VisualAttributes
from matplotlib.cm import get_cmap


def test_VA_preferred_cmap():
    # Not a real CMAP array
    abc = VisualAttributes(preferred_cmap=1)
    assert abc.preferred_cmap == 1

    # Not a valid string for a CMAP
    abc = VisualAttributes(preferred_cmap="not_a_cmap")
    assert abc.preferred_cmap is None

    # get_cmap cmap name
    abc = VisualAttributes(preferred_cmap="viridis")
    assert abc.preferred_cmap == get_cmap("viridis")

    # formal cmap name
    abc = VisualAttributes(preferred_cmap="Viridis")
    assert abc.preferred_cmap == get_cmap("viridis")
