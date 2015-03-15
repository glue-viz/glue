from . import export_d3po
from . import export_plotly
from . import ginga_viewer


def load_all_plugins():
    """
    Load built-in plugins
    """

    from .ginga_viewer import load_ginga_viewer_plugin
    load_ginga_viewer_plugin()
