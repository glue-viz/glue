from . import export_d3po
from . import export_plotly
from . import ginga_viewer


def register_plugins():
    from ..config import qt_client
    qt_client.register_plugin('glue.plugins.ginga_viewer')
    

def load_plugin(plugin):
    """
    Load plugin referred to by name 'plugin'
    """
    # TODO: load plugin here
    print("LOADING", plugin)