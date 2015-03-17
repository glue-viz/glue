from . import export_d3po
from . import export_plotly
from . import ginga_viewer


def register_plugins():
    from ..config import qt_client, exporters
    qt_client.add_plugin('glue.plugins.ginga_viewer')
    exporters.add_plugin('glue.plugins.export_d3po')
    exporters.add_plugin('glue.plugins.export_plotly')
    

def load_plugin(plugin):
    """
    Load plugin referred to by name 'plugin'
    """
    import importlib
    module = importlib.import_module(plugin)
    module.load_plugin()
