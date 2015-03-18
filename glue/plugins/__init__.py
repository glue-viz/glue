from . import export_d3po
from . import export_plotly
from . import ginga_viewer


def register_plugins():
    from ..config import qt_client, exporters
    qt_client.lazy_add('glue.plugins.ginga_viewer')
    exporters.lazy_add('glue.plugins.export_d3po')
    exporters.lazy_add('glue.plugins.export_plotly')
    

def load_plugin(plugin):
    """
    Load plugin referred to by name 'plugin'
    """
    import importlib
    module = importlib.import_module(plugin)
    if hasattr(module, 'setup'):
        module.setup()
    else:
        raise AttributeError("Plugin {0} should define 'setup' function".format(plugin))
