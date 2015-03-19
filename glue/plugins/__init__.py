import sys


def register_plugins():
    from ..config import qt_client, exporters, tool_registry
    qt_client.lazy_add('glue.plugins.ginga_viewer')
    exporters.lazy_add('glue.plugins.export_d3po')
    exporters.lazy_add('glue.plugins.export_plotly')
    tool_registry.lazy_add('glue.plugins.tools.pv_slicer')
    tool_registry.lazy_add('glue.plugins.tools.spectrum_tool')


def load_plugin(plugin):
    """
    Load plugin referred to by name 'plugin'
    """
    # When Python 2.6 is no longer supported, we can use:
    # import importlib
    # module = importlib.import_module(plugin)
    __import__(plugin)
    module = sys.modules[plugin]
    if hasattr(module, 'setup'):
        module.setup()
    else:
        raise AttributeError("Plugin {0} should define 'setup' function".format(plugin))
