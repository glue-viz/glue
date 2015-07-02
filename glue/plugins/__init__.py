import sys


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
