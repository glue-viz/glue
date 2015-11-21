import sys


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
