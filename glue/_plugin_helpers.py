# The following funtion is a thin wrapper around iter_entry_points. The reason it
# is in this separate file is that when making the Mac app, py2app doesn't
# support entry points, so we replace this function with a version that has the
# entry points we want hardcoded. If this function was in glue/main.py, the
# reference to the iter_plugin_entry_points function in load_plugin would be
# evaluated at compile time rather than at runtime, so the patched version
# wouldn't be used.

import os
from collections import defaultdict

def iter_plugin_entry_points():
    from pkg_resources import iter_entry_points
    return iter_entry_points(group='glue.plugins', name=None)


CFG_DIR = os.path.join(os.path.expanduser('~'), '.glue')
PLUGIN_CFG =  os.path.join(CFG_DIR, 'plugins.cfg')


class PluginConfig(object):

    def __init__(self, plugins={}):
        self.plugins = defaultdict(lambda: True)
        self.plugins.update(plugins)

    def __str__(self):
        string = ""
        for plugin in sorted(self.plugins):
            string += "{0}: {1}\n".format(plugin, self.plugins[plugin])
        return string

    @classmethod
    def load(cls):

        from .external.six.moves import configparser

        config = configparser.ConfigParser()
        read = config.read(PLUGIN_CFG)

        if len(read) == 0 or not 'plugins' in config:
            return cls()

        plugins = {}
        for key in sorted(config['plugins']):
            print(config['plugins'][key])
            plugins[key] = bool(int(config['plugins'][key]))

        self = cls(plugins=plugins)

        return self

    def save(self):

        from .external.six.moves import configparser

        config = configparser.ConfigParser()
        config['plugins'] = {}

        for key in sorted(self.plugins):
            config['plugins'][key] = str(int(self.plugins[key]))

        if not os.path.exists(CFG_DIR):
            os.mkdir(CFG_DIR)

        with open(PLUGIN_CFG, 'w') as fout:
            config.write(fout)
