# The following funtion is a thin wrapper around iter_entry_points. The reason it
# is in this separate file is that when making the Mac app, py2app doesn't
# support entry points, so we replace this function with a version that has the
# entry points we want hardcoded. If this function was in glue/main.py, the
# reference to the iter_plugin_entry_points function in load_plugin would be
# evaluated at compile time rather than at runtime, so the patched version
# wouldn't be used.

from __future__ import absolute_import, division, print_function

import os
from collections import defaultdict


def iter_plugin_entry_points():
    from pkg_resources import iter_entry_points
    return iter_entry_points(group='glue.plugins', name=None)


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

        # Import at runtime because some tests change this value. We also don't
        # just import the variable directly otherwise it is cached.
        from glue import config
        cfg_dir = config.CFG_DIR

        plugin_cfg = os.path.join(cfg_dir, 'plugins.cfg')

        from glue.external.six.moves import configparser

        config = configparser.ConfigParser()
        read = config.read(plugin_cfg)

        if len(read) == 0 or not config.has_section('plugins'):
            return cls()

        plugins = {}
        for name, enabled in config.items('plugins'):
            plugins[name] = bool(int(enabled))

        self = cls(plugins=plugins)

        return self

    def save(self):

        # Import at runtime because some tests change this value. We also don't
        # just import the variable directly otherwise it is cached.
        from glue import config
        cfg_dir = config.CFG_DIR

        plugin_cfg = os.path.join(cfg_dir, 'plugins.cfg')

        from glue.external.six.moves import configparser

        config = configparser.ConfigParser()
        config.add_section('plugins')

        for key in sorted(self.plugins):
            config.set('plugins', key, value=str(int(self.plugins[key])))

        if not os.path.exists(cfg_dir):
            os.mkdir(cfg_dir)

        with open(plugin_cfg, 'w') as fout:
            config.write(fout)

    def filter(self, keep):
        """
        Keep only certain plugins.

        This is used to filter out plugins that are not installed.
        """
        for key in list(self.plugins.keys())[:]:
            if not key in keep:
                self.plugins.pop(key)
