#!/usr/bin/env python

from importlib import import_module

from glue.logger import logger
from glue._plugin_helpers import REQUIRED_PLUGINS, REQUIRED_PLUGINS_QT


_loaded_plugins = set()
_installed_plugins = set()


def load_plugins(splash=None, require_qt_plugins=False, plugins_to_load=None):
    """

    Parameters
    ----------
    splash : default: None
        instance of splash http rendering service
    require_qt_plugins : boolean default: False
        whether to use qt plugins defined in constant REQUIRED_PLUGINS_QT
    plugins_to_load : list
        desired valid plugin strings

    Returns
    -------

    """

    # Search for plugins installed via entry_points. Basically, any package can
    # define plugins for glue, and needs to define an entry point using the
    # following format:
    #
    # entry_points = """
    # [glue.plugins]
    # webcam_importer=glue_exp.importers.webcam:setup
    # vizier_importer=glue_exp.importers.vizier:setup
    # dataverse_importer=glue_exp.importers.dataverse:setup
    # """
    #
    # where ``setup`` is a function that does whatever is needed to set up the
    # plugin, such as add items to various registries.

    logger.info("Loading external plugins")

    from glue._plugin_helpers import iter_plugin_entry_points, PluginConfig
    config = PluginConfig.load()

    if plugins_to_load is None:
        plugins_to_load = [i.module for i in list(iter_plugin_entry_points())]
        if require_qt_plugins:
            plugins_to_require = [*REQUIRED_PLUGINS, *REQUIRED_PLUGINS_QT]
        else:
            plugins_to_require = REQUIRED_PLUGINS
    else:
        plugins_to_require = plugins_to_load
    n_plugins = len(plugins_to_require)

    for i_plugin, item in enumerate(list(iter_plugin_entry_points())):
        if item.module in plugins_to_load:
            if item.module not in _installed_plugins:
                _installed_plugins.add(item.name)

            if item.module in _loaded_plugins:
                logger.info("Plugin {0} already loaded".format(item.name))
                continue

            # loads all plugins, want to make this more customisable
            if not config.plugins[item.name]:
                continue

        try:
            # item.load() (importlib.metadata.EntryPoint.load) doesn't *always*
            # return a module, so,  using import_module is safer here.
            module = import_module(item.module)
            function = getattr(module, item.attr)
            function()
        except Exception as exc:
            # Here we check that some of the 'core' plugins load well and
            # raise an actual exception if not.
            if item.module in plugins_to_require:
                raise
            else:
                logger.info("Loading plugin {0} failed "
                            "(Exception: {1})".format(item.name, exc))
        else:
            logger.info("Loading plugin {0} succeeded".format(item.name))
            _loaded_plugins.add(item.module)

        if splash is not None:
            splash.set_progress(100. * i_plugin / float(n_plugins))

    try:
        config.save()
    except Exception as e:
        logger.warn("Failed to load plugin configuration")

    # Reload the settings now that we have loaded plugins, since some plugins
    # may have added some settings. Note that this will not re-read settings
    # that were previously read.
    from glue._settings_helpers import load_settings
    load_settings()


def list_loaded_plugins():
    """
    Function to list all plugins that are currently loaded
    """
    return sorted(_loaded_plugins)


def list_available_plugins():
    """
    Function to list all available plugins
    """
    from glue._plugin_helpers import iter_plugin_entry_points

    plugins_load_list = [i.module for i in list(iter_plugin_entry_points())]
    return plugins_load_list
