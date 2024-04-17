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

    import setuptools
    logger.info("Loading external plugins using "
                "setuptools=={0}".format(setuptools.__version__))

    from glue._plugin_helpers import iter_plugin_entry_points, PluginConfig
    config = PluginConfig.load()

    if plugins_to_load is None:
        plugins_to_load = [i.value for i in list(iter_plugin_entry_points())]
        if require_qt_plugins:
            plugins_to_require = [*REQUIRED_PLUGINS, *REQUIRED_PLUGINS_QT]
        else:
            plugins_to_require = REQUIRED_PLUGINS
    else:
        plugins_to_require = plugins_to_load
    n_plugins = len(plugins_to_require)

    for i_plugin, item in enumerate(list(iter_plugin_entry_points())):
        if item.value.replace(':setup', '') in plugins_to_load:
            if item.module not in _installed_plugins:
                _installed_plugins.add(item.name)

            if item.module in _loaded_plugins:
                logger.info("Plugin {0} already loaded".format(item.name))
                continue

            # loads all plugins, want to make this more customisable
            if not config.plugins[item.name]:
                continue

        # We don't use item.load() because that then checks requirements of all
        # the imported packages, which can lead to errors like this one that
        # don't really matter:
        #
        # Exception: (pytest 2.6.0 (/Users/tom/miniconda3/envs/py27/lib/python2.7/site-packages),
        #             Requirement.parse('pytest>=2.8'), set(['astropy']))
        #
        # Just to be clear, this kind of error does indicate that there is an
        # old version of a package in the environment, but this can confuse
        # users as importing astropy directly would work (as setuptools then
        # doesn't do a stringent test of dependency versions). Often this kind
        # of error can occur if there is a conda version of a package and an
        # older pip version.

        try:
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
