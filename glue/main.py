#!/usr/bin/env python

import sys
import optparse
from importlib import import_module

from glue import __version__
from glue.logger import logger


def parse(argv):
    """ Parse argument list, check validity

    :param argv: Arguments passed to program

    *Returns*
    A tuple of options, position arguments
    """
    usage = """usage: %prog [options] [FILE FILE...]

    # start a new session
    %prog

    # start a new session and load a file
    %prog image.fits

    #start a new session with multiple files
    %prog image.fits catalog.csv

    #restore a saved session
    %prog saved_session.glu
    or
    %prog -g saved_session.glu

    #run a script
    %prog -x script.py

    #run the test suite
    %prog -t
    """
    parser = optparse.OptionParser(usage=usage,
                                   version=str(__version__))

    parser.add_option('-x', '--execute', action='store_true', dest='script',
                      help="Execute FILE as a python script", default=False)
    parser.add_option('-g', action='store_true', dest='restore',
                      help="Restore glue session from FILE", default=False)
    parser.add_option('-t', '--test', action='store_true', dest='test',
                      help="Run test suite", default=False)
    parser.add_option('-c', '--config', type='string', dest='config',
                      metavar='CONFIG',
                      help='use CONFIG as configuration file')
    parser.add_option('-v', '--verbose', action='store_true',
                      help="Increase the vebosity level", default=False)
    parser.add_option('--no-maximized', action='store_true', dest='nomax',
                      help="Do not start Glue maximized", default=False)
    parser.add_option('--startup', dest='startup', type='string',
                      help="Startup actions to carry out", default='')
    parser.add_option('--auto-merge', dest='auto_merge', action='store_true',
                      help="Automatically merge any data passed on the command-line", default='')
    parser.add_option('--faulthandler', dest='faulthandler', action='store_true',
                      help="Run glue with the built-in faulthandler to debug segmentation faults", default=False)

    err_msg = verify(parser, argv)
    if err_msg:
        sys.stderr.write('\n%s\n' % err_msg)
        parser.print_help()
        sys.exit(1)

    return parser.parse_args(argv)


def verify(parser, argv):
    """ Check for input errors

    :param parser: OptionParser instance
    :param argv: Argument list
    :type argv: List of strings

    *Returns*
    An error message, or None
    """
    opts, args = parser.parse_args(argv)
    err_msg = None

    if opts.script and opts.restore:
        err_msg = "Cannot specify -g with -x"
    elif opts.script and opts.config:
        err_msg = "Cannot specify -c with -x"
    elif opts.script and len(args) != 1:
        err_msg = "Must provide a script\n"
    elif opts.restore and len(args) != 1:
        err_msg = "Must provide a .glu file\n"

    return err_msg


def load_data_files(datafiles):
    """Load data files and return a list of datasets"""

    from glue.core.data_factories import load_data

    datasets = []
    for df in datafiles:
        datasets.append(load_data(df))

    return datasets


def run_tests():
    from glue import test
    test()


def start_glue(gluefile=None, config=None, datafiles=None, maximized=True,
               startup_actions=None, auto_merge=False):
    """Run a glue session and exit

    Parameters
    ----------
    gluefile : str
        An optional ``.glu`` file to restore.
    config : str
        An optional configuration file to use.
    datafiles : str
        An optional list of data files to load.
    maximized : bool
        Maximize screen on startup. Otherwise, use default size.
    auto_merge : bool, optional
        Whether to automatically merge data passed in `datafiles` (default is `False`)
    """

    import glue

    # Some Qt modules are picky in terms of being imported before the
    # application is set up, so we import them here. We do it here rather
    # than in get_qapp since in the past, including this code in get_qapp
    # caused severe issues (e.g. segmentation faults) in plugin packages
    # during testing.
    try:
        from qtpy import QtWebEngineWidgets  # noqa
    except ImportError:  # Not all PyQt installations have this module
        pass

    from glue.utils.qt.decorators import die_on_error

    from glue.utils.qt import get_qapp
    app = get_qapp()

    splash = get_splash()
    splash.show()

    # Start off by loading plugins. We need to do this before restoring
    # the session or loading the configuration since these may use existing
    # plugins.
    load_plugins(splash=splash, require_qt_plugins=True)

    from glue.app.qt import GlueApplication

    datafiles = datafiles or []

    hub = None

    splash.close()

    if gluefile is not None:
        with die_on_error("Error restoring Glue session"):
            app = GlueApplication.restore_session(gluefile, show=False)
        return app.start(maximized=maximized)

    if config is not None:
        glue.env = glue.config.load_configuration(search_path=[config])

    data_collection = glue.core.DataCollection()
    hub = data_collection.hub

    splash.set_progress(100)

    session = glue.core.Session(data_collection=data_collection, hub=hub)
    ga = GlueApplication(session=session)

    if datafiles:
        with die_on_error("Error reading data file"):
            datasets = load_data_files(datafiles)
        ga.add_datasets(datasets, auto_merge=auto_merge)

    if startup_actions is not None:
        for name in startup_actions:
            ga.run_startup_action(name)

    return ga.start(maximized=maximized)


def execute_script(script):
    """ Run a python script and exit.

    Provides a way for people with pre-installed binaries to use
    the glue library
    """
    from glue.utils.qt.decorators import die_on_error
    with die_on_error("Error running script"):
        with open(script) as fin:
            exec(fin.read())
    sys.exit(0)


def get_splash():
    """Instantiate a splash screen"""
    from glue.app.qt.splash_screen import QtSplashScreen
    splash = QtSplashScreen()
    return splash


def main(argv=sys.argv):

    opt, args = parse(argv[1:])

    if opt.verbose:
        logger.setLevel("INFO")

    if opt.faulthandler:
        import faulthandler
        faulthandler.enable()

    logger.info("Input arguments: %s", sys.argv)

    # Global keywords for Glue startup.
    kwargs = {'config': opt.config,
              'maximized': not opt.nomax,
              'auto_merge': opt.auto_merge}

    if opt.startup:
        kwargs['startup_actions'] = opt.startup.split(',')

    if opt.test:
        return run_tests()
    elif opt.restore:
        start_glue(gluefile=args[0], **kwargs)
    elif opt.script:
        execute_script(args[0])
    else:
        has_file = len(args) == 1
        has_files = len(args) > 1
        has_py = has_file and args[0].endswith('.py')
        has_glu = has_file and args[0].endswith('.glu')
        if has_py:
            execute_script(args[0])
        elif has_glu:
            start_glue(gluefile=args[0], **kwargs)
        elif has_file or has_files:
            start_glue(datafiles=args, **kwargs)
        else:
            start_glue(**kwargs)


_loaded_plugins = set()
_installed_plugins = set()

REQUIRED_PLUGINS = ['glue.plugins.coordinate_helpers',
                    'glue.core.data_exporters',
                    'glue.io.formats.fits']


REQUIRED_PLUGINS_QT = ['glue.plugins.tools.pv_slicer.qt',
                       'glue.viewers.image.qt',
                       'glue.viewers.scatter.qt',
                       'glue.viewers.histogram.qt',
                       'glue.viewers.profile.qt',
                       'glue.viewers.table.qt']


def load_plugins(splash=None, require_qt_plugins=False):

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

    n_plugins = len(list(iter_plugin_entry_points()))

    for iplugin, item in enumerate(iter_plugin_entry_points()):
        if item.module not in _installed_plugins:
            _installed_plugins.add(item.name)

        if item.module in _loaded_plugins:
            logger.info("Plugin {0} already loaded".format(item.name))
            continue

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
        # of error can occur if there is a conda version of a package and and
        # older pip version.

        try:
            module = import_module(item.module)
            function = getattr(module, item.attr)
            function()
        except Exception as exc:
            # Here we check that some of the 'core' plugins load well and
            # raise an actual exception if not.
            if item.module in REQUIRED_PLUGINS:
                raise
            elif item.module in REQUIRED_PLUGINS_QT and require_qt_plugins:
                raise
            else:
                logger.info("Loading plugin {0} failed "
                            "(Exception: {1})".format(item.name, exc))
        else:
            logger.info("Loading plugin {0} succeeded".format(item.name))
            _loaded_plugins.add(item.module)

        if splash is not None:
            splash.set_progress(100. * iplugin / float(n_plugins))

    try:
        config.save()
    except Exception as e:
        logger.warn("Failed to load plugin configuration")

    # Reload the settings now that we have loaded plugins, since some plugins
    # may have added some settings. Note that this will not re-read settings
    # that were previously read.
    from glue._settings_helpers import load_settings
    load_settings()


if __name__ == "__main__":
    sys.exit(main(sys.argv))  # pragma: no cover
