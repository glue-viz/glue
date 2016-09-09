#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import optparse

from glue import __version__
from glue.logger import logger

try:
    from glue.utils.qt.decorators import die_on_error
except ImportError:
    from glue.utils.decorators import die_on_error


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


@die_on_error("Error restoring Glue session")
def restore_session(gluefile):
    """Load a .glu file and return a DataCollection, Hub tuple"""
    from glue.app.qt import GlueApplication
    return GlueApplication.restore_session(gluefile)


@die_on_error("Error reading data file")
def load_data_files(datafiles):
    """Load data files and return a list of datasets"""
    from glue.core.data_factories import auto_data, load_data

    datasets = []
    for df in datafiles:
        datasets.append(load_data(df, auto_data))
    return datasets


def run_tests():
    from glue import test
    test()


def start_glue(gluefile=None, config=None, datafiles=None, maximized=True):
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

    """
    import glue
    from glue.app.qt import GlueApplication

    # Start off by loading plugins. We need to do this before restoring
    # the session or loading the configuration since these may use existing
    # plugins.
    load_plugins()

    datafiles = datafiles or []

    hub = None

    if gluefile is not None:
        app = restore_session(gluefile)
        return app.start()

    if config is not None:
        glue.env = glue.config.load_configuration(search_path=[config])

    data_collection = glue.core.DataCollection()
    hub = data_collection.hub

    session = glue.core.Session(data_collection=data_collection, hub=hub)
    ga = GlueApplication(session=session, maximized=maximized)

    if datafiles:
        datasets = load_data_files(datafiles)
        ga.add_datasets(data_collection, datasets)

    # ga.show()
    # splash.close()
    # ga.raise_()
    # QApplication.instance().processEvents()
    return ga.start()


@die_on_error("Error running script")
def execute_script(script):
    """ Run a python script and exit.

    Provides a way for people with pre-installed binaries to use
    the glue library
    """
    with open(script) as fin:
        exec(fin.read())
    sys.exit(0)


def get_splash():
    """Instantiate a splash screen"""
    from qtpy import QtGui, QtWidgets
    from qtpy.QtCore import Qt
    import os

    pth = os.path.join(os.path.dirname(__file__), 'logo.png')
    pm = QtGui.QPixmap(pth)
    splash = QtWidgets.QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    splash.show()

    return splash


def main(argv=sys.argv):

    opt, args = parse(argv[1:])

    if opt.verbose:
        logger.setLevel("INFO")

    logger.info("Input arguments: %s", sys.argv)

    # Global keywords for Glue startup.
    kwargs = {'config': opt.config,
              'maximized': not opt.nomax}

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


def load_plugins():

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

    for item in iter_plugin_entry_points():

        if item.module_name not in _installed_plugins:
            _installed_plugins.add(item.name)

        if item.module_name in _loaded_plugins:
            logger.info("Plugin {0} already loaded".format(item.name))
            continue

        if not config.plugins[item.name]:
            continue

        try:
            function = item.load()
            function()
        except Exception as exc:
            logger.info("Loading plugin {0} failed "
                        "(Exception: {1})".format(item.name, exc))
        else:
            logger.info("Loading plugin {0} succeeded".format(item.name))
            _loaded_plugins.add(item.module_name)

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
