#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import optparse
from .logger import logger

from glue import __version__


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


def die_on_error(msg):
    """Decorator that catches errors, displays a popup message, and quits"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import traceback
                from . import qt
                from .external.qt.QtGui import QMessageBox
                m = "%s\n%s" % (msg, e)
                detail = str(traceback.format_exc())
                if len(m) > 500:
                    detail = "Full message:\n\n%s\n\n%s" % (m, detail)
                    m = m[:500] + '...'

                qmb = QMessageBox(QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.show()
                qmb.raise_()
                qmb.exec_()
                sys.exit(1)
        return wrapper
    return decorator


@die_on_error("Error restoring Glue session")
def restore_session(gluefile):
    """Load a .glu file and return a DataCollection, Hub tuple"""
    from glue.qt.glue_application import GlueApplication
    return GlueApplication.restore(gluefile)


@die_on_error("Error reading data file")
def load_data_files(datafiles):
    """Load data files and return a DataCollection"""
    import glue
    from glue.core.data_factories import auto_data, load_data

    dc = glue.core.DataCollection()
    for df in datafiles:
        dc.append(load_data(df, auto_data))
    return dc


def run_tests():
    from glue import test
    test.main()


def start_glue(gluefile=None, config=None, datafiles=None):
    """Run a glue session and exit

    :param gluefile: An optional .glu file to restore
    :type gluefile: str

    :param config: An optional configuration file to use
    :type config: str

    :param datafiles: An optional list of data files to load
    :type datafiles: list of str
    """
    import glue
    from glue.qt.glue_application import GlueApplication

    datafiles = datafiles or []

    data, hub = None, None

    if gluefile is not None:
        app = restore_session(gluefile)
        return app.start()

    if config is not None:
        glue.env = glue.config.load_configuration(search_path=[config])

    if datafiles:
        data = load_data_files(datafiles)

    if not data:
        data = glue.core.DataCollection()

    hub = data.hub

    session = glue.core.Session(data_collection=data, hub=hub)
    ga = GlueApplication(session=session)
    #ga.show()
    #splash.close()
    #ga.raise_()
    #QApplication.instance().processEvents()
    return ga.start()


@die_on_error("Error running script")
def execute_script(script):
    """ Run a python script and exit.

    Provides a way for people with pre-installed binaries to use
    the glue library
    """
    execfile(script)
    sys.exit(0)


def get_splash():
    """Instantiate a splash screen"""
    from .external.qt.QtGui import QSplashScreen, QPixmap
    from .external.qt.QtCore import Qt
    import os

    pth = os.path.join(os.path.dirname(__file__), 'logo.png')
    pm = QPixmap(pth)
    splash = QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    splash.show()

    return splash


def main(argv=sys.argv):

    opt, args = parse(argv[1:])

    if opt.verbose:
        logger.setLevel("INFO")

    logger.info("Input arguments: %s", sys.argv)

    if opt.test:
        return run_tests()
    elif opt.restore:
        start_glue(gluefile=args[0], config=opt.config)
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
            start_glue(gluefile=args[0], config=opt.config)
        elif has_file or has_files:
            start_glue(datafiles=args, config=opt.config)
        else:
            start_glue(config=opt.config)


if __name__ == "__main__":
    sys.exit(main(sys.argv))  # prama: no cover
