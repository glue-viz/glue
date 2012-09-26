#!/usr/bin/env python
import sys
import os
import optparse

from glue.qt.glue_application import GlueApplication
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
    """
    parser = optparse.OptionParser(usage=usage,
                                   version="%s" % __version__)

    parser.add_option('-x', '--execute', action='store_true', dest='script',
                      help="Execute FILE as a python script", default=False)
    parser.add_option('-g', action='store_true', dest='restore',
                      help="Restore glue session from FILE", default=False)
    parser.add_option('-c', '--config', type='string', dest='config',
                      metavar='CONFIG',
                      help='use CONFIG as configuration file')

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

    if opts.script and len(args) != 1:
        err_msg = "Must provide a script\n"
    elif opts.restore and len(args) != 1:
        err_msg = "Must provide a .glu file\n"
    elif opts.config is not None and not os.path.exists(opts.config):
        err_msg = "Could not find configuration file: %s" % opts.config

    return err_msg


def die_on_error(msg):
    """Decorator that catches errors, displays a popup message, and quits"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import traceback
                from PyQt4.QtGui import QMessageBox
                m = "%s\n%s" % (msg, e)
                detail = str(traceback.format_exc())
                qmb = QMessageBox(QMessageBox.Critical, "Error", m)
                qmb.setDetailedText(detail)
                qmb.exec_()
                sys.exit(1)
        return wrapper
    return decorator


@die_on_error("Error restoring Glue session")
def restore_session(gluefile):
    """Load a .glu file and return a DataCollection, Hub tuple"""
    from pickle import Unpickler
    with open(gluefile) as f:
        state = Unpickler(f).load()
    return state


@die_on_error("Error reading data file")
def load_data_files(datafiles):
    """Load data files and return a DataCollection"""
    import glue
    from glue.core.data_factories import auto_data, load_data

    dc = glue.core.DataCollection()
    for df in datafiles:
        dc.append(load_data(df, auto_data))
    return dc


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

    datafiles = datafiles or []

    data, hub = None, None

    if gluefile is not None:
        data, hub = restore_session(gluefile)

    if config is not None:
        glue.env = glue.config.load_configuration(search_path=[config])

    if datafiles:
        data = load_data_files(datafiles)

    hub = hub or glue.core.Hub(data)

    ga = GlueApplication(data_collection=data, hub=hub)
    sys.exit(ga.exec_())


def execute_script(script):
    """ Run a python script and exit.

    Provides a way for people with pre-installed binaries to use
    the glue library
    """
    execfile(script)
    sys.exit(0)


def main():
    opt, args = parse(sys.argv[1:])
    if opt.restore:
        start_glue(args[0], config=opt.config)
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
            start_glue(args[0], config=opt.config)
        elif has_file or has_files:
            start_glue(datafiles=args)
        else:
            start_glue()


if __name__ == "__main__":
    main()
