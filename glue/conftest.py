from __future__ import absolute_import, division, print_function

import os
import sys

from glue.config import CFG_DIR as CFG_DIR_ORIG

try:
    import objgraph
except ImportError:
    OBJGRAPH_INSTALLED = False
else:
    OBJGRAPH_INSTALLED = True

STDERR_ORIGINAL = sys.stderr

ON_APPVEYOR = os.environ.get('APPVEYOR', 'False') == 'True'


def pytest_addoption(parser):
    parser.addoption("--no-optional-skip", action="store_true", default=False,
                     help="don't skip any tests with optional dependencies")


def pytest_runtest_teardown(item, nextitem):
    sys.stderr = STDERR_ORIGINAL


def pytest_configure(config):

    os.environ['GLUE_TESTING'] = 'True'

    if config.getoption('no_optional_skip'):
        from glue.tests import helpers
        for attr in helpers.__dict__:
            if attr.startswith('requires_'):
                # The following line replaces the decorators with a function
                # that does noting, effectively disabling it.
                setattr(helpers, attr, lambda f: f)

    # Make sure we don't affect the real glue config dir
    import tempfile
    from glue import config
    config.CFG_DIR = tempfile.mkdtemp()

    # Start up QApplication, if the Qt code is present
    try:
        from glue.utils.qt import get_qapp
    except ImportError:
        pass
    else:
        get_qapp()

    # Force loading of plugins
    from glue.main import load_plugins
    load_plugins()


def pytest_report_header(config):
    from glue import __version__
    glue_version = "%20s:\t%s" % ("glue", __version__)
    from glue._deps import get_status
    return os.linesep + glue_version + os.linesep + os.linesep + get_status()


def pytest_unconfigure(config):

    os.environ.pop('GLUE_TESTING')

    # Reset configuration directory to original one
    from glue import config
    config.CFG_DIR = CFG_DIR_ORIG

    # Remove reference to QApplication to prevent segmentation fault on PySide
    try:
        from glue.utils.qt import app
        app.qapp = None
    except ImportError:  # for when we run the tests without the qt directories
        pass

    if OBJGRAPH_INSTALLED and not ON_APPVEYOR:

        # Make sure there are no lingering references to GlueApplication
        obj = objgraph.by_type('GlueApplication')
        if len(obj) > 0:
            objgraph.show_backrefs(objgraph.by_type('GlueApplication'))
            raise ValueError('There are {0} remaining references to GlueApplication'.format(len(obj)))

        # Uncomment when checking for memory leaks
        # objgraph.show_most_common_types(limit=100)
