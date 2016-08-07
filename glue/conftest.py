from __future__ import absolute_import, division, print_function

import os

from glue.config import CFG_DIR as CFG_DIR_ORIG


def pytest_addoption(parser):
    parser.addoption("--no-optional-skip", action="store_true",
                     help="don't skip any tests with optional dependencies")


def pytest_configure(config):

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
        app = get_qapp()

    # Force loading of plugins
    from glue.main import load_plugins
    load_plugins()


def pytest_report_header(config):
    from glue import __version__
    glue_version = "%20s:\t%s" % ("glue", __version__)
    from glue._deps import get_status
    return os.linesep + glue_version + os.linesep + os.linesep + get_status()


def pytest_unconfigure(config):
    from glue import config
    config.CFG_DIR = CFG_DIR_ORIG
