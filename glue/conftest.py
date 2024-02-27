import os
import sys

from glue.config import CFG_DIR as CFG_DIR_ORIG

STDERR_ORIGINAL = sys.stderr

ON_APPVEYOR = os.environ.get('APPVEYOR', 'False') == 'True'


def pytest_runtest_teardown(item, nextitem):
    sys.stderr = STDERR_ORIGINAL
    global start_dir
    os.chdir(start_dir)


def pytest_addoption(parser):
    parser.addoption("--no-optional-skip", action="store_true", default=False,
                     help="don't skip any tests with optional dependencies")


start_dir = None


def pytest_configure(config):

    global start_dir
    start_dir = os.path.abspath('.')

    os.environ['GLUE_TESTING'] = 'True'

    from glue._mpl_backend import set_mpl_backend
    set_mpl_backend()

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

    # Force loading of plugins
    from glue.main import load_plugins
    load_plugins()


def pytest_unconfigure(config):

    os.environ.pop('GLUE_TESTING')

    # Reset configuration directory to original one
    from glue import config
    config.CFG_DIR = CFG_DIR_ORIG
