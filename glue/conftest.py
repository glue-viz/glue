import os

def pytest_addoption(parser):
    parser.addoption("--no-optional-skip", action="store_true",
                     help="don't skip any tests with optional dependencies")


def pytest_configure(config):
    if config.getoption('no_optional_skip'):
        from .tests import helpers
        for attr in helpers.__dict__:
            if attr.startswith('requires_'):
                # The following line replaces the decorators with a function
                # that does noting, effectively disabling it.
                setattr(helpers, attr, lambda f: f)


def pytest_report_header(config):
    from . import __version__
    glue_version = "%20s:\t%s" % ("glue", __version__)
    from ._deps import get_status
    return os.linesep + glue_version + os.linesep + os.linesep + get_status()
