# Set up configuration variables

__all__ = ['custom_viewer', 'qglue', 'test']

import os

import sys

import importlib.metadata

__version__ = importlib.metadata.version('glue-core')

from ._mpl_backend import MatplotlibBackendSetter
sys.meta_path.append(MatplotlibBackendSetter())

from glue.viewers.custom.helper import custom_viewer

# Load user's configuration file
from .config import load_configuration
env = load_configuration()

from .qglue import qglue

from .main import load_plugins  # noqa


def test(no_optional_skip=False):
    from pytest import main
    root = os.path.abspath(os.path.dirname(__file__))
    args = [root, '-x']
    if no_optional_skip:
        args.append('--no-optional-skip')
    return main(args=args)


from glue._settings_helpers import load_settings
load_settings()


# In PyQt 5.5+, PyQt overrides the default exception catching and fatally
# crashes the Qt application without printing out any details about the error.
# Below we revert the exception hook to the original Python one. Note that we
# can't just do sys.excepthook = sys.__excepthook__ otherwise PyQt will detect
# the default excepthook is in place and override it.


def handle_exception(exc_type, exc_value, exc_traceback):
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception
