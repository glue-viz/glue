import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.splash_screen is deprecated, use glue_qt.app.splash_screen instead', GlueDeprecationWarning)
from glue_qt.app.splash_screen import *  # noqa
