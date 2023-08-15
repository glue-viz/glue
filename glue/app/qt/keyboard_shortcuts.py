import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.keyboard_shortcuts is deprecated, use glue_qt.app.keyboard_shortcuts) instead', GlueDeprecationWarning)
from glue_qt.app.keyboard_shortcuts import *  # noqa
