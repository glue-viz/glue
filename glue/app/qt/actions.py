import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.actions is deprecated, use glue_qt.app.actions) instead', GlueDeprecationWarning)
from glue_qt.app.actions import *  # noqa
