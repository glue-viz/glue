import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.preferences is deprecated, use glue_qt.app.preferences) instead', GlueDeprecationWarning)
from glue_qt.app.preferences import *  # noqa
