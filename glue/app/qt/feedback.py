import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.feedback is deprecated, use glue_qt.app.feedback) instead', GlueDeprecationWarning)
from glue_qt.app.feedback import *  # noqa
