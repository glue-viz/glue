import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt is deprecated, use glue_qt.app) instead', GlueDeprecationWarning)
from glue_qt.app import *  # noqa
