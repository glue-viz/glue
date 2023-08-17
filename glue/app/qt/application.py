import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.application is deprecated, use glue_qt.app.application instead', GlueDeprecationWarning)
from glue_qt.app.application import *  # noqa
