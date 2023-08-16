import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.metadata is deprecated, use glue_qt.app.metadata instead', GlueDeprecationWarning)
from glue_qt.app.metadata import *  # noqa
