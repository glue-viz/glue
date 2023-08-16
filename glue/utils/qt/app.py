import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.app is deprecated, use glue_qt.utils.app instead', GlueDeprecationWarning)
from glue_qt.utils.app import *  # noqa
