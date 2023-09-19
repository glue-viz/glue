import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.colors is deprecated, use glue_qt.utils.colors instead', GlueDeprecationWarning)
from glue_qt.utils.colors import *  # noqa
