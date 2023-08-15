import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.icons.qt is deprecated, use glue_qt.icons) instead', GlueDeprecationWarning)
from glue_qt.icons import *  # noqa
