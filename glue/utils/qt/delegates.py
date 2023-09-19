import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.delegates is deprecated, use glue_qt.utils.delegates instead', GlueDeprecationWarning)
from glue_qt.utils.delegates import *  # noqa
