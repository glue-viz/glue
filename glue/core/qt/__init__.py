import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt is deprecated, use glue_qt.core instead', GlueDeprecationWarning)
from glue_qt.core import *  # noqa
