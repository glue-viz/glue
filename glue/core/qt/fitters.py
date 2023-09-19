import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.fitters is deprecated, use glue_qt.core.fitters instead', GlueDeprecationWarning)
from glue_qt.core.fitters import *  # noqa
