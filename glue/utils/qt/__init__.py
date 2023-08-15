import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt is deprecated, use glue_qt.utils) instead', GlueDeprecationWarning)
from glue_qt.utils import *  # noqa
