import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.custom.qt is deprecated, use glue_qt.viewers.custom) instead', GlueDeprecationWarning)
from glue_qt.viewers.custom import *  # noqa
