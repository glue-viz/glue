import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt is deprecated, use glue_qt.viewers.common) instead', GlueDeprecationWarning)
from glue_qt.viewers.common import *  # noqa
