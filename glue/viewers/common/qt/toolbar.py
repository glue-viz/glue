import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.toolbar is deprecated, use glue_qt.viewers.common.toolbar) instead', GlueDeprecationWarning)
from glue_qt.viewers.common.toolbar import *  # noqa
