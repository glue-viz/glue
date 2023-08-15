import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.base_widget is deprecated, use glue_qt.viewers.common.base_widget) instead', GlueDeprecationWarning)
from glue_qt.viewers.common.base_widget import *  # noqa
