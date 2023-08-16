import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.scatter.qt.options_widget is deprecated, use glue_qt.viewers.scatter.options_widget instead', GlueDeprecationWarning)
from glue_qt.viewers.scatter.options_widget import *  # noqa
