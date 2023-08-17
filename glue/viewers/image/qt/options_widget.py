import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.options_widget is deprecated, use glue_qt.viewers.image.options_widget instead', GlueDeprecationWarning)
from glue_qt.viewers.image.options_widget import *  # noqa
