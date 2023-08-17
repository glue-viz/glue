import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.slice_widget is deprecated, use glue_qt.viewers.image.slice_widget instead', GlueDeprecationWarning)
from glue_qt.viewers.image.slice_widget import *  # noqa
