import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.data_slice_widget is deprecated, use glue_qt.viewers.common.data_slice_widget instead', GlueDeprecationWarning)
from glue_qt.viewers.common.data_slice_widget import *  # noqa
