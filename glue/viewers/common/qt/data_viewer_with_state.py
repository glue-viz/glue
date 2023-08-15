import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.data_viewer_with_state is deprecated, use glue_qt.viewers.common.data_viewer_with_state) instead', GlueDeprecationWarning)
from glue_qt.viewers.common.data_viewer_with_state import *  # noqa
