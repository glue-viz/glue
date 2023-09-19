import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.data_viewer is deprecated, use glue_qt.viewers.common.data_viewer instead', GlueDeprecationWarning)
from glue_qt.viewers.common.data_viewer import *  # noqa
