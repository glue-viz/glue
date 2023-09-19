import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.histogram.qt.data_viewer is deprecated, use glue_qt.viewers.histogram.data_viewer instead', GlueDeprecationWarning)
from glue_qt.viewers.histogram.data_viewer import *  # noqa
