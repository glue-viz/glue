import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt.axes_editor is deprecated, use glue_qt.viewers.matplotlib.axes_editor instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib.axes_editor import *  # noqa
