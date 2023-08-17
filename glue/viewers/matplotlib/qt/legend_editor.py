import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt.legend_editor is deprecated, use glue_qt.viewers.matplotlib.legend_editor instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib.legend_editor import *  # noqa
