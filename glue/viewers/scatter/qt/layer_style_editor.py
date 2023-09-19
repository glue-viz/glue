import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.scatter.qt.layer_style_editor is deprecated, use glue_qt.viewers.scatter.layer_style_editor instead', GlueDeprecationWarning)
from glue_qt.viewers.scatter.layer_style_editor import *  # noqa
