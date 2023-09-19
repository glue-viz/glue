import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.layer_style_editor is deprecated, use glue_qt.viewers.image.layer_style_editor instead', GlueDeprecationWarning)
from glue_qt.viewers.image.layer_style_editor import *  # noqa
