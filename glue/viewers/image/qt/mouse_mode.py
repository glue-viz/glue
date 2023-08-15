import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.mouse_mode is deprecated, use glue_qt.viewers.image.mouse_mode) instead', GlueDeprecationWarning)
from glue_qt.viewers.image.mouse_mode import *  # noqa
