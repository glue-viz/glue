import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.style_dialog is deprecated, use glue_qt.core.style_dialog instead', GlueDeprecationWarning)
from glue_qt.core.style_dialog import *  # noqa
