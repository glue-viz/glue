import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.custom.qt.custom_viewer is deprecated, use glue_qt.viewers.custom.custom_viewer instead', GlueDeprecationWarning)
from glue_qt.viewers.custom.custom_viewer import *  # noqa
