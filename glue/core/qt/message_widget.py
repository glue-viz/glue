import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.message_widget is deprecated, use glue_qt.core.message_widget instead', GlueDeprecationWarning)
from glue_qt.core.message_widget import *  # noqa
