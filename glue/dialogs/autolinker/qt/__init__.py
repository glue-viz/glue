import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.autolinker.qt is deprecated, use glue_qt.dialogs.autolinker) instead', GlueDeprecationWarning)
from glue_qt.dialogs.autolinker import *  # noqa
