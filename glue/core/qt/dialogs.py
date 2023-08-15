import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.dialogs is deprecated, use glue_qt.core.dialogs) instead', GlueDeprecationWarning)
from glue_qt.core.dialogs import *  # noqa
