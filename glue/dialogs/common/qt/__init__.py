import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.common.qt is deprecated, use glue_qt.dialogs.common) instead', GlueDeprecationWarning)
from glue_qt.dialogs.common import *  # noqa
