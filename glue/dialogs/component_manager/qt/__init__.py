import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.component_manager.qt is deprecated, use glue_qt.dialogs.component_manager) instead', GlueDeprecationWarning)
from glue_qt.dialogs.component_manager import *  # noqa
