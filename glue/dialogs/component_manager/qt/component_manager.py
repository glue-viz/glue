import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.component_manager.qt.component_manager is deprecated, use glue_qt.dialogs.component_manager.component_manager instead', GlueDeprecationWarning)
from glue_qt.dialogs.component_manager.component_manager import *  # noqa
