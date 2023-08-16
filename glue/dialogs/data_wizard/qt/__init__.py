import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.data_wizard.qt is deprecated, use glue_qt.dialogs.data_wizard instead', GlueDeprecationWarning)
from glue_qt.dialogs.data_wizard import *  # noqa
