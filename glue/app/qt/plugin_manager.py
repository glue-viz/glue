import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.plugin_manager is deprecated, use glue_qt.app.plugin_manager) instead', GlueDeprecationWarning)
from glue_qt.app.plugin_manager import *  # noqa
