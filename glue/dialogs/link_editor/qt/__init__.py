import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.link_editor.qt is deprecated, use glue_qt.dialogs.link_editor instead', GlueDeprecationWarning)
from glue_qt.dialogs.link_editor import *  # noqa
