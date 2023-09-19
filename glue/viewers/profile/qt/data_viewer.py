import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.profile.qt.data_viewer is deprecated, use glue_qt.viewers.profile.data_viewer instead', GlueDeprecationWarning)
from glue_qt.viewers.profile.data_viewer import *  # noqa
