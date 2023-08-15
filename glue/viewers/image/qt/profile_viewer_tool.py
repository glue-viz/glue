import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.profile_viewer_tool is deprecated, use glue_qt.viewers.image.profile_viewer_tool) instead', GlueDeprecationWarning)
from glue_qt.viewers.image.profile_viewer_tool import *  # noqa
