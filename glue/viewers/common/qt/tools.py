import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.tools is deprecated, use glue_qt.viewers.common.tools instead', GlueDeprecationWarning)
from glue_qt.viewers.common.tools import *  # noqa
