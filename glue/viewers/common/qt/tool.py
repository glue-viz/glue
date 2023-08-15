import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.common.qt.tool is deprecated, use glue_qt.viewers.common.tool) instead', GlueDeprecationWarning)
from glue_qt.viewers.common.tool import *  # noqa
