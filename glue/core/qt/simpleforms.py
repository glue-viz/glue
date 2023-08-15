import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.simpleforms is deprecated, use glue_qt.core.simpleforms) instead', GlueDeprecationWarning)
from glue_qt.core.simpleforms import *  # noqa
