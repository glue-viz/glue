import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.mime is deprecated, use glue_qt.core.mime instead', GlueDeprecationWarning)
from glue_qt.core.mime import *  # noqa
