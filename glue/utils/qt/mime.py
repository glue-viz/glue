import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.mime is deprecated, use glue_qt.utils.mime) instead', GlueDeprecationWarning)
from glue_qt.utils.mime import *  # noqa
