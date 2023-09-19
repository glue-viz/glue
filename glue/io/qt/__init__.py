import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.io.qt is deprecated, use glue_qt.io instead', GlueDeprecationWarning)
from glue_qt.io import *  # noqa
