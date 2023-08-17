import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.terminal is deprecated, use glue_qt.app.terminal instead', GlueDeprecationWarning)
from glue_qt.app.terminal import *  # noqa
