import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.threading is deprecated, use glue_qt.utils.threading) instead', GlueDeprecationWarning)
from glue_qt.utils.threading import *  # noqa
