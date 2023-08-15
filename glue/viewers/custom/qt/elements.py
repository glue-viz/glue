import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.custom.qt.elements is deprecated, use glue_qt.viewers.custom.elements) instead', GlueDeprecationWarning)
from glue_qt.viewers.custom.elements import *  # noqa
