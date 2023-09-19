import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt is deprecated, use glue_qt.viewers.image instead', GlueDeprecationWarning)
from glue_qt.viewers.image import *  # noqa
