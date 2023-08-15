import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt is deprecated, use glue_qt.viewers.matplotlib) instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib import *  # noqa
