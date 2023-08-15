import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt.toolbar is deprecated, use glue_qt.viewers.matplotlib.toolbar) instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib.toolbar import *  # noqa
