import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt.widget is deprecated, use glue_qt.viewers.matplotlib.widget) instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib.widget import *  # noqa
