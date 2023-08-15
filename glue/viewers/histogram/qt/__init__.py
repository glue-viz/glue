import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.histogram.qt is deprecated, use glue_qt.viewers.histogram) instead', GlueDeprecationWarning)
from glue_qt.viewers.histogram import *  # noqa
