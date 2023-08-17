import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.histogram.qt.layer_artist is deprecated, use glue_qt.viewers.histogram.layer_artist instead', GlueDeprecationWarning)
from glue_qt.viewers.histogram.layer_artist import *  # noqa
