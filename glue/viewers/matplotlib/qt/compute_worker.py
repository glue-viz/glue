import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.matplotlib.qt.compute_worker is deprecated, use glue_qt.viewers.matplotlib.compute_worker instead', GlueDeprecationWarning)
from glue_qt.viewers.matplotlib.compute_worker import *  # noqa
