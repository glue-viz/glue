import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.scatter.qt is deprecated, use glue_qt.viewers.scatter) instead', GlueDeprecationWarning)
from glue_qt.viewers.scatter import *  # noqa
