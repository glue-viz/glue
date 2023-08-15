import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.table.qt is deprecated, use glue_qt.viewers.table) instead', GlueDeprecationWarning)
from glue_qt.viewers.table import *  # noqa
