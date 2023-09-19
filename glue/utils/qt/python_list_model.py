import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.python_list_model is deprecated, use glue_qt.utils.python_list_model instead', GlueDeprecationWarning)
from glue_qt.utils.python_list_model import *  # noqa
