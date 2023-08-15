import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.save_data is deprecated, use glue_qt.app.save_data) instead', GlueDeprecationWarning)
from glue_qt.app.save_data import *  # noqa
