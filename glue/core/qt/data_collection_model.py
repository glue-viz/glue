import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.data_collection_model is deprecated, use glue_qt.core.data_collection_model instead', GlueDeprecationWarning)
from glue_qt.core.data_collection_model import *  # noqa
