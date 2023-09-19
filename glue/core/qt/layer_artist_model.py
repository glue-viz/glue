import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.core.qt.layer_artist_model is deprecated, use glue_qt.core.layer_artist_model instead', GlueDeprecationWarning)
from glue_qt.core.layer_artist_model import *  # noqa
