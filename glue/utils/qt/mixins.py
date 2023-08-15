import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.mixins is deprecated, use glue_qt.utils.mixins) instead', GlueDeprecationWarning)
from glue_qt.utils.mixins import *  # noqa
