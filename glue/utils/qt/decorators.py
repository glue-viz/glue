import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.decorators is deprecated, use glue_qt.utils.decorators instead', GlueDeprecationWarning)
from glue_qt.utils.decorators import *  # noqa
