import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.qglue is deprecated, use glue_qt.qglue) instead', GlueDeprecationWarning)
from glue_qt.qglue import *  # noqa
