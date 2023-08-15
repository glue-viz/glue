import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.io.qt.subset_mask is deprecated, use glue_qt.io.subset_mask) instead', GlueDeprecationWarning)
from glue_qt.io.subset_mask import *  # noqa
