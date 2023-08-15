import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.mdi_area is deprecated, use glue_qt.app.mdi_area) instead', GlueDeprecationWarning)
from glue_qt.app.mdi_area import *  # noqa
