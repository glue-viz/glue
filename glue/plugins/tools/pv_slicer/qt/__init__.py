import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.plugins.tools.pv_slicer.qt is deprecated, use glue_qt.plugins.tools.pv_slicer instead', GlueDeprecationWarning)
from glue_qt.plugins.tools.pv_slicer import *  # noqa
