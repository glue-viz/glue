import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.widget_properties is deprecated, use glue_qt.utils.widget_properties instead', GlueDeprecationWarning)
from glue_qt.utils.widget_properties import *  # noqa
