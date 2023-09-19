import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.utils.qt.autocomplete_widget is deprecated, use glue_qt.utils.autocomplete_widget instead', GlueDeprecationWarning)
from glue_qt.utils.autocomplete_widget import *  # noqa
