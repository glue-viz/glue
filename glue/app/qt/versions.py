import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.versions is deprecated, use glue_qt.app.versions instead', GlueDeprecationWarning)
from glue_qt.app.versions import *  # noqa
