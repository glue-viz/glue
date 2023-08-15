import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.plugins.dendro_viewer.qt is deprecated, use glue_qt.plugins.dendro_viewer) instead', GlueDeprecationWarning)
from glue_qt.plugins.dendro_viewer import *  # noqa
