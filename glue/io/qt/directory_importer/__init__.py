import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.io.qt.directory_importer is deprecated, use glue_qt.io.directory_importer) instead', GlueDeprecationWarning)
from glue_qt.io.directory_importer import *  # noqa
