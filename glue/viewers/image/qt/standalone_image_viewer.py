import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.viewers.image.qt.standalone_image_viewer is deprecated, use glue_qt.viewers.image.standalone_image_viewer) instead', GlueDeprecationWarning)
from glue_qt.viewers.image.standalone_image_viewer import *  # noqa
