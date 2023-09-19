import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.app.qt.layer_tree_widget is deprecated, use glue_qt.app.layer_tree_widget instead', GlueDeprecationWarning)
from glue_qt.app.layer_tree_widget import *  # noqa
