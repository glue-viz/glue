import warnings
from glue.utils.error import GlueDeprecationWarning
warnings.warn('Importing from glue.dialogs.subset_facet.qt is deprecated, use glue_qt.dialogs.subset_facet instead', GlueDeprecationWarning)
from glue_qt.dialogs.subset_facet import *  # noqa
