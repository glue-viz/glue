import warnings

from glue.utils.error import GlueDeprecationWarning

warnings.warn("The glue.external.qt subpackage is deprecated - see the v0.9 "
              "release announcement for more details", GlueDeprecationWarning)

from glue.utils.qt import get_qapp, load_ui
