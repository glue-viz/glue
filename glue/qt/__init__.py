import warnings

from glue.utils.error import GlueDeprecationWarning

warnings.warn("The glue.qt subpackage is deprecated - see the v0.7 release "
              "announcement for more details", GlueDeprecationWarning)

# For compatibility
from glue.external.qt import get_qapp