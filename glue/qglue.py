import warnings
from glue.utils.error import GlueDeprecationWarning

warnings.warn('Importing from glue.qglue is deprecated, use glue_qt.qglue instead', GlueDeprecationWarning)

from glue_qt.qglue import *  # noqa


def parse_data(*args, **kwargs):
    warnings.warn('glue.qglue.parse_data is deprecated, use glue.core.parsers.parse_data instead', GlueDeprecationWarning)
    from glue.core.parsers import parse_data
    return parse_data(*args, **kwargs)


def parse_links(*args, **kwargs):
    warnings.warn('glue.qglue.parse_links is deprecated, use glue.core.parsers.parse_links instead', GlueDeprecationWarning)
    from glue.core.parsers import parse_links
    return parse_links(*args, **kwargs)
