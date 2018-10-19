from __future__ import absolute_import, division, print_function


__version__ = '0.13.4'

try:
    from glue._githash import __githash__, __dev_value__  # noqa
    __version__ += __dev_value__
except Exception:
    pass
