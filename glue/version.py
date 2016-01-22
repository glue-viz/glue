__version__ = '0.7.0.dev'

try:
    from glue._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
