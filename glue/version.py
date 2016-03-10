__version__ = '0.8.0.dev'

try:
    from glue._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
