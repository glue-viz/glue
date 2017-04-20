__version__ = '0.10.3'

try:
    from glue._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
