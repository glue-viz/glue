__version__ = '0.5.2'

try:
    from ._githash import __githash__, __dev_value__
    __version__ += __dev_value__
except Exception:
    pass
