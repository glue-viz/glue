__version__ = '0.5.0.dev'

try:
    from ._githash import __githash__
    __version__ += __githash__
except Exception:
    pass
