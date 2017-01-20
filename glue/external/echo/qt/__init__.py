try:
    import pytest
    pytest.importorskip('qtpy')
except ImportError:  # pragma: nocover
    pass

from .connect import *   # noqa
from .autoconnect import *  # noqa
