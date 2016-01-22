from __future__ import absolute_import, division, print_function

from contextlib import contextmanager

from mock import MagicMock

from glue import core
from glue.core.application_base import Application
from glue.tests.helpers import make_file


@contextmanager
def simple_catalog():
    """Context manager to create a temporary data file

    :param suffix: File suffix. string
    """
    with make_file(b'#a, b\n1, 2\n3, 4', '.csv') as result:
        yield result


def simple_session():
    collect = core.data_collection.DataCollection()
    hub = core.hub.Hub()
    result = core.Session(data_collection=collect, hub=hub,
                          application=MagicMock(Application),
                          command_stack=core.CommandStack())
    result.command_stack.session = result
    return result
