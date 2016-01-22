from __future__ import absolute_import, division, print_function

import os
import zlib
import tempfile
from contextlib import contextmanager

from mock import MagicMock

from glue.core.application_base import Application
from glue import core


@contextmanager
def make_file(contents, suffix, decompress=False):
    """Context manager to write data to a temporary file,
    and delete on exit

    :param contents: Data to write. string
    :param suffix: File suffix. string
    """
    if decompress:
        contents = zlib.decompress(contents)

    try:
        _, fname = tempfile.mkstemp(suffix=suffix)
        with open(fname, 'wb') as outfile:
            outfile.write(contents)
        yield fname
    finally:
        try:
            os.unlink(fname)
        except WindowsError:  # on Windows the unlink can fail
            pass


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
