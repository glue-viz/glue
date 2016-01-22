from __future__ import absolute_import, division, print_function

import pytest

from .. import message as msg


def test_invalid_subset_msg():
    with pytest.raises(TypeError) as exc:
        msg.SubsetMessage(None)
    assert exc.value.args[0].startswith('Sender must be a subset')


def test_invalid_data_msg():
    with pytest.raises(TypeError) as exc:
        msg.DataMessage(None)
    assert exc.value.args[0].startswith('Sender must be a data')


def test_invalid_data_collection_msg():
    with pytest.raises(TypeError) as exc:
        msg.DataCollectionMessage(None)
    assert exc.value.args[0].startswith('Sender must be a DataCollection')
