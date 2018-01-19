from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from glue.external.six.moves import range as xrange
from glue import core

from .. import command as c
from .. import roi
from ..data_factories import tabular_data
from .util import simple_session, simple_catalog


class TestCommandStack(object):

    def setup_method(self, method):
        self.session = simple_session()
        self.stack = self.session.command_stack

    def make_command(self):
        return MagicMock(c.Command)

    def make_data(self):
        with simple_catalog() as path:
            cmd = c.LoadData(path=path, factory=tabular_data)
            data = self.stack.do(cmd)
        return data

    def test_do(self):

        c1 = self.make_command()
        self.stack.do(c1)

        c1.do.assert_called_once_with(self.session)

    def test_undo(self):
        c1, c2 = self.make_command(), self.make_command()

        self.stack.do(c1)
        self.stack.do(c2)

        self.stack.undo()
        c2.undo.assert_called_once_with(self.session)

        self.stack.undo()
        c1.undo.assert_called_once_with(self.session)

    def test_redo(self):
        c1, c2 = self.make_command(), self.make_command()

        self.stack.do(c1)
        self.stack.do(c2)

        self.stack.undo()
        self.stack.redo()

        c2.undo.assert_called_once_with(self.session)

        assert c2.do.call_count == 2
        assert c2.undo.call_count == 1
        assert c1.do.call_count == 1
        assert c1.undo.call_count == 0

    def test_max_undo(self):
        cmds = [self.make_command() for _ in xrange(c.MAX_UNDO + 1)]

        for cmd in cmds:
            self.stack.do(cmd)

        for cmd in cmds[:-1]:
            self.stack.undo()

        with pytest.raises(IndexError):
            self.stack.undo()

    def test_invalid_redo(self):
        with pytest.raises(IndexError) as exc:
            self.stack.redo()
        assert exc.value.args[0] == 'No commands to redo'

    def test_load_data(self):
        data = self.make_data()
        np.testing.assert_array_equal(data['a'], [1, 3])

    def test_add_data(self):
        data = self.make_data()
        cmd = c.AddData(data=data)

        self.stack.do(cmd)
        assert len(self.session.data_collection) == 1

        self.stack.undo()
        assert len(self.session.data_collection) == 0

    def test_remove_data(self):
        data = self.make_data()

        add = c.AddData(data=data)
        remove = c.RemoveData(data=data)

        self.stack.do(add)
        assert len(self.session.data_collection) == 1

        self.stack.do(remove)
        assert len(self.session.data_collection) == 0

        self.stack.undo()
        assert len(self.session.data_collection) == 1

    def test_new_data_viewer(self):
        cmd = c.NewDataViewer(viewer=None, data=None)
        v = self.stack.do(cmd)

        self.session.application.new_data_viewer.assert_called_once_with(None, None)

        self.stack.undo()
        v.close.assert_called_once_with(warn=False)

    def test_apply_roi(self):
        x = core.Data(x=[1, 2, 3])
        s = x.new_subset()
        dc = self.session.data_collection
        dc.append(x)

        r = MagicMock(roi.Roi)
        client = MagicMock(core.client.Client)
        client.data = dc

        cmd = c.ApplyROI(data_collection=dc, roi=r,
                         apply_func=client.apply_roi)

        self.stack.do(cmd)
        client.apply_roi.assert_called_once_with(r)

        old_state = s.subset_state
        s.subset_state = MagicMock(spec_set=core.subset.SubsetState)

        self.stack.undo()
        assert s.subset_state is old_state
