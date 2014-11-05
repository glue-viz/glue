import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ...core import Data

from .test_image_client import _TestImageClientBase
from ..ginga_client import GingaClient, SubsetImage, BaseImage

from ginga.qtw.ImageViewCanvasQt import ImageViewCanvas
from ginga.misc import log


class TestGingaClient(_TestImageClientBase):

    def new_client(self, dc=None, canvas=None):
        from ...qt import get_qapp
        get_qapp()
        dc = dc or self.collect
        l = log.get_logger(name='ginga', log_stderr=True)
        canvas = ImageViewCanvas(l, render='widget')
        return GingaClient(dc, canvas)

    @pytest.mark.skipif(True, reason='unsupported by ginga')
    def skip(self):
        assert False

    test_add_scatter_layer = skip
    test_data_scatter_emphasis_updates_on_slice_change = skip
    test_scatter_persistent = skip
    test_scatter_sync = skip
    test_scatter_subsets_not_auto_added = skip
    test_scatter_layer_does_not_set_display_data = skip


class TestSubsetImage(object):

    def setup_method(self, method):
        x = np.arange(80).reshape(8, 10)
        d = Data(x=x, label='data')
        s = d.new_subset()
        s.subset_state = d.id['x'] > 30
        print(s.to_mask())

        self.subset = s
        self.x = x
        self.im = SubsetImage(s, np.s_[:, :])

        m = (s.to_mask() * 127).astype(np.uint8)
        self.base = BaseImage.BaseImage(data_np=m)

    def test_scaled_downsample(self):

        b1 = self.im.get_scaled_cutout_wdht(0, 0, 7, 9, 4, 4)
        b2 = self.base.get_scaled_cutout_wdht(0, 0, 7, 9, 4, 4)

        assert_array_equal(b1.data[..., 3], b2.data)

    def test_scaled_upsample(self):

        b1 = self.im.get_scaled_cutout_wdht(0, 0, 7, 9, 40, 40)
        b2 = self.base.get_scaled_cutout_wdht(0, 0, 7, 9, 40, 40)

        assert_array_equal(b1.data[..., 3], b2.data)

    def test_transpose_slice(self):
        m = (self.subset.to_mask() * 127).T

        im1 = SubsetImage(self.subset, np.s_[:, :], transpose=True)
        im2 = BaseImage.BaseImage(data_np=m)

        view = np.s_[0:3, 3:4]
        assert_array_equal(im1._slice(view)[..., 3], im2._slice(view))

        assert im1.shape[:2] == im2.shape
