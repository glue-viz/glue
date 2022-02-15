import numpy as np
from glue.core import Data, DataCollection
from glue.core.coordinates import AffineCoordinates, IdentityCoordinates
from glue.plugins.tools.pv_slicer.pv_sliced_data import PVSlicedData
from glue.core.link_helpers import LinkSame


class TestPVSlicedData:

    def setup_method(self, method):
        matrix = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]])

        self.data1 = Data(x=np.arange(120).reshape((6, 5, 4)), coords=AffineCoordinates(matrix))
        self.data2 = Data(y=np.arange(120).reshape((6, 5, 4)), coords=IdentityCoordinates(n_dim=3))

        self.dc = DataCollection([self.data1, self.data2])

        self.dc.add_link(LinkSame(self.data1.world_component_ids[0],
                                  self.data2.world_component_ids[0]))

        self.dc.add_link(LinkSame(self.data1.world_component_ids[1],
                                  self.data2.world_component_ids[1]))

        self.dc.add_link(LinkSame(self.data1.world_component_ids[2],
                                  self.data2.world_component_ids[2]))

        # TODO: the paths in the next two PVSlicedData objects are meant to
        # be the same conceptually. We should make sure we formalize this with
        # a UUID. Also should use proper links.

        x1 = [0, 2, 5]
        y1 = [1, 2, 3]

        self.pvdata1 = PVSlicedData(self.data1,
                                    self.data1.pixel_component_ids[1], y1,
                                    self.data1.pixel_component_ids[2], x1)

        x2, y2, _ = self.data2.coords.world_to_pixel_values(*self.data1.coords.pixel_to_world_values(x1, y1, 0))

        self.pvdata2 = PVSlicedData(self.data2,
                                    self.data2.pixel_component_ids[1], y2,
                                    self.data2.pixel_component_ids[2], x2)

    def test_fixed_resolution_buffer_linked(self):
        result = self.pvdata1.compute_fixed_resolution_buffer(bounds=[(0, 5, 15), (0, 6, 20)],
                                                              target_data=self.pvdata2,
                                                              target_cid=self.data1.id['x'])
        assert result.shape == (15, 20)
