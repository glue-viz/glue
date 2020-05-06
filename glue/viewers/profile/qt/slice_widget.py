from glue.core.coordinates import LegacyCoordinates
from glue.core.coordinate_helpers import dependent_axes, world_axis
from glue.viewers.common.qt.data_slice_widget import SliceWidget
from glue.viewers.image.state import AggregateSlice
from glue.utils.decorators import avoid_circular
from glue.core.data_derived import IndexedData, SlicedData

__all__ = ['ProfileMultiSliceWidgetHelper']


class ProfileMultiSliceWidgetHelper(object):

    def __init__(self, viewer_state=None, session=None, layout=None, *args, **kwargs):

        self.viewer_state = viewer_state

        self.session = session

        self.profile_layout = layout
        self.profile_layout.setSpacing(4)
        self.profile_layout.setContentsMargins(0, 3, 0, 3)

        self.viewer_state.add_callback('x_att_pixel', self.sync_sliders_from_state)
        self.viewer_state.add_callback('slices', self.sync_sliders_from_state)
        self.viewer_state.add_callback('reference_data', self.sync_sliders_from_state)

        self._sliced = None
        self._indexed = None

        self.slider_state = None
        self._sliders = []

        self._reference_data = None
        self._x_att = None

        self.sync_sliders_from_state()

    @property
    def data(self):
        return self.viewer_state.reference_data

    def _clear(self):

        for _ in range(self.profile_layout.count()):
            self.profile_layout.takeAt(0)

        for s in self._sliders:
            if s is not None:
                s.close()

        self._sliders = []

    @avoid_circular
    def sync_state_from_sliders(self, *args):
        slices = []
        for i, slider in enumerate(self._sliders):
            if slider is not None:
                slices.append(slider.state.slice_center)
            else:
                slices.append(slice(None))

        self.viewer_state.slices = tuple(slices)

        if self.viewer_state.reference_data is not self._reference_data:
            self._reference_data = self.viewer_state.reference_data

        for dataset in self.session.data_collection:
            if isinstance(dataset, IndexedData):
                self._indexed = dataset.indices

        if self._sliced is None:
            self._sliced = SlicedData(self._reference_data, self.viewer_state.slices)
            self.session.data_collection.append(self._sliced)
        else:
            self.session.data_collection.remove(self._sliced)
            self._sliced = SlicedData(self._reference_data, self.viewer_state.slices)
            self.session.data_collection.append(self._sliced)

    @avoid_circular
    def sync_sliders_from_state(self, *args):
        if self.data is None or self.viewer_state.x_att_pixel is None:
            return

        # Update sliders if needed

        if (self.viewer_state.reference_data is not self._reference_data or
                self.viewer_state.x_att_pixel is not self._x_att):

            self._reference_data = self.viewer_state.reference_data
            self._x_att = self.viewer_state.x_att_pixel

            self._clear()

            for i in range(self.data.ndim):

                if i == self.viewer_state.x_att_pixel.axis:
                    self._sliders.append(None)
                    continue

                # TODO: For now we simply pass a single set of world coordinates,
                # but we will need to generalize this in future. We deliberately
                # check the type of data.coords here since we want to treat
                # subclasses differently.
                if getattr(self.data, 'coords') is not None and type(self.data.coords) != LegacyCoordinates:
                    world = world_axis(self.data.coords, self.data,
                                       pixel_axis=self.data.ndim - 1 - i,
                                       world_axis=self.data.ndim - 1 - i)
                    world_unit = self.data.coords.world_axis_units[self.data.ndim - 1 - i]

                    world_warning = len(dependent_axes(self.data.coords, i)) > 1
                    world_label = self.data.world_component_ids[i].label
                else:
                    world = None
                    world_unit = None
                    world_warning = False
                    world_label = self.data.pixel_component_ids[i].label

                slider = SliceWidget(world_label,
                                     hi=self.data.shape[i] - 1, world=world,
                                     world_unit=world_unit, world_warning=world_warning)

                self.slider_state = slider.state

                self.slider_state.add_callback('slice_center', self.sync_state_from_sliders)
                self._sliders.append(slider)
                self.profile_layout.addWidget(slider)

        for i in range(self.data.ndim):
            if self._sliders[i] is not None:
                if isinstance(self.viewer_state.slices[i], AggregateSlice):
                    self._sliders[i].state.slice_center = self.viewer_state.slices[i].center
                else:
                    if self.viewer_state.slices[i] == slice(None):
                        self._sliders[i].state.slice_center = 0
                    else:
                        self._sliders[i].state.slice_center = self.viewer_state.slices[i]
