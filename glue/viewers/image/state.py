import uuid
from collections import defaultdict

from glue.core import BaseData
from glue.config import colormaps
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.utils import defer_draw, view_shape
from echo import delay_callback
from glue.core.data_combo_helper import ManualDataComboHelper, ComponentIDComboHelper
from glue.core.exceptions import IncompatibleDataException

__all__ = ['ImageViewerState', 'ImageLayerState', 'ImageSubsetLayerState', 'AggregateSlice']


def get_sliced_data_maker(x_axis=None, y_axis=None, slices=None, data=None,
                          target_cid=None, reference_data=None, transpose=False):
    """
    Convenience function for use in exported Python scripts.
    """

    if reference_data is None:
        reference_data = data

    def get_array(bounds=None):

        full_bounds = list(slices)
        full_bounds[y_axis] = bounds[0]
        full_bounds[x_axis] = bounds[1]

        if isinstance(data, BaseData):
            array = data.compute_fixed_resolution_buffer(full_bounds, target_data=reference_data,
                                                         target_cid=target_cid, broadcast=False)
        else:
            array = data.data.compute_fixed_resolution_buffer(full_bounds, target_data=reference_data,
                                                              subset_state=data.subset_state, broadcast=False)

        if transpose:
            array = array.transpose()

        return array

    return get_array


class AggregateSlice(object):

    def __init__(self, slice=None, center=None, function=None):
        self.slice = slice
        self.center = center
        self.function = function

    def __gluestate__(self, context):
        state = dict(slice=context.do(self.slice),
                     center=self.center,
                     function=context.do(self.function))
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(slice=context.object(rec['slice']),
                   center=rec['center'],
                   function=context.object(rec['function']))


class ImageViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for an image viewer.
    """

    x_att = DDCProperty(docstring='The component ID giving the pixel component '
                                  'shown on the x axis')
    y_att = DDCProperty(docstring='The component ID giving the pixel component '
                                  'shown on the y axis')
    x_att_world = DDSCProperty(docstring='The component ID giving the world component '
                                         'shown on the x axis', default_index=-1)
    y_att_world = DDSCProperty(docstring='The component ID giving the world component '
                                         'shown on the y axis', default_index=-2)
    aspect = DDSCProperty(0, docstring='Whether to enforce square pixels (``equal``) '
                                            'or fill the axes (``auto``)')
    reference_data = DDSCProperty(docstring='The dataset that is used to define the '
                                            'available pixel/world components, and '
                                            'which defines the coordinate frame in '
                                            'which the images are shown')
    slices = DDCProperty(docstring='The current slice along all dimensions')
    color_mode = DDSCProperty(0, docstring='Whether each layer can have '
                                           'its own colormap (``Colormaps``) or '
                                           'whether each layer is assigned '
                                           'a single color (``One color per layer``)')

    dpi = DDCProperty(72, docstring='The resolution (in dots per inch) of density maps, if present')

    def __init__(self, **kwargs):

        super(ImageViewerState, self).__init__()

        self.limits_cache = {}

        # NOTE: we don't need to use StateAttributeLimitsHelper here because
        # we can simply call reset_limits below when x/y attributes change.
        # Using StateAttributeLimitsHelper makes things a lot slower.

        self.ref_data_helper = ManualDataComboHelper(self, 'reference_data')

        self.xw_att_helper = ComponentIDComboHelper(self, 'x_att_world',
                                                    numeric=False, datetime=False, categorical=False)

        self.yw_att_helper = ComponentIDComboHelper(self, 'y_att_world',
                                                    numeric=False, datetime=False, categorical=False)

        self.add_callback('reference_data', self._reference_data_changed, priority=1000)
        self.add_callback('layers', self._layers_changed, priority=1000)

        self.add_callback('x_att', self._on_xatt_change, priority=500)
        self.add_callback('y_att', self._on_yatt_change, priority=500)

        self.add_callback('x_att_world', self._on_xatt_world_change, priority=1000)
        self.add_callback('y_att_world', self._on_yatt_world_change, priority=1000)

        aspect_display = {'equal': 'Square Pixels', 'auto': 'Automatic'}
        ImageViewerState.aspect.set_choices(self, ['equal', 'auto'])
        ImageViewerState.aspect.set_display_func(self, aspect_display.get)

        ImageViewerState.color_mode.set_choices(self, ['Colormaps', 'One color per layer'])

        self.update_from_dict(kwargs)

    def reset_limits(self):

        if self.reference_data is None or self.x_att is None or self.y_att is None:
            return

        nx = self.reference_data.shape[self.x_att.axis]
        ny = self.reference_data.shape[self.y_att.axis]

        with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.x_min = -0.5
            self.x_max = nx - 0.5
            self.y_min = -0.5
            self.y_max = ny - 0.5
            # We need to adjust the limits in here to avoid triggering all
            # the update events then changing the limits again.
            self._adjust_limits_aspect()

    @property
    def _display_world(self):
        return getattr(self.reference_data, 'coords', None) is not None

    def _reference_data_changed(self, *args, force=False):
        # This signal can get emitted if just the choices but not the actual
        # reference data change, so we check here that the reference data has
        # actually changed
        if self.reference_data is not getattr(self, '_last_reference_data', None) or force:
            self._last_reference_data = self.reference_data
            # Note that we deliberately use nested delay_callback here, because
            # we want to make sure that x_att_world and y_att_world both get
            # updated first, then x_att and y_att can be changed, before
            # subsequent events are fired.
            with delay_callback(self, 'x_att', 'y_att'):
                with delay_callback(self, 'x_att_world', 'y_att_world', 'slices'):
                    if self._display_world:
                        self.xw_att_helper.pixel_coord = False
                        self.yw_att_helper.pixel_coord = False
                        self.xw_att_helper.world_coord = True
                        self.yw_att_helper.world_coord = True
                    else:
                        self.xw_att_helper.pixel_coord = True
                        self.yw_att_helper.pixel_coord = True
                        self.xw_att_helper.world_coord = False
                        self.yw_att_helper.world_coord = False
                    self._update_combo_att()
                    self._set_default_slices()
                    # We need to make sure that we update x_att and y_att
                    # at the same time before any other callbacks get called,
                    # so we do this here manually.
                    self._on_xatt_world_change()
                    self._on_yatt_world_change()

    def _layers_changed(self, *args):

        # The layers callback gets executed if anything in the layers changes,
        # but we only care about whether the actual set of 'layer' attributes
        # for all layers change.

        layers_data = self.layers_data
        layers_data_cache = getattr(self, '_layers_data_cache', [])

        if layers_data == layers_data_cache:
            return

        self._update_combo_ref_data()
        self._set_reference_data()
        self._update_syncing()

        self._layers_data_cache = layers_data

    def _update_syncing(self):

        # If there are multiple layers for a given dataset, we disable the
        # syncing by default.

        layer_state_by_data = defaultdict(list)

        for layer_state in self.layers:
            if isinstance(layer_state.layer, BaseData):
                layer_state_by_data[layer_state.layer].append(layer_state)

        for data, layer_states in layer_state_by_data.items():
            if len(layer_states) > 1:
                for layer_state in layer_states:
                    # Scatter layers don't have global_sync so we need to be
                    # careful here and make sure we return a default value
                    if getattr(layer_state, 'global_sync', False):
                        layer_state.global_sync = False

    def _update_combo_ref_data(self):
        self.ref_data_helper.set_multiple_data(self.layers_data)

    def _update_combo_att(self):
        with delay_callback(self, 'x_att_world', 'y_att_world'):
            if self.reference_data is None:
                self.xw_att_helper.set_multiple_data([])
                self.yw_att_helper.set_multiple_data([])
            else:
                self.xw_att_helper.set_multiple_data([self.reference_data])
                self.yw_att_helper.set_multiple_data([self.reference_data])

    def _update_priority(self, name):
        if name == 'layers':
            return 3
        elif name == 'reference_data':
            return 2
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    @defer_draw
    def _on_xatt_change(self, *args):
        if self.x_att is not None:
            if self._display_world:
                self.x_att_world = self.reference_data.world_component_ids[self.x_att.axis]
            else:
                self.x_att_world = self.x_att

    @defer_draw
    def _on_yatt_change(self, *args):
        if self.y_att is not None:
            if self._display_world:
                self.y_att_world = self.reference_data.world_component_ids[self.y_att.axis]
            else:
                self.y_att_world = self.y_att

    @defer_draw
    def _on_xatt_world_change(self, *args):

        if self.x_att_world is not None:

            with delay_callback(self, 'y_att_world', 'x_att'):

                if self.x_att_world == self.y_att_world:
                    if self._display_world:
                        world_ids = self.reference_data.world_component_ids
                    else:
                        world_ids = self.reference_data.pixel_component_ids
                    if self.x_att_world == world_ids[-1]:
                        self.y_att_world = world_ids[-2]
                    else:
                        self.y_att_world = world_ids[-1]

                if self._display_world:
                    index = self.reference_data.world_component_ids.index(self.x_att_world)
                    self.x_att = self.reference_data.pixel_component_ids[index]
                else:
                    self.x_att = self.x_att_world

    @defer_draw
    def _on_yatt_world_change(self, *args):

        if self.y_att_world is not None:

            with delay_callback(self, 'x_att_world', 'y_att'):

                if self.y_att_world == self.x_att_world:
                    if self._display_world:
                        world_ids = self.reference_data.world_component_ids
                    else:
                        world_ids = self.reference_data.pixel_component_ids
                    if self.y_att_world == world_ids[-1]:
                        self.x_att_world = world_ids[-2]
                    else:
                        self.x_att_world = world_ids[-1]

                if self._display_world:
                    index = self.reference_data.world_component_ids.index(self.y_att_world)
                    self.y_att = self.reference_data.pixel_component_ids[index]
                else:
                    self.y_att = self.y_att_world

    def _set_reference_data(self):
        if self.reference_data is None:
            for layer in self.layers:
                if isinstance(layer.layer, BaseData):
                    self.reference_data = layer.layer
                    return

    def _set_default_slices(self):
        # Need to make sure this gets called immediately when reference_data is changed
        if self.reference_data is None:
            self.slices = ()
        else:
            self.slices = (0,) * self.reference_data.ndim

    @property
    def numpy_slice_aggregation_transpose(self):
        """
        Returns slicing information usable by Numpy.

        This returns two objects: the first is an object that can be used to
        slice Numpy arrays and return a 2D array, and the second object is a
        boolean indicating whether to transpose the result.
        """
        if self.reference_data is None:
            return None
        slices = []
        agg_func = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis or i == self.y_att.axis:
                slices.append(slice(None))
                agg_func.append(None)
            else:
                if isinstance(self.slices[i], AggregateSlice):
                    slices.append(self.slices[i].slice)
                    agg_func.append(self.slices[i].function)
                else:
                    slices.append(self.slices[i])
        transpose = self.y_att.axis > self.x_att.axis
        return slices, agg_func, transpose

    @property
    def wcsaxes_slice(self):
        """
        Returns slicing information usable by WCSAxes.

        This returns an iterable of slices, and including ``'x'`` and ``'y'``
        for the dimensions along which we are not slicing.
        """
        if self.reference_data is None:
            return None
        slices = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis:
                slices.append('x')
            elif i == self.y_att.axis:
                slices.append('y')
            else:
                if isinstance(self.slices[i], AggregateSlice):
                    slices.append(self.slices[i].center)
                else:
                    slices.append(self.slices[i])
        return slices[::-1]

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min, self.x_max = self.x_max, self.x_min

    def flip_y(self):
        """
        Flip the y_min/y_max limits.
        """
        with delay_callback(self, 'y_min', 'y_max'):
            self.y_min, self.y_max = self.y_max, self.y_min


class BaseImageLayerState(MatplotlibLayerState):

    _viewer_callbacks_set = False
    _image_cache = None
    _pixel_cache = None

    def get_sliced_data_shape(self, view=None):

        if (self.viewer_state.reference_data is None or
            self.viewer_state.x_att is None or
                self.viewer_state.y_att is None):
            return None

        x_axis = self.viewer_state.x_att.axis
        y_axis = self.viewer_state.y_att.axis

        shape = self.viewer_state.reference_data.shape
        shape_slice = shape[y_axis], shape[x_axis]

        if view is None:
            return shape_slice
        else:
            return view_shape(shape_slice, view)

    def get_sliced_data(self, view=None, bounds=None):

        full_view, agg_func, transpose = self.viewer_state.numpy_slice_aggregation_transpose

        x_axis = self.viewer_state.x_att.axis
        y_axis = self.viewer_state.y_att.axis

        # For this method, we make use of Data.compute_fixed_resolution_buffer,
        # which requires us to specify bounds in the form (min, max, nsteps).
        # We also allow view to be passed here (which is a normal Numpy view)
        # and, if given, translate it to bounds. If neither are specified,
        # we behave as if view was [slice(None), slice(None)].

        def slice_to_bound(slc, size):
            min, max, step = slc.indices(size)
            n = (max - min - 1) // step
            max = min + step * n
            return (min, max, n + 1)

        if bounds is None:

            # The view should be that which should just be applied to the data
            # slice, not to all the dimensions of the data - thus it should have at
            # most two dimensions

            if view is None:
                view = [slice(None), slice(None)]
            elif len(view) == 1:
                view = view + [slice(None)]
            elif len(view) > 2:
                raise ValueError('view should have at most two elements')

            full_view[x_axis] = view[1]
            full_view[y_axis] = view[0]

        else:

            full_view[x_axis] = bounds[1]
            full_view[y_axis] = bounds[0]

        for i in range(self.viewer_state.reference_data.ndim):
            if isinstance(full_view[i], slice):
                full_view[i] = slice_to_bound(full_view[i], self.viewer_state.reference_data.shape[i])

        # We now get the fixed resolution buffer

        if isinstance(self.layer, BaseData):
            image = self.layer.compute_fixed_resolution_buffer(full_view, target_data=self.viewer_state.reference_data,
                                                               target_cid=self.attribute, broadcast=False, cache_id=self.uuid)
        else:
            image = self.layer.data.compute_fixed_resolution_buffer(full_view, target_data=self.viewer_state.reference_data,
                                                                    subset_state=self.layer.subset_state, broadcast=False, cache_id=self.uuid)

        # We apply aggregation functions if needed

        if agg_func is None:
            if image.ndim != 2:
                raise IncompatibleDataException()
        else:
            if image.ndim != len(agg_func):
                raise ValueError("Sliced image dimensions ({0}) does not match "
                                 "aggregation function list ({1})"
                                 .format(image.ndim, len(agg_func)))
            for axis in range(image.ndim - 1, -1, -1):
                func = agg_func[axis]
                if func is not None:
                    image = func(image, axis=axis)
            if image.ndim != 2:
                raise ValueError("Image after aggregation should have two dimensions")

        # And finally we transpose the data if the order of x/y is different
        # from the native order.

        if transpose:
            image = image.transpose()

        return image


class ImageLayerState(BaseImageLayerState):
    """
    A state class that includes all the attributes for data layers in an image plot.
    """

    attribute = DDSCProperty(docstring='The attribute shown in the layer')
    v_min = DDCProperty(docstring='The lower level shown')
    v_max = DDCProperty(docstring='The upper level shown')
    percentile = DDSCProperty(docstring='The percentile value used to '
                                        'automatically calculate levels')
    contrast = DDCProperty(1, docstring='The contrast of the layer')
    bias = DDCProperty(0.5, docstring='A constant value that is added to the '
                                      'layer before rendering')
    cmap = DDCProperty(docstring='The colormap used to render the layer')
    stretch = DDSCProperty(docstring='The stretch used to render the layer, '
                                     'which should be one of ``linear``, '
                                     '``sqrt``, ``log``, or ``arcsinh``')
    global_sync = DDCProperty(False, docstring='Whether the color and transparency '
                                               'should be synced with the global '
                                               'color and transparency for the data')

    def __init__(self, layer=None, viewer_state=None, **kwargs):

        self.uuid = str(uuid.uuid4())

        super(ImageLayerState, self).__init__(layer=layer, viewer_state=viewer_state)

        self.attribute_lim_helper = StateAttributeLimitsHelper(self, attribute='attribute',
                                                               percentile='percentile',
                                                               lower='v_min', upper='v_max')

        self.attribute_att_helper = ComponentIDComboHelper(self, 'attribute',
                                                           numeric=True, categorical=False)

        percentile_display = {100: 'Min/Max',
                              99.5: '99.5%',
                              99: '99%',
                              95: '95%',
                              90: '90%',
                              'Custom': 'Custom'}

        ImageLayerState.percentile.set_choices(self, [100, 99.5, 99, 95, 90, 'Custom'])
        ImageLayerState.percentile.set_display_func(self, percentile_display.get)

        stretch_display = {'linear': 'Linear',
                           'sqrt': 'Square Root',
                           'arcsinh': 'Arcsinh',
                           'log': 'Logarithmic'}

        ImageLayerState.stretch.set_choices(self, ['linear', 'sqrt', 'arcsinh', 'log'])
        ImageLayerState.stretch.set_display_func(self, stretch_display.get)

        self.add_callback('global_sync', self._update_syncing)
        self.add_callback('layer', self._update_attribute)

        self._update_syncing()

        if layer is not None:
            self._update_attribute()

        self.update_from_dict(kwargs)

        if self.cmap is None:
            self.cmap = self.layer.style.preferred_cmap or colormaps.members[0][1]

    def _update_attribute(self, *args):
        if self.layer is not None:
            self.attribute_att_helper.set_multiple_data([self.layer])
            self.attribute = self.layer.main_components[0]

    def _update_priority(self, name):
        if name == 'layer':
            return 3
        elif name == 'attribute':
            return 2
        elif name == 'global_sync':
            return 1.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def _update_syncing(self, *args):
        if self.global_sync:
            self._sync_color.enable_syncing()
            self._sync_alpha.enable_syncing()
        else:
            self._sync_color.disable_syncing()
            self._sync_alpha.disable_syncing()

    def _get_image(self, view=None):
        return self.layer[self.attribute, view]

    def flip_limits(self):
        """
        Flip the image levels.
        """
        self.attribute_lim_helper.flip_limits()

    def reset_contrast_bias(self):
        with delay_callback(self, 'contrast', 'bias'):
            self.contrast = 1
            self.bias = 0.5


class ImageSubsetLayerState(BaseImageLayerState):
    """
    A state class that includes all the attributes for subset layers in an image plot.
    """

    # TODO: we can save memory by not showing subset multiple times for
    # different image datasets since the footprint should be the same.

    def __init__(self, *args, **kwargs):
        self.uuid = str(uuid.uuid4())
        super(ImageSubsetLayerState, self).__init__(*args, **kwargs)

    def _get_image(self, view=None):
        return self.layer.to_mask(view=view)
