import numpy as np

from glue.core.data_derived import DerivedData
from glue.core.message import NumericalDataChangedMessage

__all__ = ['PathSlicedData', 'sample_points']


def sample_points(x, y, spacing=1):
    """
    Resample a piecewise-linear path so consecutive samples are equally
    spaced along its arc length.

    The path is described by ``(x, y)`` vertices in arbitrary units. The
    function returns ``n + 1`` samples where ``n = floor(L / spacing)`` and
    ``L`` is the total path length. The first sample sits on the first
    vertex; the last sample is the last sample inside the path that is
    a multiple of ``spacing`` from the start, so the very last vertex of
    the input is typically not included.

    Parameters
    ----------
    x, y : sequences of float
        Path vertices. Must have the same length and at least two
        elements each.
    spacing : float, optional
        Spacing in input units between consecutive output samples.
        Defaults to ``1`` (one pixel of the parent dataset for the
        common case where ``x``/``y`` are pixel coordinates).

    Returns
    -------
    x_sampled, y_sampled : ndarray
        Resampled path with consecutive points ``spacing`` apart in
        the ``(x, y)`` plane.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different shapes, are not 1-d, have
        fewer than two vertices, if ``spacing`` is non-positive, or if
        the input path is shorter than ``spacing``.
    """
    # Adapted from pvextractor.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape "
                         f"(got {x.shape} and {y.shape})")
    if x.ndim != 1:
        raise ValueError("x and y must be 1-d "
                         f"(got x.ndim={x.ndim})")
    if x.size < 2:
        raise ValueError("path must have at least two vertices")
    if spacing <= 0:
        raise ValueError(f"spacing must be positive (got {spacing})")

    # Distance interval between consecutive vertices, then cumulative
    # arc length from the start.
    dx = np.diff(x)
    dy = np.diff(y)
    dd = np.hypot(dx, dy)
    d = np.hstack([0., np.cumsum(dd)])

    # Stop short of the last vertex so all output samples are exactly
    # `spacing` apart.
    n_points = int(np.floor(d[-1] / spacing))

    if n_points == 0:
        raise ValueError("Path is shorter than spacing")

    d_sampled = np.linspace(0., n_points * spacing, n_points + 1)

    x_sampled = np.interp(d_sampled, d, x)
    y_sampled = np.interp(d_sampled, d, y)

    return x_sampled, y_sampled


class PathSlicedData(DerivedData):
    """
    A dataset where two dimensions of a parent dataset have been replaced
    with one dimension that indexes along a piecewise-linear path.

    The path is supplied as ``(x, y)`` pixel coordinates in the parent's
    sliced axes. Internally :func:`sample_points` resamples the path so
    consecutive points are evenly spaced; the resampled length becomes
    the new last axis of ``self.shape``::

        shape == (...non-sliced axes of parent..., n_path)

    Parameters
    ----------
    original_data : :class:`~glue.core.Data`
        The data being sliced.
    cid_x, cid_y : pixel ComponentID of ``original_data``
        Identify which axes of the original dataset are collapsed into
        the path. ``cid_x`` and ``cid_y`` must refer to two distinct
        axes.
    x, y : sequences of float
        Vertices of the path in pixel coordinates of ``original_data``.
        ``x`` is along ``cid_x.axis``; ``y`` is along ``cid_y.axis``.
        Lengths must match and be at least 2.
    label : str, optional
        Human-readable label for the dataset.
    spacing : float, optional
        Spacing in pixels of the parent dataset between consecutive
        samples along the path. Defaults to ``1``.

    """

    def __init__(self, original_data, cid_x, x, cid_y, y, label='', spacing=1):
        super(DerivedData, self).__init__()
        if cid_x.axis == cid_y.axis:
            raise ValueError("cid_x and cid_y must refer to different axes "
                             f"of the parent (both are axis {cid_x.axis})")
        self.original_data = original_data
        self.cid_x = cid_x
        self.cid_y = cid_y
        self.sliced_dims = (cid_x.axis, cid_y.axis)
        self._label = label
        # set_xy validates x/y and the spacing; assigning self.x/self.y
        # before the validation would leave the object in a broken state.
        self.set_xy(x, y, spacing=spacing)

    def set_xy(self, x, y, spacing=1):
        """
        Replace the path with a new ``(x, y)`` and notify listeners.

        The path is resampled by :func:`sample_points`. Any cached
        fixed-resolution-buffer entries that reference this dataset
        (as either ``data`` or ``target_data``) are dropped so that
        the next draw recomputes against the new path.

        Parameters
        ----------
        x, y : sequences of float
            New path vertices (see :func:`sample_points`).
        spacing : float, optional
            Sample spacing in parent-pixel units. Defaults to ``1``.
        """
        from glue.core.fixed_resolution_buffer import invalidate_cache
        x, y = sample_points(x, y, spacing=spacing)
        self.x = x
        self.y = y
        # Cached FRB results are keyed on this dataset's identity, but the
        # underlying path has just changed -- evict so the next draw
        # recomputes rather than returning stale data.
        invalidate_cache(self)
        hub = getattr(self.original_data, 'hub', None)
        if hub is not None:
            hub.broadcast(NumericalDataChangedMessage(self))

    @property
    def label(self):
        return self._label

    def _without_sliced(self, iterable):
        return [x for ix, x in enumerate(iterable) if ix not in self.sliced_dims]

    @property
    def shape(self):
        return tuple(self._without_sliced(self.original_data.shape) + [len(self.x)])

    @property
    def main_components(self):
        return self.original_data.main_components

    def get_kind(self, cid):
        return self.original_data.get_kind(cid)

    def _get_pix_coords(self, view=None):
        """
        Build pixel coordinates in the parent dataset that correspond to
        each PV cell.

        Returns ``(pix_coords, keep, shape)``:

        * ``pix_coords`` -- list of flat int arrays, one per parent axis,
          giving the parent pixel index for every in-bounds PV cell.
          Suitable for use as ``view=`` on a glue ``Data``'s
          ``get_data``/``get_mask``/``compute_statistic`` (which then does
          combined fancy indexing).
        * ``keep`` -- boolean mask in the *PV* coordinate frame indicating
          which cells projected into the parent's bounds.
        * ``shape`` -- shape of ``keep`` (i.e. the PV frame after applying
          ``view``).

        Note in particular that the two PV-sliced parent axes
        (``cid_x.axis`` and ``cid_y.axis``) are *coupled* to the same path
        index. A naive ``np.meshgrid`` over both axes independently would
        give the cross product instead, producing nonsensical output.
        """
        if view is None or view is Ellipsis:
            view = (slice(None),) * self.ndim

        # Detect advanced indexing -- the FRB path supplies a tuple of
        # 1-d ndarrays of equal length and expects a 1-d result.
        advanced_indexing = (
            len(view) > 0 and isinstance(view[0], np.ndarray)
        )

        if advanced_indexing:
            path_view = view[-1]
            shape = path_view.shape
            pix_coords = []
            pv_idim = 0
            for idim in range(self.original_data.ndim):
                if idim == self.cid_x.axis:
                    pix = self.x[path_view]
                elif idim == self.cid_y.axis:
                    pix = self.y[path_view]
                else:
                    pix = np.arange(self.original_data.shape[idim])[view[pv_idim]]
                    pv_idim += 1
                pix_coords.append(pix)
        else:
            # View is a tuple of slices, applied to the PV frame.
            # Build per-PV-axis index arrays after slicing, then meshgrid
            # them. The path axis is the LAST PV axis.
            pv_axis_arrays = []
            pv_idim = 0
            for idim in range(self.original_data.ndim):
                if idim in self.sliced_dims:
                    continue
                pv_axis_arrays.append(
                    np.arange(self.original_data.shape[idim])[view[pv_idim]])
                pv_idim += 1
            pv_axis_arrays.append(np.arange(len(self.x))[view[-1]])

            pv_meshes = np.meshgrid(*pv_axis_arrays, indexing='ij', copy=False)
            shape = pv_meshes[0].shape

            # Map each parent axis to its meshgrid array. Both sliced
            # parent axes index into the same path mesh (the last one).
            pix_coords = []
            kept_idx = 0
            for idim in range(self.original_data.ndim):
                if idim == self.cid_x.axis:
                    pix_coords.append(self.x[pv_meshes[-1]])
                elif idim == self.cid_y.axis:
                    pix_coords.append(self.y[pv_meshes[-1]])
                else:
                    pix_coords.append(pv_meshes[kept_idx])
                    kept_idx += 1

        # Compute the keep mask: cells whose parent index is in bounds
        # along every parent axis.
        keep = np.ones(shape, dtype=bool)
        for idim in range(self.original_data.ndim):
            keep &= (pix_coords[idim] >= 0) & (pix_coords[idim] < self.original_data.shape[idim])

        pix_coords = [c[keep].astype(int) for c in pix_coords]
        return pix_coords, keep, shape

    def get_data(self, cid, view=None):

        if cid in self.pixel_component_ids:
            return super().get_data(cid, view)

        pix_coords, keep, shape = self._get_pix_coords(view=view)
        result = np.zeros(shape)
        # Pass `view` to the parent as a *tuple* so numpy does combined
        # fancy indexing (1-d result). A list would be treated as a
        # single index along axis 0, producing a multi-dim result.
        result[keep] = self.original_data.get_data(cid, view=tuple(pix_coords))

        return result

    def get_mask(self, subset_state, view=None):

        pix_coords, keep, shape = self._get_pix_coords(view=view)
        result = np.zeros(shape, dtype=bool)
        result[keep] = self.original_data.get_mask(
            subset_state, view=tuple(pix_coords))

        return result

    def compute_statistic(self, statistic, cid, subset_state=None, axis=None,
                          finite=True, positive=False, percentile=None,
                          view=None, random_subset=None):
        """
        Reduce a statistic over the PV-frame data.

        Unlike the parent dataset, we *can't* delegate to
        ``original_data.compute_statistic`` with a fancy-indexed view
        because the view collapses the PV shape to 1-d, which makes
        ``axis`` meaningless. Instead we materialise the PV view of the
        data and reduce locally using the same helper the regular
        :class:`~glue.core.Data.compute_statistic` uses.
        """
        from glue.utils import compute_statistic
        values = self.get_data(cid, view=view)
        mask = None
        if subset_state is not None:
            mask = self.get_mask(subset_state, view=view)
        return compute_statistic(statistic, values, mask=mask, axis=axis,
                                 finite=finite, positive=positive,
                                 percentile=percentile)

    def compute_histogram(self, *args, **kwargs):
        return self.original_data.compute_histogram(*args, **kwargs)

    def compute_fixed_resolution_buffer(self, bounds, target_data=None, target_cid=None,
                                        subset_state=None, broadcast=True, cache_id=None):
        """
        Compute a fixed-resolution buffer.

        Coordinate translation between PathSlicedData instances goes
        through glue's :class:`~glue.core.component_link.ComponentLink`
        graph; the PV slicer tool is expected to register the
        appropriate links (typically via
        :func:`glue.plugins.tools.pv_slicer.path_sliced_data_links.link_path_sliced_group`)
        before any FRB is requested.
        """
        from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer
        return compute_fixed_resolution_buffer(
            self, bounds,
            target_data=target_data,
            target_cid=target_cid,
            subset_state=subset_state,
            broadcast=broadcast,
            cache_id=cache_id)
