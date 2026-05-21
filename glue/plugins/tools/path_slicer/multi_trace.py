"""
Backend-neutral mixin that gives a path slicer tool its multi-trace
data model and the public ``menu_entries`` / ``set_target`` API.

A "trace" is one Enter on the path tool: it produces one
:class:`PathSlicedData` per Data layer of the source viewer, and is
displayed in its own slice viewer. The mixin doesn't know how the
path is drawn (matplotlib :class:`PathMode` for the Qt and Jupyter
matplotlib tools, click-to-add-vertex for the bqplot tool); it only
tracks the traces and orchestrates create-vs-update via the helper
functions in :mod:`.common`.

Subclasses are expected to:

* Call :meth:`_init_multi_trace` from their ``__init__`` after the
  rest of the class has initialised.
* Set :attr:`slice_viewer_cls` -- the viewer class opened for new
  traces.
* Provide ``self.viewer`` (the source image viewer).
* Override :meth:`_refresh_overlays` to draw their backend's overlay
  artists (matplotlib :class:`Line2D`, bqplot :class:`Lines`, ...).
* Optionally override :meth:`hover_preview` for a non-committing
  highlight on UI hover.
* Optionally override :meth:`_on_traces_changed` to refresh a
  dropdown UI after each trace.
"""
from .common import create_trace, open_slice_viewer_for, update_trace


__all__ = ['MultiTracePathSlicerMixin']


class MultiTracePathSlicerMixin:
    """See module docstring."""

    slice_viewer_cls = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _init_multi_trace(self):
        # list[list[PathSlicedData]] -- one trace per Enter.
        self._traces = []
        # Parallel to _traces: the slice viewer hosting each trace.
        self._slice_viewers = []
        # The most recent slice viewer; shadowed here so callers that
        # only know the singular attribute (e.g. crosshair-tool
        # introspection) still find one.
        self._slice_viewer = None
        # ``None`` means "create new on next Enter"; otherwise must be
        # one of ``self._traces``.
        self._target_trace = None

    # ------------------------------------------------------------------
    # Trace orchestration
    # ------------------------------------------------------------------

    def _open_or_update(self, vx, vy):
        if self._target_trace is None:
            new_paths = create_trace(self.viewer, vx, vy, self._traces)
            self._traces.append(new_paths)
            slice_viewer = open_slice_viewer_for(
                self.viewer, self.slice_viewer_cls, new_paths)
            self._slice_viewers.append(slice_viewer)
            self._slice_viewer = slice_viewer
            # The just-created trace becomes the target for the next
            # Enter, so consecutive Enters tweak the same path until
            # the user picks something else from the dropdown.
            self._target_trace = self._traces[-1]
        else:
            update_trace(self._target_trace, vx, vy)
        self._refresh_overlays()
        self._on_traces_changed()

    # ------------------------------------------------------------------
    # Public API used by UI dropdowns
    # ------------------------------------------------------------------

    def menu_entries(self):
        """The ``(label, target)`` pairs a dropdown UI should show.
        ``target`` is ``None`` for "Create new path" or a trace from
        :attr:`_traces` for "Update path N"."""
        entries = [('Create new path', None)]
        for i, _trace in enumerate(self._traces, start=1):
            entries.append((f'Update path {i}', _trace))
        return entries

    def set_target(self, target):
        """Set the next-Enter target. ``None`` creates a new trace;
        otherwise must be one of :attr:`_traces`."""
        self._target_trace = target
        self._refresh_overlays()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _close_slice_viewers(self):
        for slice_viewer in self._slice_viewers:
            slice_viewer.close()
        self._slice_viewers.clear()
        self._slice_viewer = None

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _refresh_overlays(self):
        """Draw / update overlay artists on the source viewer to show
        all traces with the active one highlighted. Override in the
        backend subclass."""

    def hover_preview(self, target):
        """Temporarily highlight ``target`` (or none if ``target`` is
        ``None``) without committing the selection. UI dropdowns call
        this on hover; :meth:`_refresh_overlays` restores the
        committed state when the menu closes. Default no-op."""

    def _on_traces_changed(self):
        """Override to refresh the backend's dropdown UI after a trace
        is added or updated. Default no-op."""
