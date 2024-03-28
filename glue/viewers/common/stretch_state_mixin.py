from glue.config import stretches
from glue.viewers.matplotlib.state import (
    DeferredDrawDictCallbackProperty as DDDCProperty,
    DeferredDrawSelectionCallbackProperty as DDSCProperty,
)

__all__ = ["StretchStateMixin"]


class StretchStateMixin:
    stretch = DDSCProperty(
        docstring="The stretch used to render the layer, "
        "which should be one of ``linear``, "
        "``sqrt``, ``log``, or ``arcsinh``"
    )
    stretch_parameters = DDDCProperty(
        docstring="Keyword arguments to pass to the stretch"
    )

    _stretch_set_up = False

    def setup_stretch_callback(self):
        type(self).stretch.set_choices(self, list(stretches.members))
        type(self).stretch.set_display_func(self, stretches.display_func)
        self._reset_stretch()
        self.add_callback("stretch", self._reset_stretch)
        self.add_callback("stretch_parameters", self._sync_stretch_parameters)
        self._stretch_set_up = True

    @property
    def stretch_object(self):
        if not self._stretch_set_up:
            raise Exception("setup_stretch_callback has not been called")
        return self._stretch_object

    def _sync_stretch_parameters(self, *args):
        for key, value in self.stretch_parameters.items():
            if hasattr(self._stretch_object, key):
                setattr(self._stretch_object, key, value)
            else:
                raise ValueError(
                    f"Stretch object {self._stretch_object.__class__.__name__} has no attribute {key}"
                )

    def _reset_stretch(self, *args):
        self._stretch_object = stretches.members[self.stretch]()
        self.stretch_parameters.clear()
