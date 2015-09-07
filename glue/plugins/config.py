from collections import namedtuple

from ..core.config import Registry

__all__ = ['ProfileFitterRegistry', 'fit_plugin']

class ProfileFitterRegistry(Registry):
    item = namedtuple('ProfileFitter', 'cls')

    def add(self, cls):
        """
        Add colormap *cmap* with label *label*.
        """
        self.members.append(cls)

    def default_members(self):
        from ..core.fitters import __FITTERS__
        return list(__FITTERS__)

fit_plugin = ProfileFitterRegistry()
