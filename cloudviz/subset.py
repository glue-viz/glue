class Subset(object):
    data = None

    def __init__(self, data):
        self.data = data
        hub = self.data.hub
        hub.broadcast_subset_update(self, new=True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        hub = self.data.hub
        hub.broadcast_subset_update(self, attr=name)


class TreeSubset(Subset):
    pass


class PixelSubset(Subset):
    pass
