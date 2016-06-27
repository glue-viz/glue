"""
The AxesCache class alters how an Axes instance is rendered.
While enabled, the AxesCache quickly re-renders an original view,
properly scaled and translated to reflect changes in the viewport.
The downside is that the re-rendered image is fuzzy and/or truncated.

The best way to use an AxesCache is to enable it during
window resize drags and pan/zoom mouse drags; these generate
rapid draw requests, and users might prefer high refresh
rates to pixel-perfect renders.

Unfortunately, Matplotlib on it's own doesn't provide an easy
mechanism to attach event handlers to either window resize drags
or pan/zoom drags. This code must be added separately.
"""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh


class RenderCapture(object):

    """
    A RemderCapture saves an image of a fully-rendered
    Axes instance, and provides a method for re-rendering
    a properly transformed image during panning and zooming
    """

    def __init__(self, axes, renderer):
        self.axes = axes
        self._corners = self._get_corners(axes)
        px, py, dx, dy = self._corners

        im = self.extract_image(renderer)
        im = im[py[0]: py[-1] + 1, px[0]: px[-1] + 1, :]
        self.im = im
        self._mesh = None
        self._image = None
        self.image

    @property
    def image(self):
        if self._image is not None:
            return self._image

        px, py, dx, dy = self._corners
        self._image = AxesImage(self.axes,
                                origin='lower',
                                interpolation='nearest')
        self._image.set_data(self.im)
        self._image.set_extent((dx[0], dx[-1], dy[0], dy[-1]))
        self.axes._set_artist_props(self._image)

        return self._image

    @property
    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        px, py, dx, dy = self._corners
        x, y, c = self.axes._pcolorargs('pcolormesh', dx, dy,
                                        self.im[:, :, 0],
                                        allmatch=False)
        ny, nx = x.shape
        coords = np.column_stack((x.ravel(), y.ravel()))
        collection = QuadMesh(nx - 1, ny - 1, coords,
                              shading='flat', antialiased=False,
                              edgecolors='None',
                              cmap='gray')
        collection.set_array(c.ravel())
        collection.set_clip_path(self.axes.patch)
        collection.set_transform(self.axes.transData)
        self._mesh = collection
        return self._mesh

    def draw(self, renderer, *args, **kwargs):
        if self.axes.get_xscale() == 'linear' and \
                self.axes.get_yscale() == 'linear':
            self.image.draw(renderer, *args, **kwargs)
        else:
            self.mesh.draw(renderer, *args, **kwargs)

    @staticmethod
    def _get_corners(axes):
        """
        Return the device and data coordinates
        for a box slightly inset from the edge
        of an axes instance

        Returns 4 1D arrays:
        px : Pixel X locations for each column of the box
        py : Pixel Y locations for each row of the box
        dx : Data X locations for each column of the box
        dy : Data Y locations for each row of the box
        """
        xlim = axes.get_xlim()
        ylim = axes.get_ylim()
        pts = np.array([[xlim[0], ylim[0]],
                        [xlim[1], ylim[1]]])

        corners = axes.transData.transform(pts).astype(np.int)

        # move in 5 pixels, to avoid grabbing the tick marks
        px = np.arange(corners[0, 0] + 5, corners[1, 0] - 5)
        py = np.arange(corners[0, 1] + 5, corners[1, 1] - 5)

        tr = axes.transData.inverted().transform
        dx = tr(np.column_stack((px, px)))[:, 0]
        dy = tr(np.column_stack((py, py)))[:, 1]
        return px, py, dx, dy

    @staticmethod
    def extract_image(renderer):
        try:
            buf = renderer.buffer_rgba()
        except TypeError:  # mpl v1.1 has different signature
            buf = renderer.buffer_rgba(0, 0)

        result = np.frombuffer(buf, dtype=np.uint8)
        result = result.reshape((int(renderer.height),
                                 int(renderer.width), 4)).copy()
        return np.flipud(result)


class AxesCache(object):

    def __init__(self, axes):
        self.axes = axes

        self._capture = None
        self.axes.draw = self.draw
        self._enabled = False

    def draw(self, renderer, *args, **kwargs):
        if self._capture is None or not self._enabled:
            Axes.draw(self.axes, renderer, *args, **kwargs)
            if hasattr(renderer, 'buffer_rgba'):
                self._capture = RenderCapture(self.axes, renderer)
        else:
            self.axes.axesPatch.draw(renderer, *args, **kwargs)
            self._capture.draw(renderer, *args, **kwargs)
            self.axes.xaxis.draw(renderer, *args, **kwargs)
            self.axes.yaxis.draw(renderer, *args, **kwargs)
            for s in self.axes.spines.values():
                s.draw(renderer, *args, **kwargs)

    def clear_cache(self):
        """
        Clear the cache, forcing the a full re-render
        """
        self._capture = None

    def disable(self):
        """
        Temporarily disable cache re-renders. Render
        results are still saved, for when
        enable() is next called
        """
        self._enabled = False
        self.axes.figure.canvas.draw()

    def enable(self):
        """
        Enable cached-rerenders
        """
        self._enabled = True

    def teardown(self):
        """
        Permanently disable this cache, and restore
        normal Axes render behavior
        """
        self.axes.draw = Axes.draw.__get__(self.axes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num = 1000000
    plt.subplot(111)
    plt.subplots_adjust(bottom=.5, top=.8)
    plt.scatter(np.random.randn(num), np.random.randn(num),
                s=np.random.randint(10, 50, num),
                c=np.random.randint(0, 255, num),
                alpha=.2, linewidths=0)
    plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
    cache = AxesCache(plt.gca())
    cache.enable()
    plt.grid('on')
    # plt.xscale('log')

    plt.show()
