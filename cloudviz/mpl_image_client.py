from cloudviz.image_client import ImageClient
import matplotlib.pyplot as plt


class MplImageClient(ImageClient):

    def __init__(self, data, figure=None, axes=None, area_style='filled'):

        if axes is not None and figure is not None and \
                axes.figure is not figure:
            raise Exception("Axes and figure are incompatible")

        ImageClient.__init__(self, data)

        if axes is not None:
            self._ax = axes
            self._figure = axes.figure
        else:
            if figure is None:
                self._figure = plt.figure()
            self._ax = self._figure.add_subplot(1, 1, 1)

        if area_style in ['contour', 'filled']:
            self.area_style = area_style
        else:
            raise Exception("area_style should be one of contour/filled")

    def _redraw(self):
        """
        Re-render the screen
        """
        self._figure.canvas.draw()

    def _remove_subset(self, message):

        s = message.subset
        if s in self._plots:
            for item in self._plots[s].collections:
                item.remove()
            self._plots[s].pop(s)

        super(MplImageClient, self)._remove_subset(self, message)

    def _update_data_plot(self):
        """
        Sync the location of the scatter points to
        reflect what components are being plotted
        """

        if self._image is None:
            return

        if self.data not in self._plots:
            plot = self._ax.imshow(self._image, cmap=plt.cm.gray,
                                   interpolation='nearest', origin='lower')
            self._plots[self.data] = plot
        else:
            self._plots[self.data].set_data(self._image)

    def _update_axis_labels(self):
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')

    def _update_subset_single(self, s):
        """
        Update the location and visual properties
        of each point in a single subset

        Parameters:
        ----------
        s: A subset instance
        The subset to refresh.

        """

        if self._image is None:
            return

        if s not in self.data.subsets:
            raise Exception("Input is not one of data's subsets: %s" % s)

        if s in self._plots:
            for item in self._plots[s].collections:
                item.remove()
            self._plots.pop(s)

        # Handle special case of empty subset
        if s.mask.sum() == 0:
            return

        if self.area_style == 'contour':
            self._plots[s] = self._ax.contour(s.mask.astype(float),
                                              levels=[0.5],
                                              colors=s.style['color'])
        else:
            self._plots[s] = self._ax.contourf(s.mask.astype(float),
                                               levels=[0.5, 0.5], alpha=0.3)
