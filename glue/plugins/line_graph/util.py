from __future__ import absolute_import, print_function, division

from colorsys import hls_to_rgb
import numpy as np


def get_colors(num_colors):
    """
    Taken from: http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    Creates a list of distinct colors to plot with
    :param num_colors: number of colors to generate
    :return: list of colors
    """
    colors = []
    if num_colors:
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i / 360.
            lightness = (50 + np.random.rand() * 10) / 100.
            saturation = (90 + np.random.rand() * 10) / 100.
            colors.append(hls_to_rgb(hue, lightness, saturation))
    return colors