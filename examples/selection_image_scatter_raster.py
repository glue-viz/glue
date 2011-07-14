"""
This is a self contained (non-interactive) example of setting up
a cloudviz environment with 2 catalog clients linked to the same data.
"""
import cloudviz as cv

import matplotlib.pyplot as plt
from cloudviz import RasterAxes

#set up the data
d = cv.GriddedData()
d.read_data('dendro_oph.fits', use_hdu=[0, 1])

# create the hub
h = cv.Hub()

# create the 2 clients
fig = plt.figure()
ax = RasterAxes(fig, [0.1,0.1,0.8,0.8])
fig.add_axes(ax)
c1 = cv.MplScatterClient(d, axes=ax)
c2 = cv.MplImageClient(d)

# register the clients and data to the hub
# (to receive and send events, respectively)
c1.register_to_hub(h)
c2.register_to_hub(h)
d.register_to_hub(h)

# #define the axes to plot
c1.set_xdata('PRIMARY')
c1.set_ydata('INDEX_MAP')
c2.set_component('PRIMARY')

# create a new subset. Note we need to register() each subset
s = cv.subset.ElementSubset(d)
s.mask = d.components['INDEX_MAP'].data > 0.5
s.register()

#change plot properties. Updated autmatically
s.style['color'] = 'm'
s.style['alpha'] = .8

selection_type = 'lasso'

if selection_type == 'box':
    t1 = cv.MplBoxTool(s, 'PRIMARY', 'INDEX_MAP', c1._ax)
    t2 = cv.MplBoxTool(s, 'XPIX', 'YPIX', c2._ax)
elif selection_type == 'circle':
    t1 = cv.MplCircleTool(s, 'PRIMARY', 'INDEX_MAP', c1._ax)
    t2 = cv.MplCircleTool(s, 'XPIX', 'YPIX', c2._ax)
elif selection_type == 'polygon':
    t1 = cv.MplPolygonTool(s, 'PRIMARY', 'INDEX_MAP', c1._ax)
    t2 = cv.MplPolygonTool(s, 'XPIX', 'YPIX', c2._ax)
elif selection_type == 'lasso':
    t1 = cv.MplLassoTool(s, 'PRIMARY', 'INDEX_MAP', c1._ax)
    t2 = cv.MplLassoTool(s, 'XPIX', 'YPIX', c2._ax)
else:
    raise Exception("Unknown selection type: %s" % selection_type)
