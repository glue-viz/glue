import cloudviz as cv
from cloudviz.mpl_scatter_client import MplScatterClient


""" An example of using the scatter client with a lasso-based ROI
object.  For simplicity, all mouse events are sent to global functions
which update the ROI

In a real application, the client would manage the ROI object manually
"""

# set up the data
d = cv.TabularData()
d.read_data('oph_c2d_yso_catalog.tbl')

# create the hub
h = cv.Hub()

# create the 2 clients
c1 = MplScatterClient(d)
c2 = MplScatterClient(d)
c1.set_xdata('ra')
c1.set_ydata('dec')
c2.set_xdata('IR1_flux_1')
c2.set_ydata('IR2_flux_1')

# create a subset
s = cv.subset.ElementSubset(d)
mask = d.components['ra'].data.ravel() > 248
s.mask = mask

# register clients and data to hub
# (to receive and send events, respectively)
# register subset to send events
c1.register_to_hub(h)
c2.register_to_hub(h)
d.register_to_hub(h)
s.register()

selection_type = 'box'

if selection_type == 'box':
    t1 = cv.MplBoxTool(d, 'ra', 'dec', c1._ax)
    t2 = cv.MplBoxTool(d, 'IR1_flux_1', 'IR2_flux_1', c2._ax)
elif selection_type == 'circle':
    t1 = cv.MplCircleTool(d, 'ra', 'dec', c1._ax)
    t2 = cv.MplCircleTool(d, 'IR1_flux_1', 'IR2_flux_1', c2._ax)
elif selection_type == 'polygon':
    t1 = cv.MplPolygonTool(d, 'ra', 'dec', c1._ax)
    t2 = cv.MplPolygonTool(d, 'IR1_flux_1', 'IR2_flux_1', c2._ax)
elif selection_type == 'lasso':
    t1 = cv.MplLassoTool(d, 'ra', 'dec', c1._ax)
    t2 = cv.MplLassoTool(d, 'IR1_flux_1', 'IR2_flux_1', c2._ax)
else:
    raise Exception("Unknown selection type: %s" % selection_type)
