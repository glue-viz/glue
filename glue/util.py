import glue
from glue.data import DerivedComponent
from glue.component_link import ComponentLink

def relim(lo, hi, log=False):
    x, y = lo, hi
    if log:
        if lo < 0:
            x = 1e-5
        if hi < 0:
            y = 1e5
    return (x, y)


def glue_components_1to1(data1, component_1,
                         data2, component_2):

    link1 = ComponentLink([component_1], component_2)
    link2 = ComponentLink([component_2], component_1)

    dc1 = DerivedComponent(data1, link1)
    dc2 = DerivedComponent(data2, link2)
    data1.add_component(dc1, component_2)
    data2.add_component(dc2, component_1)
