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
    data1.add_virtual_component(component_2,
                                lambda: data1[component_1])
    data2.add_virtual_component(component_1,
                                lambda: data2[component_2])
