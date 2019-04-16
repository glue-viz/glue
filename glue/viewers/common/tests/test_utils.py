from glue.viewers.common.viewer import Viewer
from glue.viewers.common.utils import get_viewer_tools


class ViewerWithTools(Viewer):
    inherit_tools = True
    tools = ['save']
    subtools = {'save': []}


def test_get_viewer_tools():

    class CustomViewer1(ViewerWithTools):
        pass

    tools, subtools = get_viewer_tools(CustomViewer1)

    assert tools == ['save']
    assert subtools == {'save': []}

    class CustomViewer2(ViewerWithTools):
        tools = ['banana']
        subtools = {'save': ['apple', 'pear']}

    tools, subtools = get_viewer_tools(CustomViewer2)

    assert tools == ['save', 'banana']
    assert subtools == {'save': ['apple', 'pear']}

    CustomViewer2.inherit_tools = False

    tools, subtools = get_viewer_tools(CustomViewer2)

    assert tools == ['banana']
    assert subtools == {'save': ['apple', 'pear']}

    class Mixin(object):
        pass

    class CustomViewer3(CustomViewer2, Mixin):
        tools = ['orange']
        subtools = {'banana': ['one', 'two']}
        inherit_tools = True

    tools, subtools = get_viewer_tools(CustomViewer3)

    assert tools == ['banana', 'orange']
    assert subtools == {'save': ['apple', 'pear'], 'banana': ['one', 'two']}
