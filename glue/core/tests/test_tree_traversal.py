#pylint: disable=W0613,W0201,W0212,E1101,E1103
import pytest

from ..tree import NewickTree
from ..tree_traversal import PreOrderTraversal, PostOrderTraversal


def pre_string(tree):
    result = []
    for t in PreOrderTraversal(tree):
        result.append(str(t.id))
    return ''.join(result)


def post_string(tree):
    result = []
    for t in PostOrderTraversal(tree):
        result.append(str(t.id))
    return ''.join(result)


class TestTreeTraversal(object):

    def setup_method(self, method):
        n0 = "0;"
        n1 = "(0,1)2;"
        n2 = "((0,1)4,(2,3)5)6;"
        n3 = "(3,(2,(0,1)4)5)6;"
        n4 = "(1,2,3,4)5;"

        self.t0 = NewickTree(n0)
        self.t1 = NewickTree(n1)
        self.t2 = NewickTree(n2)
        self.t3 = NewickTree(n3)
        self.t4 = NewickTree(n4)

    def test_leaf(self):
        assert pre_string(self.t0) == '0'
        assert post_string(self.t0) == '0'

    def test_simple(self):
        assert pre_string(self.t1) == '201'
        assert post_string(self.t1) == '012'

    def test_two_deep(self):
        assert pre_string(self.t2) == '6401523'
        assert post_string(self.t2) == '0142356'

    def test_unbalanced(self):
        assert pre_string(self.t3) == '6352401'
        assert post_string(self.t3) == '3201456'

    def test_non_binary(self):
        assert pre_string(self.t4) == '51234'
        assert post_string(self.t4) == '12345'

    def test_input_check(self):
        with pytest.raises(TypeError) as exc:
            PreOrderTraversal(5)
        assert exc.value.args[0].startswith("Input is not a tree object")
        with pytest.raises(TypeError) as exc:
            PreOrderTraversal(None)
        assert exc.value.args[0].startswith("Input is not a tree object")
        with pytest.raises(TypeError) as exc:
            PostOrderTraversal(5)
        assert exc.value.args[0].startswith("Input is not a tree object")
