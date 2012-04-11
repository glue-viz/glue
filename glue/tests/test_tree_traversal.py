import unittest
from glue.tree import NewickTree
from glue.tree_traversal import PreOrderTraversal, PostOrderTraversal

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


class TestTreeTraversal(unittest.TestCase):

    def setUp(self):
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
        self.assertEquals(pre_string(self.t0), '0')
        self.assertEquals(post_string(self.t0), '0')

    def test_simple(self):
        self.assertEquals(pre_string(self.t1), '201')
        self.assertEquals(post_string(self.t1), '012')

    def test_two_deep(self):
        self.assertEquals(pre_string(self.t2), '6401523')
        self.assertEquals(post_string(self.t2), '0142356')

    def test_unbalanced(self):
        self.assertEquals(pre_string(self.t3), '6352401')
        self.assertEquals(post_string(self.t3), '3201456')

    def test_non_binary(self):
        self.assertEquals(pre_string(self.t4), '51234')
        self.assertEquals(post_string(self.t4), '12345')

    def test_input_check(self):
        self.assertRaises(TypeError, PreOrderTraversal, 5)
        self.assertRaises(TypeError, PreOrderTraversal, None)
        self.assertRaises(TypeError, PostOrderTraversal, 5)
