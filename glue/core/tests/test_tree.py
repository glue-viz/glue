#pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103
import pytest

import numpy as np

from ..tree import Tree, NewickTree, DendroMerge


class TestTree(object):

    def test_tree_creation(self):

        root = Tree(id=0, value=100)
        c1 = Tree()
        c2 = Tree()
        c11 = Tree()
        c12 = Tree()
        c21 = Tree()
        c211 = Tree()

        root.add_child(c1)
        root.add_child(c2)
        c1.add_child(c11)
        c1.add_child(c12)
        c2.add_child(c21)
        c21.add_child(c211)

        assert root.id == 0
        assert root.value == 100

        assert c1.parent is root
        assert c2.parent is root
        assert c11.parent is c1
        assert c12.parent is c1
        assert c21.parent is c2
        assert c211.parent is c21

        assert c1 in root.children
        assert c2 in root.children
        assert c11 in c1.children
        assert c12 in c1.children
        assert c21 in c2.children
        assert c211 in c21.children

    def test_newick_tree(self):

        # no labels
        n0 = "0;"
        n1 = "(0,1)2;"
        n2 = "((0,1)4,(2,3)5)6;"
        n3 = "(3,(2,(0,1)4)5)6;"
        n4 = "(1,2,3,4)5;"

        tree0 = NewickTree(n0)
        tree1 = NewickTree(n1)
        tree2 = NewickTree(n2)
        tree3 = NewickTree(n3)
        tree4 = NewickTree(n4)

        assert tree0.id == 0
        assert tree1.id == 2
        assert not tree1.value
        assert 0 in [x.id for x in tree1.children]
        assert 1 in [x.id for x in tree1.children]

        assert tree2.id == 6
        assert not tree1.value
        assert 4 in [x.id for x in tree2.children]
        assert 5 in [x.id for x in tree2.children]
        assert not 0 in [x.id for x in tree2.children]
        assert not 1 in [x.id for x in tree2.children]

        assert 1 in [x.id for x in tree4.children]
        assert 2 in [x.id for x in tree4.children]
        assert 3 in [x.id for x in tree4.children]
        assert 4 in [x.id for x in tree4.children]

        assert n1 == tree1.to_newick()
        assert n2 == tree2.to_newick()
        assert n3 == tree3.to_newick()
        assert n4 == tree4.to_newick()

        # with labels
        n1 = "(0:0,1:10)2:20;"
        n2 = "((0:0,1:10)4:4,(2:20,3:30)5:50)6:60;"
        n3 = "(3:30,(2:20,(0:0,1:10)4:40)5:50)6:60;"

        tree1 = NewickTree(n1)
        tree2 = NewickTree(n2)
        tree3 = NewickTree(n3)

        assert tree1.value == '20'

        assert n1 == tree1.to_newick()
        assert n2 == tree2.to_newick()
        assert n3 == tree3.to_newick()

    def test_dendro_merge(self):
        m1 = np.array([[0, 1], [2, 3], [4, 5]])
        n1 = "((0,1)4,(2,3)5)6;"
        m2 = np.array([[0, 1], [4, 2], [5, 3]])
        n2 = "(3,(2,(0,1)4)5)6;"
        t1 = DendroMerge(m1)
        t2 = DendroMerge(m2)

        #invalid merge lists
        m3 = np.array([[0, 1], [5, 2], [5, 3]])
        m4 = np.array([[-1, 1], [4, 2], [5, 3]])
        m5 = np.array([[0, 1], [1, 2], [5, 3]])

        assert t1.to_newick() == n1
        assert t2.to_newick() == n2
        assert t1.id == 6
        assert t2.id == 6

        with pytest.raises(TypeError) as exc:
            DendroMerge(m3)
        assert exc.value.args[0] == "Invalid mergelist: contains duplicates"
        with pytest.raises(TypeError) as exc:
            DendroMerge(m4)
        assert exc.value.args[0] == ("Invalid mergelist: does not "
                                     "run from 0-nleaf")

        with pytest.raises(TypeError) as exc:
            DendroMerge(m5)
        assert exc.value.args[0] == "Invalid mergelist: contains duplicates"
