import unittest
import numpy as np
from cloudviz.tree import Tree, NewickTree, DendroMerge

class TestTree(unittest.TestCase):

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

        self.assertEqual(root.id, 0)
        self.assertEqual(root.value, 100)

        self.assertIs(c1.parent, root)
        self.assertIs(c2.parent, root)
        self.assertIs(c11.parent, c1)
        self.assertIs(c12.parent, c1)
        self.assertIs(c21.parent, c2)
        self.assertIs(c211.parent, c21)

        self.assertIn(c1, root.children)
        self.assertIn(c2, root.children)
        self.assertIn(c11, c1.children)
        self.assertIn(c12, c1.children)
        self.assertIn(c21, c2.children)
        self.assertIn(c211, c21.children)

    def test_newick_tree(self):
        
        # no labels
        n1 = "(0,1)2;"
        n2 = "((0,1)4,(2,3)5)6;"
        n3 = "(3,(2,(0,1)4)5)6;"
        n4 = "(1,2,3,4)5;"
        
        tree1 = NewickTree(n1)
        tree2 = NewickTree(n2)
        tree3 = NewickTree(n3)
        tree4 = NewickTree(n4)
        
        self.assertEquals(tree1.id, 2)
        self.assertFalse(tree1.value)
        self.assertIn(0, [x.id for x in tree1.children])
        self.assertIn(1, [x.id for x in tree1.children])

        self.assertEquals(tree2.id, 6)
        self.assertFalse(tree1.value)
        self.assertIn(4, [x.id for x in tree2.children])
        self.assertIn(5, [x.id for x in tree2.children])
        self.assertNotIn(0, [x.id for x in tree2.children])
        self.assertNotIn(1, [x.id for x in tree2.children])
        

        self.assertIn(1, [x.id for x in tree4.children])
        self.assertIn(2, [x.id for x in tree4.children])
        self.assertIn(3, [x.id for x in tree4.children])
        self.assertIn(4, [x.id for x in tree4.children])

        self.assertEqual(n1, tree1.to_newick())
        self.assertEqual(n2, tree2.to_newick())
        self.assertEqual(n3, tree3.to_newick())
        self.assertEqual(n4, tree4.to_newick())

        # with labels
        n1 = "(0:0,1:10)2:20;"
        n2 = "((0:0,1:10)4:4,(2:20,3:30)5:50)6:60;"
        n3 = "(3:30,(2:20,(0:0,1:10)4:40)5:50)6:60;"
        
        tree1 = NewickTree(n1)
        tree2 = NewickTree(n2)
        tree3 = NewickTree(n3)
        
        self.assertEquals(tree1.value, '20')
        
        self.assertEqual(n1, tree1.to_newick())
        self.assertEqual(n2, tree2.to_newick())
        self.assertEqual(n3, tree3.to_newick())

    def test_dendro_merge(self):
        m1 = np.array( [ [0,1], [2,3], [4,5] ] )
        n1 = "((0,1)4,(2,3)5)6;"
        m2 = np.array( [ [0,1], [4,2], [5,3] ])
        n2 = "(3,(2,(0,1)4)5)6;"
        t1 = DendroMerge(m1)
        t2 = DendroMerge(m2)

        #invalid merge lists
        m3 = np.array( [ [0,1], [5,2], [5,3] ] )
        m4 = np.array( [ [-1,1], [4,2], [5,3] ] )
        m5 = np.array( [ [0,1], [1,2], [5,3] ] )


        self.assertEqual(t1.to_newick(), n1)
        self.assertEqual(t2.to_newick(), n2)
        self.assertEquals(t1.id, 6)
        self.assertEquals(t2.id, 6)

        self.assertRaises(TypeError, DendroMerge, m3)
        self.assertRaises(TypeError, DendroMerge, m4)
        self.assertRaises(TypeError, DendroMerge, m5)

if __name__ == "__main__":
    unittest.main()
