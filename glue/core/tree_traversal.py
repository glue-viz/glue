from __future__ import absolute_import, division, print_function

from .tree import Tree


class TreeTraversal(object):
    def __init__(self, tree):
        if not isinstance(tree, Tree):
            raise TypeError("Input is not a tree object: %s" %
                            type(tree))
        self.tree = tree
        self.stack = [tree]

    def __iter__(self):
        return self

    def next(self):
        raise NotImplementedError()

    def __next__(self):
        return self.next()


class PreOrderTraversal(TreeTraversal):

    def next(self):
        if not self.stack:
            raise StopIteration()
        result = self.stack.pop()
        c = result.children

        for i in range(len(c)):
            self.stack.append(c[len(c) - i - 1])
        return result


class PostOrderTraversal(TreeTraversal):

    def __init__(self, tree):
        TreeTraversal.__init__(self, tree)
        self.popped = {}

    def next(self):
        if not self.stack:
            raise StopIteration()

        result = self.stack.pop()
        c = result.children
        if result in self.popped:
            return result

        self.popped[result] = 1
        self.stack.append(result)
        for i in range(len(c)):
            self.stack.append(c[len(c) - i - 1])

        return self.next()
