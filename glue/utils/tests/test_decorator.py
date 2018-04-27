from __future__ import absolute_import, division, print_function

from ..decorators import avoid_circular


def test_avoid_circular():

    class CircularCall(object):

        @avoid_circular
        def a(self):
            self.b()

        @avoid_circular
        def b(self):
            self.a()

    c = CircularCall()

    # Without avoid_circular, the following causes a recursion error
    c.a()
