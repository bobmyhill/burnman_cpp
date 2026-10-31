# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_decorators.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
from burnman_cpp.utils.misc import copy_documentation


class ClassA_for_copy_documentation(object):

    def my_func(self):
        """A.my_func doc"""
        pass

    def func_with_args(self, x, y):
        """doc for func_with_args"""
        return x + y


class ClassB_for_copy_documentation(object):

    @copy_documentation(ClassA_for_copy_documentation.my_func)
    def some_func(self):
        """B.some_func doc"""
        pass

    @copy_documentation(ClassA_for_copy_documentation.my_func)
    def some_func_without_doc(self):
        pass


class test_decorators_copy_documentation(BurnManTest):

    def test(self):
        self.assertEqual(ClassA_for_copy_documentation.my_func.__doc__, "A.my_func doc")
        self.assertEqual(
            ClassB_for_copy_documentation.some_func.__doc__,
            "(copied from my_func):\nA.my_func doc\nB.some_func doc",
        )
        self.assertEqual(
            ClassB_for_copy_documentation.some_func_without_doc.__doc__,
            "(copied from my_func):\nA.my_func doc",
        )

    def test_with_args(self):
        """early versions couldn't deal with functions with parameters"""

        class C(object):

            @copy_documentation(ClassA_for_copy_documentation.func_with_args)
            def another(self):
                return 1.0

        self.assertEqual(
            C.another.__doc__, "(copied from func_with_args):\ndoc for func_with_args"
        )
        c = C()
        self.assertEqual(c.another(), 1.0)

    def test_with_args2(self):
        """early versions couldn't deal with functions with parameters"""

        class C(object):

            @copy_documentation(ClassA_for_copy_documentation.func_with_args)
            def bla(self, z):
                """bla"""
                return 1.0 + z

        self.assertEqual(
            C.bla.__doc__, "(copied from func_with_args):\ndoc for func_with_args\nbla"
        )
        c = C()
        self.assertEqual(c.bla(1.0), 2.0)
