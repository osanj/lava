# -*- coding: UTF-8 -*-

import pprint
import unittest

import numpy as np

from lava.api.bytes import Array, Matrix, Vector, Scalar, Struct
from lava.api.cache import ByteCache
from lava.api.constants.spirv import DataType, Layout, Order
from lava.api.constants.vk import BufferUsage
from lava.api.util import NdArray

from test.api.base import GlslBasedTest


class TestByteCache(GlslBasedTest):

    @unittest.skip("test for development purposes")
    def test_manually(self):
        # byte cache test
        buffer_usage = BufferUsage.STORAGE_BUFFER
        buffer_layout = Layout.STD430
        buffer_order = Order.ROW_MAJOR

        struct1 = Struct([Vector.vec3(), Vector.ivec2()], buffer_layout, member_names=["a", "b"], type_name="structB")
        struct2 = Struct([Scalar.double(), Scalar.double(), struct1], buffer_layout, type_name="structC")

        structs = [struct1, struct2]

        variables = [
            Scalar.uint(),
            Array(Vector.vec2(), (5, 2, 3), buffer_layout),
            Array(Scalar.float(), 5, buffer_layout),
            struct2,  # this struct needs padding at the end
            Scalar.uint(),
            Array(struct1, (2, 3), buffer_layout)
        ]

        container = Struct(variables, buffer_layout, type_name="block")

        cache = ByteCache(container)

        print("")
        print("")
        pprint.pprint(cache.values)
        print(cache[-1][0][0]["a"])
        print("")
        print("")
        pprint.pprint(cache)
        print(cache[-1][0][0])
        print("")
        print("")
        pprint.pprint(cache.get_as_dict())

    @classmethod
    def assert_cache_struct_equals(cls, cache, value):
        for d in cache.definition.definitions:
            if isinstance(d, Struct):
                cls.assert_cache_struct_equals(cache[d], value)

            elif isinstance(d, Array) and Array.is_array_of_structs(d):
                cls.assert_cache_array_equals(cache[d], d.shape(), value)

            else:
                np.testing.assert_equal(cache[d], value)

    @classmethod
    def assert_cache_array_equals(cls, caches, shape, value):
        for idx in NdArray.iterate(shape):
            cache_for_idx = NdArray.get(caches, idx)
            cls.assert_cache_struct_equals(cache_for_idx, value)

    def test_defaults(self):
        layout = Layout.STD430

        container = Struct([
            Scalar.uint(),
            Vector.dvec4(),
            Matrix(3, 3, DataType.FLOAT, layout),
            Array(Struct([
                Scalar.double(),
            ], layout), (2,), layout),
            Struct([
                Array(Scalar.float(), (4, 4), layout),
                Array(Vector.vec2(), (5,), layout),
                Array(Matrix(3, 2, DataType.FLOAT, layout), (6, 2, 3), layout),
            ], layout),
        ], layout)

        cache = ByteCache(container)
        cache.set_defaults()
        self.assert_cache_struct_equals(cache, 0)

    def test_invalidate_array_members(self):
        pass
