# -*- coding: UTF-8 -*-

import logging
import pprint
import unittest

import numpy as np

from lava.api.bytes import Array, ByteCache, Vector, Scalar, Struct
from lava.api.constants.spirv import Layout, Order
from lava.api.constants.vk import BufferUsage

from test.api.base import GlslBasedTest

logger = logging.getLogger(__name__)


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

