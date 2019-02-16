# -*- coding: UTF-8 -*-

import itertools
import logging
import unittest

import numpy as np

from lava.api.bytes import Array, Vector, Scalar, Struct
from lava.api.constants.spirv import DataType, Layout, Order
from lava.api.constants.vk import BufferUsage

from test.api.bytes.base import TestByteRepresentation, Random

logger = logging.getLogger(__name__)


class TestCpuToShader(TestByteRepresentation):
    """Transferring data in arbitrary order into the shader"""

    @classmethod
    def build_glsl_program(cls, container, structs, buffer_usage):
        template = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

{}

{}

layout(std430, binding = 1) buffer dataOut {{
    float array[];
}}; // stdout430 so we have no array alignment fiddling

void main() {{
{}
}}"""

        struct_definitions = "\n\n".join([cls.build_glsl_struct_definition(struct) for struct in structs])
        block_definition = cls.build_glsl_block_definition(container, usage=buffer_usage)
        assignments, _ = cls.build_glsl_assignments(container.definitions)

        return template.format(struct_definitions, block_definition, assignments)

    def run_test(self, container, structs, buffer_usage):
        glsl = self.build_glsl_program(container, structs, buffer_usage)

        values, array_expected = self.build_values(container.definitions)
        array_expected = np.array(array_expected, dtype=np.float32)

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes, usage_input=buffer_usage, verbose=False)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        diff = array_expected - array
        equal = (diff == 0).all()
        if not equal:
            np.set_printoptions(precision=3, suppress=True)
            print "{}\n".format(glsl)
            print "{}\nexpected\n{}\nactual\n{}\ndiff\n{}\n".format(container, array_expected, array, diff)
        self.assertTrue(equal)

    @unittest.skip("test for development purposes")
    def test_manual(self):
        buffer_usage = BufferUsage.STORAGE_BUFFER
        buffer_layout = Layout.STD140
        buffer_order = Order.ROW_MAJOR

        structA = Struct([Vector.ivec2(), Scalar.double()], buffer_layout, member_names=["a", "b"], type_name="structA")
        structB = Struct([Scalar.uint(), Scalar.double()], buffer_layout, type_name="structB")
        structC = Struct([structB, Vector.ivec2()], buffer_layout, type_name="structC")

        structs = [structA, structB, structC]

        variables = [
            Vector.vec3(),
            Vector.ivec4(),
            Array(structC, 2, buffer_layout),
            Vector.ivec4(),
            Scalar.uint(),
            Array(Scalar.double(), (5, 2), buffer_layout),
            Scalar.int(),
            Array(Vector.vec4(), (2, 3, 4), buffer_layout),
            Vector.dvec2(),
            structA
        ]

        container = Struct(variables, buffer_layout, type_name="block")
        print container

        glsl = self.build_glsl_program(container, structs, buffer_usage)
        # print glsl

        values, array_expected = self.build_values(container.definitions)
        array_expected = np.array(array_expected, dtype=np.float32)

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes, usage_input=buffer_usage)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        print array_expected
        print array
        print "equal", ((array_expected - array) == 0).all()

    def test_scalars_and_vectors(self):
        rng = np.random.RandomState(123)

        variables = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        variables += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        containers_std140 = [Struct(variables, Layout.STD140)]
        containers_std430 = [Struct(variables, Layout.STD430)]

        for i in range(5):
            containers_std140.append(Struct(rng.permutation(variables), Layout.STD140))
            containers_std430.append(Struct(rng.permutation(variables), Layout.STD430))

        for container in containers_std140 + containers_std430:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)

    def test_arrays_of_scalars(self):
        scalar_types = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        rng = np.random.RandomState(123)
        containers = []

        for definition, _ in itertools.product(scalar_types, range(5)):
            containers.append(Struct([Array(definition, Random.shape(rng, 5, 10), Layout.STD140)], Layout.STD140))
            containers.append(Struct([Array(definition, Random.shape(rng, 5, 10), Layout.STD430)], Layout.STD430))

        for container in containers:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)

    def test_vectors(self):
        vector_types = [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]
        rng = np.random.RandomState(123)
        containers = []

        for definition, _ in itertools.product(vector_types, range(5)):
            containers.append(Struct([Array(definition, Random.shape(rng, 5, 10), Layout.STD140)], Layout.STD140))
            containers.append(Struct([Array(definition, Random.shape(rng, 5, 10), Layout.STD430)], Layout.STD430))

        for container in containers:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)


# def test_manually2(self):
    #     # buffer padding test
    #     buffer_usage = BufferUsage.STORAGE_BUFFER
    #     buffer_layout = Layout.STD430
    #     buffer_order = Order.ROW_MAJOR
    #
    #     struct1 = Struct([Vector.vec3(), Vector.ivec2()], buffer_layout, type_name="structB")
    #     struct2 = Struct([Scalar.double(), Scalar.double(), struct1], buffer_layout, type_name="structC")
    #
    #     structs = [struct1, struct2]
    #
    #     variables = [
    #         Scalar.uint(),
    #         Array(Vector.vec2(), (5, 2, 3), buffer_layout),
    #         Array(Scalar.float(), 5, buffer_layout),
    #         struct2,  # this struct needs padding at the end
    #         Scalar.uint(),
    #         Array(struct1, 2, buffer_layout)
    #     ]
    #
    #     container = Struct(variables, buffer_layout, type_name="block")
    #     print container
    #
    #     glsl = self.build_glsl_program(container, structs, buffer_usage)
    #     # print glsl
    #
    #     values, array_expected = self.build_values(container.definitions)
    #     array_expected = np.array(array_expected, dtype=np.float32)
    #
    #     bytez_input = container.to_bytes(values)
    #     bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes, usage_input=buffer_usage)
    #     array = np.frombuffer(bytez_output, dtype=array_expected.dtype)
    #
    #     print array_expected
    #     print array
    #     print "equal", ((array_expected - array) == 0).all()


if __name__ == "__main__":
    unittest.main()
