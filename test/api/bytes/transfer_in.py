# -*- coding: UTF-8 -*-

import itertools
import logging
import unittest

import numpy as np

from lava.api.bytes import Array, Matrix, Vector, Scalar, Struct
from lava.api.constants.spirv import DataType, Layout, Order
from lava.api.constants.vk import BufferUsage

from test.api.base import GlslBasedTest, Random

logger = logging.getLogger(__name__)


class TestCpuToShader(GlslBasedTest):
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

        glsl_structs = "\n\n".join([cls.build_glsl_struct_definition(struct) for struct in structs])
        glsl_block = cls.build_glsl_block_definition(container, binding=0, usage=buffer_usage)
        glsl_assignments, _ = cls.build_glsl_assignments(container.definitions)

        return template.format(glsl_structs, glsl_block, glsl_assignments)

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

        for _ in range(5):
            containers_std140.append(Struct(rng.permutation(variables), Layout.STD140))
            containers_std430.append(Struct(rng.permutation(variables), Layout.STD430))

        for container in containers_std140:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)
        for container in containers_std430:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)

    def test_scalars_and_vectors_and_matrices(self):
        rng = np.random.RandomState(123)

        variables = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        variables += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        matrix_combinations = itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])
        variables_std140 = variables + [Matrix(n, m, dtype, Layout.STD140) for n, m, dtype in matrix_combinations]
        variables_std430 = variables + [Matrix(n, m, dtype, Layout.STD430) for n, m, dtype in matrix_combinations]

        containers_std140 = [Struct(variables_std140, Layout.STD140)]
        containers_std430 = [Struct(variables_std430, Layout.STD430)]

        for _ in range(5):
            containers_std140.append(Struct(rng.permutation(variables_std140), Layout.STD140))
            containers_std430.append(Struct(rng.permutation(variables_std430), Layout.STD430))

        for container in containers_std140:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)
        for container in containers_std430:
            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)

    def test_matrices(self):
        # skipping ROW_MAJOR order for now since the glsl generation does not support it
        for n, m, dtype, order, layout in itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE],
                                                            [Order.COLUMN_MAJOR], [Layout.STD140, Layout.STD430]):
            matrix = Matrix(n, m, dtype, layout, order)
            container = Struct([matrix], layout)

            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)

    def test_array_of_scalars(self):
        rng = np.random.RandomState(123)
        scalar_types = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]

        for definition, layout, _ in itertools.product(scalar_types, [Layout.STD140, Layout.STD430], range(5)):
            container = Struct([Array(definition, Random.shape(rng, 5, 7), layout)], layout)

            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)

    def test_array_of_vectors(self):
        rng = np.random.RandomState(123)
        vector_types = [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for definition, layout, _ in itertools.product(vector_types, [Layout.STD140, Layout.STD430], range(5)):
            container = Struct([Array(definition, Random.shape(rng, 3, 5), layout)], layout)

            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)

    def test_array_of_matrices(self):
        # skipping ROW_MAJOR order for now since the glsl generation does not support it
        rng = np.random.RandomState(123)
        matrix_combinations = itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])

        for (n, m, dtype), layout, _ in itertools.product(matrix_combinations, [Layout.STD140, Layout.STD430], range(3)):
            matrix = Matrix(n, m, dtype, layout)
            container = Struct([Array(matrix, Random.shape(rng, 3, 5), layout)], layout)

            self.run_test(container, [], BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, [], BufferUsage.UNIFORM_BUFFER)

    def test_array_of_structs(self):
        rng = np.random.RandomState(123)
        simple = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        simple += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for layout, _ in itertools.product([Layout.STD140, Layout.STD430], range(5)):
            struct = Struct(rng.choice(simple, size=3, replace=False), layout, type_name="SomeStruct")
            array = Array(struct, Random.shape(rng, 3, 5), layout)
            container = Struct([array], layout)

            self.run_test(container, [struct], BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, [struct], BufferUsage.UNIFORM_BUFFER)

    def test_nested_with_structs(self):
        rng = np.random.RandomState(123)

        simple = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        simple += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for layout, _ in itertools.product([Layout.STD140, Layout.STD430], range(5)):
            struct = Struct(rng.choice(simple, size=3, replace=False), layout, type_name="SomeStruct")
            structs = [struct]

            for _ in range(4):
                members = [structs[-1]] + rng.choice(simple, size=2, replace=False).tolist()
                structs.append(Struct(rng.permutation(members), layout, type_name="SomeStruct{}".format(len(structs))))

            container = structs[-1]
            structs = structs[:-1]

            self.run_test(container, structs, BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, structs, BufferUsage.UNIFORM_BUFFER)

    def test_nested_with_arrays_of_structs(self):
        rng = np.random.RandomState(123)

        simple = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        simple += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for layout, _ in itertools.product([Layout.STD140, Layout.STD430], range(5)):
            struct = Struct(rng.choice(simple, size=3, replace=False), layout, type_name="SomeStruct")
            structs = [struct]
            arrays = [Array(struct, Random.shape(rng, 2, 3), layout)]

            for _ in range(2):
                members = [arrays[-1]] + rng.choice(simple, size=2, replace=False).tolist()
                structs.append(Struct(rng.permutation(members), layout, type_name="SomeStruct{}".format(len(structs))))
                arrays.append(Array(structs[-1], Random.shape(rng, 2, 3), layout))

            container = structs[-1]
            structs = structs[:-1]

            self.run_test(container, structs, BufferUsage.STORAGE_BUFFER)
            if layout == Layout.STD140:
                self.run_test(container, structs, BufferUsage.UNIFORM_BUFFER)

    def test_arb_example_std140(self):
        layout = Layout.STD430

        struct_a = Struct([
            Scalar.int(),
            Vector.uvec2()  # actually bvec2
        ], layout, type_name="structA")

        struct_b = Struct([
            Vector.uvec3(),
            Vector.vec2(),
            Array(Scalar.float(), 2, layout),
            Vector.vec2(),
            # Array(mat3, 2, layout)
        ], layout, type_name="structB")

        container = Struct([
            Scalar.float(),
            Vector.vec2(),
            Vector.vec3(),
            struct_a,
            Scalar.float(),
            Array(Scalar.float(), 2, layout),
            # mat2x3
            Array(struct_b, 2, layout)
        ], layout)

        self.run_test(container, [struct_a, struct_b], BufferUsage.STORAGE_BUFFER)


if __name__ == "__main__":
    unittest.main()
