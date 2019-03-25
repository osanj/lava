# -*- coding: UTF-8 -*-

import itertools
import unittest

import numpy as np

from lava.api.bytes import Array, Matrix, Vector, Scalar, Struct
from lava.api.constants.spirv import DataType, Layout
from lava.api.constants.vk import BufferUsage

from test.api.base import GlslBasedTest, Random


class TestByteCodeInspection(GlslBasedTest):

    @classmethod
    def build_glsl_program(cls, container_data, structs=()):
        template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
        
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        
            {}
        
            {}
                
            void main() {{
            }}"""

        glsl_structs = "\n\n".join([cls.build_glsl_struct_definition(struct) for struct in structs])

        glsl_blocks = []
        for container, binding, usage in container_data:
            glsl_blocks.append(cls.build_glsl_block_definition(container, binding, usage))

        return template.format(glsl_structs, "\n\n".join(glsl_blocks))

    @unittest.skip("test for development purposes")
    def test_manual(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=16) in;
            //layout(local_size_x=100, local_size_y=1, local_size_z=1) in;

            struct Data1 {
                vec3 var1;
                ivec2 var2;
            };

            struct Data2 {
                double var1;
                double var2;
                Data1 var3;
                dmat4 var4;
            };

            struct Data3 {
                mat3x3[3] var1;
            };

            //layout(std140, binding = 0, row_major) uniform uniIn
            layout(std430, binding = 0, row_major) readonly buffer bufIn
            {
                bool flag2;
                vec2[5][2][3] abc;
                float bufferIn[5];
                //bool flag;
                mat3x4 model;
                dmat3x2[99] modelz;
                Data1[2] datas1;
                Data2 datas2;
            };

            layout(std140, binding = 1) writeonly buffer bufOut
            //layout(std430, binding = 1) writeonly buffer bufOut
            {
                uint width;
                uint height;
                layout(row_major) Data2 datas3;
                float bufferOut[4];
                uint asd;
                dmat3x2[99] modelz2;
            };

            void main() {
                uint index = gl_GlobalInvocationID.x;
                //bufferOut[index] = bufferIn[index] + 1;

                Data3 something;
                something.var1[0][0].x = 1;
            }
            """

        shader = self.shader_from_txt(glsl)

        shader.inspect()

        print(shader.byte_code)

        print("")
        print("scalars")
        print(shader.byte_code.types_scalar)
        print("")
        print("vectors")
        print(shader.byte_code.types_vector)
        print("")
        print("matrices")
        print(shader.byte_code.types_matrix)
        print("")
        print("array")
        print(shader.byte_code.types_array)
        print("")
        print("struct")
        print(shader.byte_code.types_struct)
        print("")
        print("names")
        names = []
        for idx in shader.byte_code.types_struct:
            struct_name, member_names = shader.byte_code.find_names(idx)
            offsets = shader.byte_code.find_offsets(idx)
            names.append("  {}) {} {{ {} }}".format(idx, struct_name, ", ".join(
                ["{}({})".format(mname, offsets.get(i)) for i, mname in enumerate(member_names)])))
        print("\n".join(names))
        print("")
        print("blocks")
        print(shader.byte_code.find_blocks())
        print("")

    def test_detection_type_nested_with_structs(self):
        rng = np.random.RandomState(321)

        simple = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        simple += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for layout, _ in itertools.product([Layout.STD140, Layout.STD430], range(5)):
            matrices = [Matrix(n, m, dtype, layout) for n, m, dtype in
                        itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])]
            simple_and_matrices = simple + matrices

            struct = Struct(rng.choice(simple_and_matrices, size=3, replace=False), layout, type_name="SomeStruct")
            structs = [struct]

            for _ in range(4):
                members = [structs[-1]] + rng.choice(simple_and_matrices, size=2, replace=False).tolist()
                structs.append(Struct(rng.permutation(members), layout, type_name="SomeStruct{}".format(len(structs))))

            container = structs[-1]
            structs = structs[:-1]

            glsl = self.build_glsl_program(((container, 0, BufferUsage.STORAGE_BUFFER),), structs)
            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            definition, _ = shader.get_block(0)
            self.assertTrue(container.compare(definition, quiet=True))

    def test_detection_type_arrays(self):
        rng = np.random.RandomState(321)
        variables = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        variables += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        for definition, layout, _ in itertools.product(variables, [Layout.STD140, Layout.STD430], range(3)):
            container = Struct([Array(definition, Random.shape(rng, 3, 5), layout)], layout)

            glsl = self.build_glsl_program(((container, 0, BufferUsage.STORAGE_BUFFER),))
            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            detected_definition, _ = shader.get_block(0)
            self.assertTrue(container.compare(detected_definition, quiet=True))

            if isinstance(definition, Vector):
                if definition.length() < 3 and definition.dtype != DataType.DOUBLE:
                    self.assertEqual(detected_definition.layout, layout)

    def test_detection_type_arrays_of_matrices(self):
        rng = np.random.RandomState(321)
        matrix_attributes = itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])

        for (n, m, dtype), layout, _ in itertools.product(matrix_attributes, [Layout.STD140, Layout.STD430], range(3)):
            matrix = Matrix(n, m, dtype, layout)
            container = Struct([Array(matrix, Random.shape(rng, 3, 5), layout)], layout)

            glsl = self.build_glsl_program(((container, 0, BufferUsage.STORAGE_BUFFER),))
            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            detected_definition, _ = shader.get_block(0)
            self.assertTrue(container.compare(detected_definition, quiet=True))

    def test_detection_type_bools(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout({layout}, binding = 0) buffer Buffer {{
                bool var1;
                bool[720][1280] var2;
                bvec2 var3;
                bvec3 var4;
                bvec4 var5;
                bvec3[5] var6;
            }};

            void main() {{}}
            """

        for layout in (Layout.STD140, Layout.STD430):
            # the vulkan spir-v compiler turns bools into uints
            expected_definition = Struct([
                Scalar.uint(),
                Array(Scalar.uint(), (720, 1280), layout),
                Vector.uvec2(),
                Vector.uvec3(),
                Vector.uvec4(),
                Array(Vector.uvec3(), 5, layout)
            ], layout)

            shader = self.shader_from_txt(glsl.format(layout=layout), verbose=False)
            shader.inspect()

            detected_definition, _ = shader.get_block(0)
            equal = expected_definition.compare(detected_definition, quiet=True)
            self.assertTrue(equal)

    def test_detection_layout_stdxxx_ssbo(self):
        variables = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        variables += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        binding = 0
        usage = BufferUsage.STORAGE_BUFFER

        glsl_std140 = self.build_glsl_program(((Struct(variables, Layout.STD140), binding, usage),))
        glsl_std430 = self.build_glsl_program(((Struct(variables, Layout.STD430), binding, usage),))

        glsls = [glsl_std140, glsl_std430]

        for glsl in glsls:
            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            definition, detected_usage = shader.get_block(binding)

            self.assertEqual(detected_usage, usage)
            self.assertEqual(definition.layout, Layout.STDXXX)

    def test_detection_layout_stdxxx_ubo(self):
        variables = [Scalar.uint(), Scalar.int(), Scalar.float(), Scalar.double()]
        variables += [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        binding = 0
        usage = BufferUsage.UNIFORM_BUFFER

        glsl = self.build_glsl_program(((Struct(variables, Layout.STD140), binding, usage),))
        shader = self.shader_from_txt(glsl, verbose=False)
        shader.inspect()

        definition, detected_usage = shader.get_block(binding)

        self.assertEqual(detected_usage, usage)
        self.assertEqual(definition.layout, Layout.STD140)  # uniform buffer objects can not use std430

    def test_detection_name(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(std140, binding = 0) buffer BufferA {
                float var1;
                double var2;
                int var3;
                uint var4;
                vec3 var5;
                ivec4 var6;
                dvec2[5][5] var7;
            };

            void main() {}
            """
        shader = self.shader_from_txt(glsl, verbose=False)
        shader.inspect()

        definition, _ = shader.get_block(0)
        self.assertListEqual(definition.member_names, ["var{}".format(i) for i in range(1, 8)])

    def test_detection_binding(self):
        container = Struct([Scalar.int(), Vector.vec3()], Layout.STD140)

        for binding, usage in itertools.product([0, 1, 2, 3, 4, 99, 512], [BufferUsage.UNIFORM_BUFFER, BufferUsage.STORAGE_BUFFER]):
            glsl = self.build_glsl_program(((container, binding, usage),))
            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            detected_definition, detected_usage = shader.get_block(binding)

            self.assertEqual(detected_usage, usage)
            equal = container.compare(detected_definition, quiet=True)
            self.assertTrue(equal)

    def test_struct_shared_between_different_layouts(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            struct Shared {
                int var1;
                double var2;
            };

            layout(std140, binding = 0) buffer BufferA {
                uvec2 varA1;
                Shared varA2; // expected offset 16
            };
            
            layout(std430, binding = 1) buffer BufferB {
                uvec2 varB1;
                Shared varB2; // expected offset 8
            };

            void main() {}
            """
        shared_std140 = Struct([Scalar.int(), Scalar.double()], Layout.STD140)
        shared_std430 = Struct([Scalar.int(), Scalar.double()], Layout.STD430)

        container_std140 = Struct([Vector.uvec2(), shared_std140], Layout.STD140)
        container_std430 = Struct([Vector.uvec2(), shared_std430], Layout.STD430)

        shader = self.shader_from_txt(glsl, verbose=False)
        shader.inspect()

        definition0, _ = shader.get_block(0)
        definition1, _ = shader.get_block(1)
        self.assertTrue(container_std140.compare(definition0, quiet=True))
        self.assertFalse(container_std140.compare(definition1, quiet=True))
        self.assertFalse(container_std430.compare(definition0, quiet=True))
        self.assertTrue(container_std430.compare(definition1, quiet=True))

    def test_struct_unused(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            struct NonUsed {
                uint var1;
                dmat4x3 var2;
            };

            struct Shared {
                int var1;
                double var2;
            };

            layout(std140, binding = 0) buffer BufferA {
                uvec2 varA1;
                Shared varA2; // expected offset 16
            };

            layout(std430, binding = 1) buffer BufferB {
                uvec2 varB1;
                Shared varB2; // expected offset 8
            };

            void main() {
                NonUsed nonUsed;
                nonUsed.var2[0].x = 1;
            }
            """
        shader = self.shader_from_txt(glsl, verbose=False)
        shader.inspect()  # just test whether this blows up


if __name__ == "__main__":
    unittest.main()
