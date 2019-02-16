# -*- coding: UTF-8 -*-

import logging

import numpy as np

from lava.api.bytes import Array, Vector, Scalar, Struct
from lava.api.constants.spirv import Layout, Order
from lava.api.constants.vk import BufferUsage

from test.api.bytes.base import TestByteRepresentation

logger = logging.getLogger(__name__)


class TestShaderToCpu(TestByteRepresentation):
    """Transferring data in arbitrary order from the shader"""

    @classmethod
    def build_glsl_program(cls, container, structs, buffer_usage):
        template = """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

layout(std430, binding = 0) buffer dataOut {{
    float array[];
}}; // stdout430 so we have no array alignment fiddling

{}

{}


void main() {{
{}
}}"""

        struct_definitions = "\n\n".join([cls.build_glsl_struct_definition(struct) for struct in structs])
        block_definition = cls.build_glsl_block_definition(container, binding=1, usage=buffer_usage)
        assignments, _ = cls.build_glsl_assignments(container.definitions, to_array=False)

        return template.format(struct_definitions, block_definition, assignments)


    #@unittest.skip("test for development purposes")
    def test_manually(self):
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

        values_expected, array = self.build_values(container.definitions)
        array = np.array(array, dtype=np.float32)

        bytez_in = array.tobytes()  # std430
        bytez_out = self.run_program(glsl, bytez_in, container.size())
        values = container.from_bytes(bytez_out)

        register = {}
        steps = {Scalar: 0, Vector: 0, Array: 0, Struct: 0}

        for struct in structs + [container]:
            self.build_register(register, struct, steps)

        values_ftd = self.format_values(container, values, register)
        values_expected_ftd = self.format_values(container, values_expected, register)

        print ""
        self.print_formatted_values(values_ftd)
        print ""
        self.print_formatted_values(values_expected_ftd)
        print ""

        print ""

        print values_ftd == values_expected_ftd

    def test_manually2(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            struct Data1 {
                dvec4 var1;
                uint var2;
                ivec2[3] var3;
            };

            layout(std430, binding = 0) readonly buffer dataIn {
                float[16] array;
            }; // stdout430 so we have no array alignment fiddling

            layout(std140, binding = 1) writeonly buffer dataOut {
                double output1;
                vec3 output2;
                Data1 output3;
            };

            void main() {
                output1 = float(array[0]);
                output2.x = float(array[1]);
                output2.y = float(array[2]);
                output2.z = float(array[3]);
                output3.var1.x = double(array[4]);
                output3.var1.y = double(array[5]);
                output3.var1.z = double(array[6]);
                output3.var1.w = double(array[7]);
                output3.var2 = uint(array[8]);
                output3.var3[0].x = int(array[9]);
                output3.var3[0].y = int(array[10]);
                output3.var3[1].x = int(array[11]);
                output3.var3[1].y = int(array[12]);
                output3.var3[2].x = int(array[13]);
                output3.var3[2].y = int(array[14]);
            }
            """
        layout = Layout.STD140
        matrix_order = Order.ROW_MAJOR

        data1 = Struct([
            Vector.dvec4(),
            Scalar.int(),
            Array(Vector.ivec2(), 3, layout)
        ], layout, type_name="Data1")

        container = Struct([
            Scalar.double(),
            Vector.vec3(),
            data1,
        ], layout)

        bytez_in = np.arange(16, dtype=np.float32).tobytes()  # std430
        bytez_out = self.run_program(glsl, bytez_in, container.size())

        print container.from_bytes(bytez_out)