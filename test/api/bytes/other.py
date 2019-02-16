# -*- coding: UTF-8 -*-

import logging

import numpy as np

from lava.api.bytes import Array, Vector, Scalar, Struct
from lava.api.constants.spirv import Layout, Order
from lava.api.constants.vk import BufferUsage

from test.api.bytes.framework import TestByteRepresentation

logger = logging.getLogger(__name__)


class TestMatrixIn(TestByteRepresentation):

    def test(self):
        print "tbi"


class TestArrayIn(TestByteRepresentation):

    """Transferring multidimensional arrays into the shader"""

    def test1(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std140, binding = 0) readonly buffer dataIn {
                uint sca1;
                float[2][5] mat;
                double sca2;
            };
            layout(std430, binding = 1) writeonly buffer dataOut {
                float array[];
            }; // stdout430 so we have no array alignment fiddling

            void main() {
                // uint index = gl_GlobalInvocationID.x;

                for (int i = 0; i < mat.length(); i++) {
                    for (int j = 0; j < mat[i].length(); j++) {
                        //if (i * mat[i].length() + j < 2) {
                        //    continue;
                        //}
                        array[i * mat[i].length() + j] = mat[i][j];
                    }
                }

                //out1 = 999.;
                //array[0] = float(sca1);
                array[1] = float(sca2);


                // array[0] = mat.length();
                // array[1] = mat[0].length();
                // array[5] = mat[3][0];
            }
            """
        layout = Layout.STD140
        matrix_order = Order.ROW_MAJOR

        shape = (2, 5)
        array_expected = np.arange(np.product(shape), dtype=np.float32)

        scalar_uint = Scalar.uint()
        scalar_double = Scalar.double()

        array_outer = Array(Scalar.float(), shape, layout)

        order = [scalar_uint, array_outer, scalar_double]

        container = Struct(order, layout)
        print container

        values = {
            scalar_uint: 111,
            scalar_double: 222.,
            array_outer: array_expected.reshape(shape)
        }

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        print array_expected
        print array

    def test2(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std430, binding = 0) readonly buffer dataIn {
                float[3][5][11] mat;
            };
            layout(std430, binding = 1) writeonly buffer dataOut {
                float array[];
            }; // stdout430 so we have no array alignment fiddling

            void main() {
                // uint index = gl_GlobalInvocationID.x;

                for (int i = 0; i < mat.length(); i++) {
                    for (int j = 0; j < mat[i].length(); j++) {
                        for (int k = 0; k < mat[i][j].length(); k++) {
                            uint index = i * mat[i].length() * mat[i][j].length() + j * mat[i][j].length() + k;
                            array[index] = mat[i][j][k];
                        }
                    }
                }


                // array[0] = mat.length();
                // array[1] = mat[0].length();
                // array[5] = mat[3][0];
            }
            """
        layout = Layout.STD430
        matrix_order = Order.ROW_MAJOR

        shape = (3, 5, 11)
        array_expected = np.arange(np.product(shape), dtype=np.float32)

        array = Array(Scalar.float(), shape, layout)
        order = [array]

        container = Struct(order, layout)

        values = {
            array: array_expected.reshape(shape)
        }

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        print array_expected
        print array


class TestStructIn(TestByteRepresentation):
    """Transferring structs into the shader"""

    def test1(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            struct Struct1 {
              double var1;
              float[5] var2;
              ivec2 var3;
            };

            layout(std140, binding = 0) readonly buffer dataIn {
                uint var1;
                Struct1 var2;
                dvec3 var3;
            };
            layout(std430, binding = 1) writeonly buffer dataOut {
                float array[];
            }; // stdout430 so we have no array alignment fiddling

            void main() {
                // uint index = gl_GlobalInvocationID.x;

                array[0] = float(var1);
                array[1] = float(var2.var1);
                array[2] = float(var2.var2[0]);
                array[3] = float(var2.var2[1]);
                array[4] = float(var2.var2[2]);
                array[5] = float(var2.var2[3]);
                array[6] = float(var2.var2[4]);
                array[7] = float(var2.var3.x);
                array[8] = float(var2.var3.y);
                array[9] = float(var3.x);
                array[10] = float(var3.y);
                array[11] = float(var3.z);

            }
            """
        layout = Layout.STD140
        matrix_order = Order.ROW_MAJOR

        array_expected = np.zeros(12, dtype=np.float32)

        scalar_uint = Scalar.uint()
        scalar_double = Scalar.double()
        vector_double3 = Vector.dvec3()
        vector_int2 = Vector.ivec2()
        array = Array(Scalar.float(), 5, layout)
        struct = Struct([scalar_double, array, vector_int2], layout)

        order = [scalar_uint, struct, vector_double3]

        container = Struct(order, layout)

        values = {
            scalar_uint: 111,
            struct: {
                scalar_double: 123.5,
                array: np.arange(5, dtype=np.float32),
                vector_int2: [101, 99]
            },
            vector_double3: [-1., -3., -5.]
        }

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        print array_expected
        print array

    def test2(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            struct Struct1 {
              double var1;
              ivec3 var2;
              uint var3;
            };

            layout(std140, binding = 0) readonly buffer dataIn {
                uint var1;
                Struct1[3][5] var2;
            };
            layout(std430, binding = 1) writeonly buffer dataOut {
                float array[];
            }; // stdout430 so we have no array alignment fiddling

            void main() {
                array[0] = float(var1);

                uint off = 1;
                for (uint i1 = 0, n1 = var2.length(); i1 < n1; i1++) {
                    for (uint i2 = 0, n2 = var2[0].length(); i2 < n2; i2++) {
                        uint n3 = 5;
                        array[off + i1 * n2 * n3 + i2 * n3 + 0] = float(var2[i1][i2].var1);
                        array[off + i1 * n2 * n3 + i2 * n3 + 1] = float(var2[i1][i2].var2.x);
                        array[off + i1 * n2 * n3 + i2 * n3 + 2] = float(var2[i1][i2].var2.y);
                        array[off + i1 * n2 * n3 + i2 * n3 + 3] = float(var2[i1][i2].var2.z);
                        array[off + i1 * n2 * n3 + i2 * n3 + 4] = float(var2[i1][i2].var3);
                    }
                }

            }
            """
        layout = Layout.STD140
        matrix_order = Order.ROW_MAJOR

        array_expected = np.zeros(76, dtype=np.float32)

        scalar_uint = Scalar.uint()
        scalar_double = Scalar.double()
        vector = Vector.ivec3()
        struct = Struct([scalar_double, vector, scalar_uint], layout)

        # array_inner = Array(struct, 5, layout, matrix_order)
        # array_outer = Array(array_inner, 3, layout, matrix_order)
        # container = Block([scalar_uint, array_outer], layout, matrix_order)
        # values_array = [{scalar_uint: i + 1, scalar_double: i + 0.1, vector: [-i, -i, -i]} for i in
        #                 range(array_inner.length())]
        # values = {scalar_uint: 111, array_outer: [values_array for _ in range(array_outer.length())]}

        array_outer2 = Array(struct, (3, 5), layout)
        container = Struct([scalar_uint, array_outer2], layout)
        values_array = []
        for i in range(3):
            values_array.append([])
            for j in range(5):
                values_array[i].append({scalar_uint: j + 1, scalar_double: j + 0.1, vector: [-j, -j, -j]})
        values = {scalar_uint: 111, array_outer2: values_array}

        # TODO: works if struct padding_after is not used...
        print container

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        np.set_printoptions(precision=3, suppress=True)
        print array_expected
        print array