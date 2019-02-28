# -*- coding: UTF-8 -*-

import itertools
import logging
import unittest

import numpy as np

from lava.api.bytes import ByteCache, Matrix, Scalar, Struct, Vector
from lava.api.constants.spirv import DataType, Layout, Order

from test.api.base import GlslBasedTest

logger = logging.getLogger(__name__)


class CombinedTest(GlslBasedTest):

    LAYOUTS = [Layout.STD140, Layout.STD430]
    LAYOUT_MAP = {Layout.STD140: "std140", Layout.STD430: "std430"}
    ORDERS = [Order.COLUMN_MAJOR, Order.ROW_MAJOR]
    ORDERS_MAP = {Order.COLUMN_MAJOR: "column_major", Order.ROW_MAJOR: "row_major"}

    def test_pass_through_array_of_scalars(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout({layout_in}, binding = 0) buffer BufferA {{
                {dtype}[720][1280][3] imageIn;
            }};

            layout({layout_out}, binding = 1) buffer BufferB {{
                {dtype}[720][1280][3] imageOut;
            }};

            void main() {{
                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);
                int c = int(pixel.z);

                imageOut[h][w][c] = imageIn[h][w][c];
            }}
            """

        rng = np.random.RandomState(123)
        w = 1280
        h = 720

        for layout_in, layout_out, dtype in itertools.product(self.LAYOUTS, self.LAYOUTS, DataType.ALL):
            scalar = Scalar.of(dtype)
            im = rng.randint(0, 255, size=(h, w, 3)).astype(scalar.numpy_dtype())

            glsl = glsl_template.format(**{
                "layout_in": self.LAYOUT_MAP[layout_in],
                "layout_out": self.LAYOUT_MAP[layout_out],
                "dtype": scalar.glsl_dtype()
            })

            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            cache_in = ByteCache(shader.get_block_definition(0))
            cache_in["imageIn"] = im
            bytez_in = cache_in.definition.to_bytes(cache_in.get_as_dict())

            cache_out = ByteCache(shader.get_block_definition(1))
            bytez_out_count = cache_out.definition.size()
            bytez_out = self.run_compiled_program(shader, bytez_in, bytez_out_count, groups=im.shape)
            cache_out.set_from_dict(cache_out.definition.from_bytes(bytez_out))

            self.assertTrue((cache_out["imageOut"] == im).all())

    def test_pass_through_array_of_vectors(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout({layout_in}, binding = 0) buffer BufferA {{
                {dtype}[720][1280] imageIn;
            }};

            layout({layout_out}, binding = 1) buffer BufferB {{
                {dtype}[720][1280] imageOut;
            }};

            void main() {{
                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);

                imageOut[h][w] = imageIn[h][w];
            }}
            """

        rng = np.random.RandomState(123)
        w = 1280
        h = 720

        for layout_in, layout_out, n, dtype in itertools.product(self.LAYOUTS, self.LAYOUTS, range(2, 5), DataType.ALL):
            vector = Vector(n, dtype)
            im = rng.randint(0, 255, size=(h, w, n)).astype(vector.scalar.numpy_dtype())

            glsl = glsl_template.format(**{
                "layout_in": self.LAYOUT_MAP[layout_in],
                "layout_out": self.LAYOUT_MAP[layout_out],
                "dtype": vector.glsl_dtype()
            })

            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            cache_in = ByteCache(shader.get_block_definition(0))
            cache_in["imageIn"] = im
            bytez_in = cache_in.definition.to_bytes(cache_in.get_as_dict())

            cache_out = ByteCache(shader.get_block_definition(1))
            bytez_out_count = cache_out.definition.size()
            bytez_out = self.run_compiled_program(shader, bytez_in, bytez_out_count, groups=(h, w, 1))
            cache_out.set_from_dict(cache_out.definition.from_bytes(bytez_out))

            self.assertTrue((cache_out["imageOut"] == im).all())

    def test_pass_through_array_of_matrices(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout({layout_in}, {order_in}, binding = 0) buffer BufferA {{
                {dtype}[32][48] dataIn;
            }};

            layout({layout_out}, {order_out}, binding = 1) buffer BufferB {{
                {dtype}[32][48] dataOut;
            }};

            void main() {{
                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);

                dataOut[h][w] = dataIn[h][w];
            }}
            """

        rng = np.random.RandomState(123)
        w = 48
        h = 32

        matrix_combinations = itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])
        layout_order_combinations = itertools.product(self.LAYOUTS, self.LAYOUTS, self.ORDERS, self.ORDERS)

        for combos1, combos2 in itertools.product(layout_order_combinations, matrix_combinations):
            layout_in, layout_out, order_in, order_out = combos1
            matrix_in = Matrix(*combos2, layout=layout_in, order=order_in)

            shape = [h, w] + list(matrix_in.shape())
            mat = rng.randint(0, 255, size=shape).astype(matrix_in.vector.scalar.numpy_dtype())

            glsl = glsl_template.format(**{
                "layout_in": self.LAYOUT_MAP[layout_in],
                "layout_out": self.LAYOUT_MAP[layout_out],
                "order_in": self.ORDERS_MAP[order_in],
                "order_out": self.ORDERS_MAP[order_out],
                "dtype": matrix_in.glsl_dtype()
            })

            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            cache_in = ByteCache(shader.get_block_definition(0))
            cache_in["dataIn"] = mat
            bytez_in = cache_in.definition.to_bytes(cache_in.get_as_dict())

            cache_out = ByteCache(shader.get_block_definition(1))
            bytez_out_count = cache_out.definition.size()
            bytez_out = self.run_compiled_program(shader, bytez_in, bytez_out_count, groups=(h, w, 1))
            cache_out.set_from_dict(cache_out.definition.from_bytes(bytez_out))

            self.assertTrue((cache_out["dataOut"] == mat).all())

    def test_pass_through_matrix(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
    
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
    
            layout({layout_in}, {order_in}, binding = 0) buffer BufferA {{
                {dtype} matrixIn;
            }};
    
            layout({layout_out}, {order_out}, binding = 1) buffer BufferB {{
                {dtype} matrixOut;
            }};
    
            void main() {{
                matrixOut = matrixIn;
            }}
            """

        rng = np.random.RandomState(123)
        matrix_combinations = itertools.product(range(2, 5), range(2, 5), [DataType.FLOAT, DataType.DOUBLE])
        layout_order_combinations = itertools.product(self.LAYOUTS, self.LAYOUTS, self.ORDERS, self.ORDERS)

        for combos1, combos2 in itertools.product(layout_order_combinations, matrix_combinations):
            layout_in, layout_out, order_in, order_out = combos1
            matrix_in = Matrix(*combos2, layout=layout_in, order=order_in)
            mat = rng.randint(0, 255, size=matrix_in.shape()).astype(matrix_in.vector.scalar.numpy_dtype())

            glsl = glsl_template.format(**{
                "layout_in": self.LAYOUT_MAP[layout_in],
                "layout_out": self.LAYOUT_MAP[layout_out],
                "order_in": self.ORDERS_MAP[order_in],
                "order_out": self.ORDERS_MAP[order_out],
                "dtype": matrix_in.glsl_dtype()
            })

            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            cache_in = ByteCache(shader.get_block_definition(0))
            cache_in["matrixIn"] = mat
            bytez_in = cache_in.definition.to_bytes(cache_in.get_as_dict())

            cache_out = ByteCache(shader.get_block_definition(1))
            bytez_out_count = cache_out.definition.size()
            bytez_out = self.run_compiled_program(shader, bytez_in, bytez_out_count)
            cache_out.set_from_dict(cache_out.definition.from_bytes(bytez_out))

            self.assertTrue((cache_out["matrixOut"] == mat).all())

    def test_pass_through_struct(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            {struct_glsl}

            layout({layout_in}, binding = 0) buffer BufferA {{
                {dtype} structIn;
            }};

            layout({layout_out}, binding = 1) buffer BufferB {{
                {dtype} structOut;
            }};

            void main() {{
                structOut = structIn;
            }}
            """

        rng = np.random.RandomState(123)

        scalars = [Scalar.of(dtype) for dtype in DataType.ALL]
        vectors = [Vector(n, dtype) for n, dtype in itertools.product(range(2, 5), DataType.ALL)]

        type_name = "SomeStruct"

        for layout_in, layout_out in itertools.product(self.LAYOUTS, self.LAYOUTS):
            members = rng.permutation(scalars + scalars + vectors + vectors)

            glsl_struct = ["struct {} {{".format(type_name)]
            for i, member in enumerate(members):
                glsl_struct.append("{} member{};".format(member.glsl_dtype(), i))
            glsl_struct.append("};")

            glsl = glsl_template.format(**{
                "layout_in": self.LAYOUT_MAP[layout_in],
                "layout_out": self.LAYOUT_MAP[layout_out],
                "struct_glsl": "\n".join(glsl_struct),
                "dtype": type_name
            })

            shader = self.shader_from_txt(glsl, verbose=False)
            shader.inspect()

            cache_in = ByteCache(shader.get_block_definition(0))

            for i, member in enumerate(members):
                if isinstance(member, Scalar):
                    value = member.numpy_dtype()(1.)
                elif isinstance(member, Vector):
                    value = np.ones(member.length(), member.scalar.numpy_dtype())
                else:
                    value = None

                cache_in["structIn"][i] = value

            bytez_in = cache_in.definition.to_bytes(cache_in.get_as_dict())

            cache_out = ByteCache(shader.get_block_definition(1))
            bytez_out_count = cache_out.definition.size()
            bytez_out = self.run_compiled_program(shader, bytez_in, bytez_out_count)
            cache_out.set_from_dict(cache_out.definition.from_bytes(bytez_out))

            for i, member in enumerate(members):
                a = cache_in["structIn"][i]
                b = cache_out["structOut"][i]
                if isinstance(member, Vector):
                    a = a.tolist()
                    b = b.tolist()
                self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
