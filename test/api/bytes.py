# -*- coding: UTF-8 -*-

import logging
import unittest

from lava.api.constants import BufferUsage, MemoryType
from lava.api.bytes import *
from lava.api.memory import Buffer
from lava.api.pipeline import Executor, Pipeline
from lava.api.shader import compile_glsl, Shader
from lava.session import Session

from test import TestUtil
#from test import TestSession

logger = logging.getLogger(__name__)


class TestByteRepresentation(unittest.TestCase):

    SESSION = None
    MEMORY = None

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.DEBUG)
        TestUtil.set_vulkan_environment_variables()
        cls.SESSION = Session.discover()  # TestSession() ?
        cls.MEMORY = {}

    @classmethod
    def tearDownClass(cls):
        del cls.SESSION

    # Util

    def shader_from_txt(self, txt):
        path_shader = TestUtil.write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = compile_glsl(path_shader)
        return Shader(self.SESSION.device, shader_path_spirv)

    def allocate_buffer(self, size, usage, types):
        buf = Buffer(self.SESSION.device, size, usage, self.SESSION.queue_index)
        mem = self.SESSION.device.allocate_memory(types, buf.get_memory_requirements()[0])
        buf.bind_memory(mem)
        self.MEMORY[buf] = (buf, mem)
        return buf

    def destroy_buffer(self, buf):
        buf, mem = self.MEMORY[buf]
        del buf
        del mem

    # Tests

    def test_scalar_to_array(self):
        def glsl(dtype):
            return """\
                    #version 450
                    #extension GL_ARB_separate_shader_objects : enable

                    layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
                    layout(std140, binding = 0) uniform scalarIn {{ {} value; }};
                    layout(std430, binding = 1) writeonly buffer arrayOut {{ {} arr[]; }}; // stdout430 so we have no array alignment fiddling

                    void main() {{
                        uint index = gl_GlobalInvocationID.x;
                        arr[index] = value;
                    }}
                    """.format(dtype, dtype)

        test_data = (
            (Scalar.of(Scalar.INT), -12345, "int"),
            (Scalar.of(Scalar.UINT), 12345, "uint"),
            (Scalar.of(Scalar.FLOAT), 3.14, "float"),
            (Scalar.of(Scalar.DOUBLE), 1e+45, "double")
        )

        session = self.SESSION
        array_length = 8

        for scalar, value, shader_dtype in test_data:
            numpy_dtype = scalar.numpy_dtype()
            shader = self.shader_from_txt(glsl(shader_dtype))

            x1 = value
            x1_size = scalar.size_aligned()
            x2 = np.zeros(array_length, dtype=numpy_dtype)
            x2_size = x2.nbytes

            # memory setup
            buffer_in = self.allocate_buffer(x1_size, BufferUsage.UNIFORM_BUFFER, MemoryType.CPU)
            buffer_out = self.allocate_buffer(x2_size, BufferUsage.STORAGE_BUFFER, MemoryType.CPU)

            # map value into the buffer
            with buffer_in.mapped() as bytebuffer:
                bytebuffer[:] = scalar.convert_data_to_aligned_bytes(x1)

            # setup pipeline
            pipeline = Pipeline(session.device, shader, [buffer_in, buffer_out])
            executor = Executor(session.device, pipeline, session.queue_index)

            executor.record(array_length, 1, 1)
            executor.execute_and_wait()

            with buffer_out.mapped() as bytebuffer:
                y = np.frombuffer(bytebuffer[:], dtype=numpy_dtype).copy()

            # clean up
            del executor, pipeline
            self.destroy_buffer(buffer_in)
            self.destroy_buffer(buffer_out)

            y_expected = np.array([value] * array_length, dtype=numpy_dtype)
            logger.debug("\nvar like {}\nexpected {}\ngot {}".format(scalar.glsl(), y_expected, y))
            self.assertTrue((y == y_expected).all())

    def test_vector_to_array(self):
        def glsl(dtype_vec, dtype_array, size_vec):
            return """\
                    #version 450
                    #extension GL_ARB_separate_shader_objects : enable

                    layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
                    layout(std140, binding = 0) uniform vecIn {{ {} vec; }};
                    layout(std430, binding = 1) writeonly buffer arrayOut {{ {} arr[]; }};  // stdout430 so we have no array alignment fiddling

                    void main() {{
                        uint index = gl_GlobalInvocationID.x;
                        uint n = {};
                        for(int i = 0; i < n; i++) {{ arr[index * n + i] = vec[i]; }}
                    }}
                    """.format(dtype_vec, dtype_array, size_vec)

        max_int = 0x7FFFFFFF
        max_float = 1e+40
        test_data = (
            (Vector(n=2, dtype=Scalar.INT), (-1, 2), ("ivec2", "int")),
            (Vector(n=3, dtype=Scalar.INT), (-1, 2, -3), ("ivec3", "int")),
            (Vector(n=4, dtype=Scalar.INT), (-1, 2, -3, 4), ("ivec4", "int")),
            (Vector(n=2, dtype=Scalar.UINT), (max_int, max_int + 1), ("uvec2", "uint")),
            (Vector(n=3, dtype=Scalar.UINT), (max_int, max_int + 1, max_int + 2), ("uvec3", "uint")),
            (Vector(n=4, dtype=Scalar.UINT), (max_int, max_int + 1, max_int + 2, max_int + 3), ("uvec4", "uint")),
            (Vector(n=2, dtype=Scalar.FLOAT), (np.pi, np.pi + 1), ("vec2", "float")),
            (Vector(n=3, dtype=Scalar.FLOAT), (np.pi, np.pi + 1, np.pi + 2), ("vec3", "float")),
            (Vector(n=4, dtype=Scalar.FLOAT), (np.pi, np.pi + 1, np.pi + 2, np.pi + 3), ("vec4", "float")),
            (Vector(n=2, dtype=Scalar.DOUBLE), (max_float, -1.), ("dvec2", "double")),
            (Vector(n=3, dtype=Scalar.DOUBLE), (max_float, -1., max_float + 2), ("dvec3", "double")),
            (Vector(n=4, dtype=Scalar.DOUBLE), (max_float, -1., max_float + 2, max_float + 3), ("dvec4", "double")),
        )

        session = self.SESSION
        array_length = 8

        for vector, values, shader_dtypes in test_data:
            assert vector.length() == len(values)

            numpy_dtype = vector.scalar.numpy_dtype()
            shader = self.shader_from_txt(glsl(*shader_dtypes, size_vec=len(values)))

            x1 = values
            x1_size = vector.size_aligned()
            x2 = np.zeros(array_length * vector.length(), dtype=numpy_dtype)
            x2_size = x2.nbytes

            # memory setup
            buffer_in = self.allocate_buffer(x1_size, BufferUsage.UNIFORM_BUFFER, MemoryType.CPU)
            buffer_out = self.allocate_buffer(x2_size, BufferUsage.STORAGE_BUFFER, MemoryType.CPU)

            # map value into the buffer
            with buffer_in.mapped() as bytebuffer:
                bytebuffer[:] = vector.convert_data_to_aligned_bytes(x1)

            # setup pipeline
            pipeline = Pipeline(session.device, shader, [buffer_in, buffer_out])
            executor = Executor(session.device, pipeline, session.queue_index)

            executor.record(array_length, 1, 1)
            executor.execute_and_wait()

            with buffer_out.mapped() as bytebuffer:
                y = np.frombuffer(bytebuffer[:], dtype=numpy_dtype).copy()

            # clean up
            del executor, pipeline
            self.destroy_buffer(buffer_in)
            self.destroy_buffer(buffer_out)

            y_expected = np.array(values * array_length, dtype=numpy_dtype)
            logger.debug("\nvar like {}\nexpected {}\ngot {}".format(vector.glsl(), y_expected, y))
            self.assertTrue((y == y_expected).all())

