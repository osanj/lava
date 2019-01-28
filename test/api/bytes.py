# -*- coding: UTF-8 -*-

import unittest

from lava.api.constants import BufferUsage, MemoryType
from lava.api.bytes import *
from lava.api.memory import Buffer
from lava.api.pipeline import Executor, Pipeline
from lava.api.shader import compile_glsl, Shader
from lava.session import Session

from test import TestUtil
#from test import TestSession


class TestByteRepresentation(unittest.TestCase):

    SESSION = None

    @classmethod
    def setUpClass(cls):
        TestUtil.set_vulkan_environment_variables()
        #cls.SESSION = TestSession()
        cls.SESSION = Session.discover()

    @classmethod
    def tearDownClass(cls):
        del cls.SESSION

    @staticmethod
    def shader_scalars(dtype):
        return """\
        #version 450
        #extension GL_ARB_separate_shader_objects : enable
        
        layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        
        layout(std140, binding = 0) uniform scalarIn
        {{
            {} value;
        }};
        
        layout(std430, binding = 1) writeonly buffer bufOut  // stdout430 so we have no array alignment fiddling
        {{
            {} bufferOut[];
        }};
        
        void main() {{
            uint index = gl_GlobalInvocationID.x;
            bufferOut[index] = value;
        }}
        """.format(dtype, dtype)

    def test_scalars(self):
        test_data = (
            (Scalar.of(Scalar.INT), -12345, self.shader_scalars("int"), np.int32),
            (Scalar.of(Scalar.UINT), 12345, self.shader_scalars("uint"), np.uint32),
            (Scalar.of(Scalar.FLOAT), 3.14, self.shader_scalars("float"), np.float32),
            (Scalar.of(Scalar.DOUBLE), 1e+45, self.shader_scalars("double"), np.float64)
        )

        session = self.SESSION
        array_length = 32

        for scalar, value, shader_txt, numpy_dtype in test_data:
            x1 = value
            x1_size = scalar.size_aligned()
            x2 = np.zeros(array_length, dtype=numpy_dtype)
            x2_size = x2.nbytes

            # dummy shader
            path_shader = TestUtil.write_to_temp_file(shader_txt, suffix=".comp")
            shader_path_spirv = compile_glsl(path_shader)
            shader = Shader(session.device, shader_path_spirv)

            # memory setup
            buffer_in = Buffer(session.device, x1_size, BufferUsage.UNIFORM_BUFFER, session.queue_index)
            memory_in = session.device.allocate_memory(MemoryType.CPU, buffer_in.get_memory_requirements()[0])
            buffer_in.bind_memory(memory_in)

            buffer_out = Buffer(session.device, x2_size, BufferUsage.STORAGE_BUFFER, session.queue_index)
            memory_out = session.device.allocate_memory(MemoryType.CPU, buffer_out.get_memory_requirements()[0])
            buffer_out.bind_memory(memory_out)

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
            del executor, pipeline, memory_in, memory_out, buffer_in, buffer_out

            y_expected = np.array([value] * array_length, dtype=numpy_dtype)
            self.assertTrue((y == y_expected).all())


