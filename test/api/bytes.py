# -*- coding: UTF-8 -*-

import itertools
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

    @classmethod
    def shader_from_txt(cls, txt):
        path_shader = TestUtil.write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = compile_glsl(path_shader, verbose=True)
        return Shader(cls.SESSION.device, shader_path_spirv)

    @classmethod
    def allocate_buffer(cls, size, usage, types):
        buf = Buffer(cls.SESSION.device, size, usage, cls.SESSION.queue_index)
        mem = cls.SESSION.device.allocate_memory(types, buf.get_memory_requirements()[0])
        buf.bind_memory(mem)
        cls.MEMORY[buf] = (buf, mem)
        return buf

    @classmethod
    def destroy_buffer(cls, buf):
        buf, mem = cls.MEMORY[buf]
        del buf
        del mem


class TestByteRepresentationBasicToArray(TestByteRepresentation):

    @classmethod
    def build_glsl_program(cls, container, buffer_usage):
        return """
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

{}

layout(std430, binding = 1) writeonly buffer dataOut {{
    float array[];
}}; // stdout430 so we have no array alignment fiddling

void main() {{
{}
}}""".format(cls.build_glsl_definitions(container, usage=buffer_usage), cls.build_glsl_assignemnts(container))

    @classmethod
    def build_glsl_definitions(cls, container, binding=0, usage=BufferUsage.STORAGE_BUFFER, var_name="var"):
        glsl = "layout({}, binding = {}) {} dataIn {{".format(
            "std140" if container.layout == ByteRepresentation.LAYOUT_STD140 else "std430",
            binding,
            "readonly buffer" if usage == BufferUsage.STORAGE_BUFFER else "uniform",
        )
        glsl += "\n"

        for i, d in enumerate(container.definitions):
            glsl += "{} {}{};".format(d.glsl_dtype(), var_name, i)
            glsl += "\n"

        glsl += "};"
        return glsl

    @classmethod
    def build_glsl_assignemnts(cls, container, var_name="var", array_name="array"):
        glsl = ""
        j = 0

        for i, d in enumerate(container.definitions):
            var_name_complete = "{}{}".format(var_name, i)

            if isinstance(d, Scalar):
                glsl_line, step = cls.build_glsl_assignemnts_scalar(j, var_name_complete, array_name)
            elif isinstance(d, Vector):
                glsl_line, step = cls.build_glsl_assignments_vector(j, d.length(), var_name_complete, array_name)
            elif isinstance(d, Matrix):
                glsl_line, step = cls.build_glsl_assignments_matrix(j, d.n, d.m, var_name_complete, array_name)
            else:
                raise RuntimeError()

            glsl += glsl_line
            glsl += "\n"
            j += step

        return glsl

    @classmethod
    def build_glsl_assignemnts_scalar(cls, i, var_name_complete, array_name="array"):
        glsl = "{}[{}] = float({});".format(array_name, i, var_name_complete)
        glsl += "\n"
        return glsl, 1

    @classmethod
    def build_glsl_assignments_vector(cls, i, n, var_name_complete, array_name="array"):
        glsl = ""
        for j in range(n):
            glsl += "{}[{}] = float({}[{}]);".format(array_name, i + j, var_name_complete, j)
            glsl += "\n"
        return glsl, n

    @classmethod
    def build_glsl_assignments_matrix(cls, i, cols, rows, var_name_complete, array_name="array"):
        glsl = ""
        for r, c in itertools.product(range(cols), range(rows)):
            glsl += "{}[{}] = float({}[{}][{}]);".format(array_name, i + r * cols + c, var_name_complete, c, r)
            glsl += "\n"
        return glsl, cols * rows

    @classmethod
    def build_input_values(cls, container):
        count = 0
        values_raw = []
        values_mapped = {}

        for i, d in enumerate(container.definitions):
            if isinstance(d, Scalar):
                values_mapped[d] = d.numpy_dtype()(count)
                values_raw.append(values_mapped[d])
                count += 1
            elif isinstance(d, Vector):
                values_mapped[d] = np.arange(count, count + d.length(), dtype=d.scalar.numpy_dtype())
                values_raw.extend(values_mapped[d])
                count += d.length()
            elif isinstance(d, Matrix):
                rows, cols = d.shape()
                values_mapped[d] = np.arange(count, count + rows * cols, dtype=d.scalar.numpy_dtype())
                values_raw.extend(values_mapped[d])
                count += rows * cols
            else:
                raise RuntimeError()

        return values_mapped, np.array(values_raw, dtype=np.float32)

    @classmethod
    def run_program(cls, glsl, bytez_input, expected_output, usage=BufferUsage.STORAGE_BUFFER):
        session = cls.SESSION
        shader = cls.shader_from_txt(glsl)

        buffer_in = cls.allocate_buffer(len(bytez_input), usage, MemoryType.CPU)
        buffer_out = cls.allocate_buffer(expected_output.nbytes, usage, MemoryType.CPU)

        buffer_in.map(bytez_input)

        pipeline = Pipeline(session.device, shader, [buffer_in, buffer_out])
        executor = Executor(session.device, pipeline, session.queue_index)

        executor.record(1, 1, 1)
        executor.execute_and_wait()

        with buffer_out.mapped() as bytebuffer:
            y = np.frombuffer(bytebuffer[:], dtype=expected_output.dtype).copy()

        return y

    def test4(self):
        buffer_usage = BufferUsage.STORAGE_BUFFER
        buffer_layout = Container.LAYOUT_STD140
        buffer_order = Container.ORDER_ROW_MAJOR

        order = [
            Vector.vec3(),
            Scalar.float(),
            Vector.ivec2(),
            Scalar.float(),
            Vector.dvec4(),
            Scalar.uint(),
            Scalar.uint(),
            Vector.vec3(),
            Scalar.uint(),
            Vector.vec3(),
            Vector.vec3(),
        ]

        container = Container(buffer_layout, buffer_order, *order)

        glsl = self.build_glsl_program(container, buffer_usage)
        values, expected = self.build_input_values(container)

        output = self.run_program(glsl, container.to_bytes(values), expected, buffer_usage)

        print expected
        print output


    def test3(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
    
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
    
            layout(std140, binding = 0) readonly buffer dataIn {
                float value1;
                float value2;
                vec3 vector;
            };
            layout(std430, binding = 1) writeonly buffer dataOut {
                float array[];
            }; // stdout430 so we have no array alignment fiddling
    
            void main() {
                // uint index = gl_GlobalInvocationID.x;
                array[0] = value1;
                array[1] = value2;
                array[2] = vector.x;
                array[3] = vector.y;
                array[4] = vector.z;
            }
            """


        expected = np.arange(1000, dtype=np.float32)

        vector1 = Vector(n=3, dtype=Scalar.FLOAT)
        value1 = Scalar.float()
        value2 = Scalar.float()

        order = [value1, value2, vector1]

        container = Container(Container.LAYOUT_STD140, *order)
        values = {}
        index = 0
        for d in order:
            if isinstance(d, Scalar):
                values[d] = expected[index]
                index += 1
            elif isinstance(d, Vector):
                values[d] = expected[index:index + d.length()]
                index += d.length()
            else:
                raise NotImplementedError()

            #print d.glsl(), values[d], values[d].dtype

        expected = expected[:index]
        bytez = container.to_bytes(values)
        print len(bytez)

        # do the stuff
        session = self.SESSION
        shader = self.shader_from_txt(glsl)

        buffer_in = self.allocate_buffer(len(bytez), BufferUsage.STORAGE_BUFFER, MemoryType.CPU)
        buffer_out = self.allocate_buffer(expected.nbytes, BufferUsage.STORAGE_BUFFER, MemoryType.CPU)

        buffer_in.map(bytez)

        pipeline = Pipeline(session.device, shader, [buffer_in, buffer_out])
        executor = Executor(session.device, pipeline, session.queue_index)

        executor.record(1, 1, 1)
        executor.execute_and_wait()

        with buffer_out.mapped() as bytebuffer:
            y = np.frombuffer(bytebuffer[:], dtype=expected.dtype).copy()

        # clean up
        del executor, pipeline
        self.destroy_buffer(buffer_in)
        self.destroy_buffer(buffer_out)

        print expected
        print y
