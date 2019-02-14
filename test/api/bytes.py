# -*- coding: UTF-8 -*-

import itertools
import logging
import unittest

from lava.api.bytes import *
from lava.api.constants.spirv import Layout, Order
from lava.api.constants.vk import BufferUsage, MemoryType
from lava.api.memory import Buffer
from lava.api.pipeline import Executor, Pipeline
from lava.api.shader import Shader
from lava.session import Session
from lava.util import compile_glsl

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

    @classmethod
    def run_program(cls, glsl, bytez_input, bytez_output_size, usage_input=BufferUsage.STORAGE_BUFFER,
                    usage_output=BufferUsage.STORAGE_BUFFER):
        session = cls.SESSION
        shader = cls.shader_from_txt(glsl)

        buffer_in = cls.allocate_buffer(len(bytez_input), usage_input, MemoryType.CPU)
        buffer_out = cls.allocate_buffer(bytez_output_size, usage_output, MemoryType.CPU)

        buffer_in.map(bytez_input)

        pipeline = Pipeline(session.device, shader, [buffer_in, buffer_out])
        executor = Executor(session.device, pipeline, session.queue_index)

        executor.record(1, 1, 1)
        executor.execute_and_wait()

        # with buffer_out.mapped() as bytebuffer:
        #     y = np.frombuffer(bytebuffer[:], dtype=expected_output.dtype).copy()
        #
        # return y

        with buffer_out.mapped() as bytebuffer:
            bytez_output = bytebuffer[:]

        return bytez_output

    @classmethod
    def generate_var_name(cls, definition, index, prefix=""):
        return "{}{}{}".format(prefix, definition.__class__.__name__.lower(), index)

    @classmethod
    def build_glsl_block_definition(cls, container, binding=0, usage=BufferUsage.STORAGE_BUFFER):
        glsl = "layout({}, binding = {}) {} dataIn {{".format(
            "std140" if container.layout == Layout.STD140 else "std430", binding,
            "buffer" if usage == BufferUsage.STORAGE_BUFFER else "uniform", )
        glsl += "\n"

        for i, d in enumerate(container.definitions):
            glsl += "  {} {};".format(d.glsl_dtype(), cls.generate_var_name(d, i))
            glsl += "\n"

        return glsl + "};"

    @classmethod
    def build_glsl_struct_definition(cls, struct):
        glsl = "struct {} {{".format(struct.glsl_dtype())
        glsl += "\n"

        for i, d in enumerate(struct.definitions):
            glsl += "  {} {};".format(d.glsl_dtype(), cls.generate_var_name(d, i))
            glsl += "\n"

        return glsl + "};"

    @classmethod
    def build_glsl_assignments(cls, definitions, var_name=None, array_name="array", array_index=0, parent=None, var_name_prefix="", to_array=True):
        glsl = ""
        j = array_index

        for i, d in enumerate(definitions):
            if isinstance(d, Scalar):
                var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix)
                glsl_code, step = cls.build_glsl_assignments_scalar(d, j, var_name_complete, array_name, to_array=to_array)

            elif isinstance(d, Vector):
                var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix)
                glsl_code, step = cls.build_glsl_assignments_vector(d, j, var_name_complete, array_name, to_array=to_array)

            # elif isinstance(d, Matrix):
            #     var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix)
            #     glsl_code, step = cls.build_glsl_assignments_matrix(d, j, var_name_complete, array_name, to_array=to_array)

            elif isinstance(d, Array):
                var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix)
                if isinstance(d.definition, Scalar):
                    glsl_code, step = cls.build_glsl_assignments_array_scalar(d, j, var_name_complete, array_name, to_array=to_array)
                else:
                    glsl_code, step = cls.build_glsl_assignments_array_complex(d, j, var_name_complete, array_name, to_array=to_array)

            elif isinstance(d, Struct):
                if isinstance(parent, Array):
                    var_name_complete = var_name + "."
                    glsl_code, step_overall = cls.build_glsl_assignments(d.definitions, var_name=None, var_name_prefix=var_name_complete,
                                                                         array_name=array_name, array_index=j, parent=d, to_array=to_array)
                    step = step_overall - j
                else:
                    var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix) + "."
                    glsl_code, step_overall = cls.build_glsl_assignments(d.definitions, var_name=None, var_name_prefix=var_name_complete,
                                                                         array_name=array_name, array_index=j, parent=d, to_array=to_array)
                    step = step_overall - j

            else:
                raise RuntimeError()

            glsl += glsl_code
            j += step

        return glsl, j

    @classmethod
    def build_glsl_assignments_scalar(cls, dfn, i, var_name_complete, array_name="array", to_array=True):
        if to_array:
            glsl = "{}[{}] = float({});".format(array_name, i, var_name_complete)
        else:
            glsl = "{} = {}({}[{}]);".format(var_name_complete, dfn.glsl_dtype(), array_name, i)
        glsl += "\n"
        return glsl, 1

    @classmethod
    def build_glsl_assignments_vector(cls, dfn, i, var_name_complete, array_name="array", to_array=True):
        glsl = ""
        n = dfn.length()
        for j in range(n):
            if to_array:
                glsl += "{}[{}] = float({}.{});".format(array_name, i + j, var_name_complete, "xyzw"[j])
            else:
                glsl += "{}.{} = {}({}[{}]);".format(var_name_complete, "xyzw"[j], dfn.scalar.glsl_dtype(), array_name, i + j)
            glsl += "\n"
        return glsl, n

    @classmethod
    def build_glsl_assignments_matrix(cls, dfn, i, var_name_complete, array_name="array", to_array=True):
        glsl = ""
        cols, rows = dfn.n, dfn.m
        for k, r, c in enumerate(itertools.product(range(cols), range(rows))):
            if to_array:
                glsl += "{}[{}] = float({}[{}][{}]);".format(array_name, i + k, var_name_complete, c, r)
            else:
                glsl += "{}[{}][{}] = {}({}[{}]);".format(var_name_complete, c, r, dfn.scalar.glsl_dtype(), array_name, i + k)
            glsl += "\n"
        return glsl, cols * rows

    @classmethod
    def build_glsl_assignments_array_scalar(cls, dfn, i, var_name_complete, array_name="array", to_array=True):
        glsl = ""
        dims = dfn.shape()
        glsl_dtype = None

        if isinstance(dfn.definition, Scalar):
            glsl_dtype = dfn.definition.glsl_dtype()
        if isinstance(dfn.definition, Vector):
            glsl_dtype = dfn.definition.scalar.glsl_dtype()
        # if isinstance(dfn.definition, Matrix):
        #     glsl_dtype = dfn.definition.scalar.glsl_dtype()

        for k, indices in enumerate(itertools.product(*[range(d) for d in dims])):
            var_name_complete_with_indices = ("{}" + "[{}]" * len(dims)).format(var_name_complete, *indices)
            if to_array:
                glsl += "{}[{}] = float({});".format(array_name, i + k, var_name_complete_with_indices)
            else:
                glsl += "{} = {}({}[{}]);".format(var_name_complete_with_indices, glsl_dtype, array_name, i + k)
            glsl += "\n"
        return glsl, np.product(dims)

    @classmethod
    def build_glsl_assignments_array_complex(cls, array, i, var_name_complete, array_name="array", to_array=True):
        glsl = ""
        old_i = i
        for indices in itertools.product(*[range(d) for d in array.dims]):
            new_var_name_complete = ("{}" + "[{}]" * len(array.dims)).format(var_name_complete, *indices)
            new_glsl, new_i = cls.build_glsl_assignments([array.definition], new_var_name_complete, array_name=array_name, array_index=i, parent=array, to_array=to_array)
            glsl += new_glsl
            i = new_i
        return glsl, i - old_i

    @classmethod
    def build_values(cls, definitions, offset=0):
        count = offset
        values_raw = []
        values_mapped = {}

        for i, d in enumerate(definitions):
            if isinstance(d, Scalar):
                values_mapped[d] = d.numpy_dtype()(count)
                values_raw.append(values_mapped[d])
                count += 1

            elif isinstance(d, Vector):
                values_mapped[d] = np.arange(count, count + d.length(), dtype=d.scalar.numpy_dtype())
                values_raw.extend(values_mapped[d])
                count += d.length()

            # elif isinstance(d, Matrix):
            #     rows, cols = d.shape()
            #     values_mapped[d] = np.arange(count, count + rows * cols, dtype=d.scalar.numpy_dtype())
            #     values_raw.extend(values_mapped[d])
            #     count += rows * cols

            elif isinstance(d, Array):
                if isinstance(d.definition, Scalar):
                    data = np.zeros(d.shape(), dtype=d.definition.numpy_dtype())
                    for indices in itertools.product(*[range(s) for s in d.shape()]):
                        data[indices] = count
                        values_raw.append(count)
                        count += 1
                    values_mapped[d] = data

                else:
                    data = np.zeros(d.shape()).tolist()
                    for indices in itertools.product(*[range(s) for s in d.shape()]):
                        tmp1, tmp2 = cls.build_values([d.definition], offset=count)
                        _data = data
                        for index in indices[:-1]:
                            _data = _data[index]
                        _data[indices[-1]] = tmp1[d.definition]
                        values_raw.extend(tmp2)
                        count += len(tmp2)
                    values_mapped[d] = data

            elif isinstance(d, Struct):
                tmp1, tmp2 = cls.build_values(d.definitions, offset=count)
                values_mapped[d] = tmp1
                values_raw.extend(tmp2)
                count += len(tmp2)
            else:
                raise RuntimeError()

        return values_mapped, values_raw


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

        values, array_expected = self.build_values(container.definitions)
        array_expected = np.array(array_expected, dtype=np.float32)

        bytez_input = container.to_bytes(values)
        bytez_output = self.run_program(glsl, bytez_input, array_expected.nbytes, usage_input=buffer_usage)
        array = np.frombuffer(bytez_output, dtype=array_expected.dtype)

        print array_expected
        print array
        print "equal", ((array_expected - array) == 0).all()

    def test_manually2(self):
        # buffer padding test
        buffer_usage = BufferUsage.STORAGE_BUFFER
        buffer_layout = Layout.STD430
        buffer_order = Order.ROW_MAJOR

        struct1 = Struct([Vector.vec3(), Vector.ivec2()], buffer_layout, type_name="structB")
        struct2 = Struct([Scalar.double(), Scalar.double(), struct1], buffer_layout, type_name="structC")

        structs = [struct1, struct2]

        variables = [
            Scalar.uint(),
            Array(Vector.vec2(), (5, 2, 3), buffer_layout),
            Array(Scalar.float(), 5, buffer_layout),
            struct2,  # this struct needs padding at the end
            Scalar.uint(),
            Array(struct1, 2, buffer_layout)
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

    def test_manually3(self):
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

        import pprint

        print ""
        print ""
        pprint.pprint(cache.values)
        print cache[-1][0][0]["a"]
        print ""
        print ""
        pprint.pprint(cache)
        print cache[-1][0][0]
        print ""
        print ""
        pprint.pprint(cache.get_as_dict())


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

        print ""
        print values
        print ""
        print values_expected

        # output = self.run_program(glsl, container.to_bytes(values), array, usage_input=buffer_usage)
        #
        # print array
        # print output
        # print "equal", ((array - output) == 0).all()

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
