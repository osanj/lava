# -*- coding: UTF-8 -*-

import itertools
import os
import pprint
import unittest

import numpy as np

import lava as lv
from lava.api.bytes import Array, Matrix, Vector, Scalar, Struct
from lava.api.constants.spirv import Layout
from lava.api.constants.vk import BufferUsage, MemoryType
from lava.api.memory import Buffer
from lava.api.pipeline import ShaderOperation, Pipeline
from lava.api.shader import Shader

from test.util import write_to_temp_file


class GlslBasedTest(unittest.TestCase):

    SESSION = None
    MEMORY = None

    LAYOUTS = [Layout.STD140, Layout.STD430]
    LAYOUT_MAP = {Layout.STD140: "std140", Layout.STD430: "std430"}

    @classmethod
    def setUpClass(cls):
        # lv.VALIDATION_LEVEL = lv.VALIDATION_LEVEL_DEBUG
        cls.SESSION = lv.Session(lv.devices()[0])
        cls.MEMORY = {}

    @classmethod
    def tearDownClass(cls):
        cls.SESSION.destroy()

    # Util

    @classmethod
    def shader_from_txt(cls, txt, verbose=True, clean_up=True):
        path_shader = write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = lv.compile_glsl(path_shader, verbose)
        shader = Shader.from_file(cls.SESSION.device, shader_path_spirv)
        if clean_up:
            os.remove(path_shader)
            os.remove(shader_path_spirv)
        return shader

    @classmethod
    def allocate_buffer(cls, size, usage, types):
        buf = Buffer(cls.SESSION.device, size, usage, cls.SESSION.queue_index)
        min_size, _, supported_indices = buf.get_memory_requirements()
        mem = cls.SESSION.device.allocate_memory(types, min_size, supported_indices)
        buf.bind_memory(mem)
        cls.MEMORY[buf] = (buf, mem)
        return buf

    @classmethod
    def destroy_buffer(cls, buf):
        buf, mem = cls.MEMORY[buf]
        buf.destroy()
        mem.destroy()

    @classmethod
    def run_program(cls, glsl, bytez_input, bytez_output_size, usage_input=BufferUsage.STORAGE_BUFFER,
                    usage_output=BufferUsage.STORAGE_BUFFER, groups=(1, 1, 1), verbose=True):
        shader = cls.shader_from_txt(glsl, verbose)
        return cls.run_compiled_program(shader, bytez_input, bytez_output_size, usage_input, usage_output, groups)

    @classmethod
    def run_compiled_program(cls, shader, bytez_input, bytez_output_size, usage_input=BufferUsage.STORAGE_BUFFER,
                             usage_output=BufferUsage.STORAGE_BUFFER, groups=(1, 1, 1)):
        session = cls.SESSION

        buffer_in = cls.allocate_buffer(len(bytez_input), usage_input, MemoryType.CPU)
        buffer_out = cls.allocate_buffer(bytez_output_size, usage_output, MemoryType.CPU)

        buffer_in.map(bytez_input)

        pipeline = Pipeline(session.device, shader, {0: buffer_in, 1: buffer_out})
        operation = ShaderOperation(session.device, pipeline, session.queue_index)

        operation.record(*groups)
        operation.run_and_wait()

        with buffer_out.mapped() as bytebuffer:
            bytez_output = bytebuffer[:]

        operation.destroy()
        pipeline.destroy()
        cls.destroy_buffer(buffer_in)
        cls.destroy_buffer(buffer_out)

        return bytez_output

    @classmethod
    def generate_var_name(cls, definition, index, prefix=""):
        return "{}{}{}".format(prefix, definition.__class__.__name__.lower(), index)

    @classmethod
    def build_glsl_block_definition(cls, container, binding=0, usage=BufferUsage.STORAGE_BUFFER):
        # TODO: extend for matrix order?
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

            elif isinstance(d, Matrix):
                var_name_complete = var_name or cls.generate_var_name(d, i, var_name_prefix)
                glsl_code, step = cls.build_glsl_assignments_matrix(d, j, var_name_complete, array_name, to_array=to_array)

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
        cols, rows = dfn.cols, dfn.rows
        for k, (r, c) in enumerate(itertools.product(range(rows), range(cols))):
            if to_array:
                glsl += "{}[{}] = float({}[{}][{}]);".format(array_name, i + k, var_name_complete, c, r)
            else:
                glsl += "{}[{}][{}] = {}({}[{}]);".format(var_name_complete, c, r, dfn.vector.scalar.glsl_dtype(), array_name, i + k)
            glsl += "\n"
        return glsl, cols * rows

    @classmethod
    def build_glsl_assignments_array_scalar(cls, dfn, i, var_name_complete, array_name="array", to_array=True):
        glsl = ""
        dims = dfn.shape()
        glsl_dtype = dfn.definition.glsl_dtype()

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

            elif isinstance(d, Matrix):
                rows, cols = d.shape()
                values_flat = np.arange(count, count + rows * cols, dtype=d.vector.scalar.numpy_dtype())
                values_mapped[d] = values_flat.reshape(rows, cols)
                values_raw.extend(values_flat)
                count += rows * cols

            elif isinstance(d, Array):
                if isinstance(d.definition, Scalar):
                    data = np.zeros(d.shape(), dtype=d.definition.numpy_dtype())
                    for indices in itertools.product(*[range(s) for s in d.shape()]):
                        data[indices] = count
                        values_raw.append(count)
                        count += 1
                    values_mapped[d] = data

                elif isinstance(d.definition, Vector):
                    shape = list(d.shape()) + [d.definition.length()]
                    data = np.zeros(shape, dtype=d.definition.scalar.numpy_dtype())
                    for indices in itertools.product(*[range(s) for s in shape]):
                        data[indices] = count
                        values_raw.append(count)
                        count += 1
                    values_mapped[d] = data

                elif isinstance(d.definition, Matrix):
                    shape = list(d.shape()) + list(d.definition.shape())
                    data = np.zeros(shape, dtype=d.definition.vector.scalar.numpy_dtype())
                    for indices in itertools.product(*[range(s) for s in shape]):
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

    @classmethod
    def build_register(cls, register, definition, steps):
        """Readable keys for values dict"""
        simple_types = (Scalar, Vector, Matrix)

        if isinstance(definition, Struct):
            if definition not in register:
                register[definition] = "struct{}".format(steps[Struct])
                steps[Struct] += 1

            for d in definition.definitions:
                if isinstance(d, simple_types):
                    cls.build_register_simple(register, d, steps)
                else:
                    cls.build_register(register, d, steps)

        elif isinstance(definition, Array):
            if definition not in register:
                register[definition] = "array{}".format(steps[Array])
                steps[Array] += 1

            d = definition.definition

            if isinstance(d, simple_types):
                cls.build_register_simple(register, d, steps)
            else:
                cls.build_register(register, d, steps)

        else:
            RuntimeError("Unexpected type {}".format(type(definition)))

    @classmethod
    def build_register_simple(cls, register, definition, steps):
        if isinstance(definition, Scalar) and definition not in register:
            register[definition] = "{}{}".format(definition.glsl_dtype(), steps[Scalar])
            steps[Scalar] += 1

        elif isinstance(definition, Vector) and definition not in register:
            register[definition] = "vector{}".format(steps[Vector])
            steps[Vector] += 1

        elif isinstance(definition, Matrix) and definition not in register:
            register[definition] = "matrix{}".format(steps[Matrix])
            steps[Matrix] += 1

    @classmethod
    def format_values(cls, definition, values, register):
        """Values with readable keys and no numpy arrays (allows comparison with '==' operator)"""
        simple_types = (Scalar, Vector, Matrix)

        if isinstance(definition, Struct):
            data = []

            for d in definition.definitions:
                if isinstance(d, simple_types):
                    # data[register[d]] = cls.format_values_simple(d, values[d])
                    data.append((register[d], cls.format_values_simple(d, values[d])))
                else:
                    # data[register[d]] = cls.format_values(d, values[d], register)
                    data.append((register[d], cls.format_values(d, values[d], register)))

            return data

        elif isinstance(definition, Array):
            d = definition.definition
            data = np.zeros(definition.shape()).tolist()

            for indices in itertools.product(*[range(s) for s in definition.shape()]):
                _tmp1 = values
                _tmp2 = data

                for index in indices[:-1]:
                    _tmp1 = _tmp1[index]
                    _tmp2 = _tmp2[index]

                if isinstance(d, simple_types):
                    _tmp2[indices[-1]] = cls.format_values_simple(d, _tmp1[indices[-1]])
                else:
                    _tmp2[indices[-1]] = cls.format_values(d, _tmp1[indices[-1]], register)

            return data

        else:
            RuntimeError("Unexpected type {}".format(type(definition)))

    @classmethod
    def format_values_simple(cls, definition, value):
        if isinstance(definition, Scalar):
            return value
        if isinstance(definition, Vector):
            return value.tolist()
        if isinstance(definition, Matrix):
            return value.tolist()

    @classmethod
    def print_formatted_values(cls, values_ftd):
        pprint.pprint(values_ftd, width=200)


class Random(object):

    @classmethod
    def shape(cls, rng, max_dims, max_dim, min_dims=1, min_dim=1):
        dims = rng.randint(min_dims, max_dims + 1)
        return tuple(rng.randint(min_dim, max_dim, dims))

