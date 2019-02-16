# -*- coding: UTF-8 -*-

import itertools
import logging

import numpy as np

from lava.api.constants.spirv import DataType, Layout

logger = logging.getLogger(__name__)


class ByteRepresentation(object):

    """Data structure implementation of Vulkan specification 14.5.4."""

    # References
    # https://www.khronos.org/registry/vulkan/specs/1.1/html/chap14.html#interfaces-resources-layout
    # https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt
    #   search document for "Sub-section 2.15.3.1.2"
    #   search document for "The following example illustrates the rules specified by the "std140" layout"

    def __init__(self):
        pass

    def size(self):
        raise NotImplementedError()

    def alignment(self):
        # called 'base alignment' in the specs
        raise NotImplementedError()

    def __str__(self, indent=2):
        raise NotImplementedError()

    def glsl(self, var_name):
        return "{} {};".format(self.glsl_dtype(), var_name)

    def glsl_dtype(self):
        raise NotImplementedError()

    @classmethod
    def compare_type(cls, type_expected, type_other, path, quiet=True):
        if type_expected != type_other:
            if not quiet:
                raise RuntimeError(
                    "Expected type {}, but got type {} at {}".format(type_expected, type_other, ">".join(path)))
            return False

        return True

    @classmethod
    def compare_layout(cls, layout_expected, layout_other, path, quiet=True):
        if layout_expected == Layout.STDXXX:
            return True

        if layout_expected != layout_other and layout_other != Layout.STDXXX:
            if not quiet:
                raise RuntimeError(
                    "Expected layout {}, but got layout {} at {}".format(layout_expected, layout_other, " > ".join(path)))
            return False

        return True

    def compare(self, other, path, quiet=True):
        raise NotImplementedError()

    def to_bytes(self, values):
        raise NotImplementedError()

    def from_bytes(self, bytez):
        raise NotImplementedError()


class Scalar(ByteRepresentation):

    def __init__(self, input_dtypes):
        super(Scalar, self).__init__()
        self.input_dtypes = input_dtypes

    @classmethod
    def int(cls):
        return cls.of(DataType.INT)

    @classmethod
    def uint(cls):
        return cls.of(DataType.UINT)

    @classmethod
    def float(cls):
        return cls.of(DataType.FLOAT)

    @classmethod
    def double(cls):
        return cls.of(DataType.DOUBLE)

    @classmethod
    def of(cls, dtype):
        if dtype == DataType.INT:
            return ScalarInt()
        if dtype == DataType.UINT:
            return ScalarUnsignedInt()
        if dtype == DataType.FLOAT:
            return ScalarFloat()
        if dtype == DataType.DOUBLE:
            return ScalarDouble()
        raise RuntimeError("Unknown scalar type '{}'".format(dtype))

    def alignment(self):
        # "A scalar of size N has a base alignment of N."
        return self.size()

    def __str__(self, name=None, indent=2):
        return "{} [{}]".format(self.glsl_dtype(), name or "?")

    def glsl_dtype(self):
        raise NotImplementedError()

    def compare(self, other, path=(), quiet=True):
        return self.compare_type(type(self), type(other), path, quiet)

    def to_bytes(self, value):
        if not isinstance(value, self.input_dtypes):
            raise RuntimeError("{} got dtype {} (expects one of the following {})".format(
                self.__class__.__name__, type(value), self.input_dtypes
            ))

    def from_bytes(self, bytez):
        return np.frombuffer(bytez, self.numpy_dtype())[0]

    def numpy_dtype(self):
        raise NotImplementedError()


class ScalarInt(Scalar):

    def __init__(self):
        super(ScalarInt, self).__init__((int, np.int32))

    def numpy_dtype(self):
        return np.int32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "int"

    def to_bytes(self, value):
        super(ScalarInt, self).to_bytes(value)
        if type(value) is int:
            if not (-(0x7FFFFFFF + 1) <= value <= 0x7FFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.int32(value)
        return bytearray(value.tobytes())


class ScalarUnsignedInt(Scalar):

    def __init__(self):
        super(ScalarUnsignedInt, self).__init__((int, np.uint32))

    def numpy_dtype(self):
        return np.uint32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "uint"

    def to_bytes(self, value):
        super(ScalarUnsignedInt, self).to_bytes(value)
        if type(value) is int:
            if not (0 <= value <= 0xFFFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.uint32(value)
        return bytearray(value.tobytes())


class ScalarFloat(Scalar):

    def __init__(self):
        super(ScalarFloat, self).__init__((float, np.float32))

    def numpy_dtype(self):
        return np.float32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "float"

    def to_bytes(self, value):
        super(ScalarFloat, self).to_bytes(value)
        if type(value) is float:
            # TODO: add range check
            value = np.float32(value)
        return bytearray(value.tobytes())


class ScalarDouble(Scalar):

    def __init__(self):
        super(ScalarDouble, self).__init__((float, np.float64))

    def numpy_dtype(self):
        return np.float64

    def size(self):
        return 8

    def glsl_dtype(self):
        return "double"

    def to_bytes(self, value):
        super(ScalarDouble, self).to_bytes(value)
        if type(value) is float:
            # TODO: add range check
            value = np.float64(value)
        return bytearray(value.tobytes())


class Vector(ByteRepresentation):

    def __init__(self, n=4, dtype=DataType.FLOAT):
        super(Vector, self).__init__()
        self.dtype = dtype
        self.n = n
        self.scalar = Scalar.of(dtype)

    @classmethod
    def ivec2(cls):
        return Vector(2, DataType.INT)

    @classmethod
    def ivec3(cls):
        return Vector(3, DataType.INT)

    @classmethod
    def ivec4(cls):
        return Vector(4, DataType.INT)

    @classmethod
    def uvec2(cls):
        return Vector(2, DataType.UINT)

    @classmethod
    def uvec3(cls):
        return Vector(3, DataType.UINT)

    @classmethod
    def uvec4(cls):
        return Vector(4, DataType.UINT)

    @classmethod
    def vec2(cls):
        return Vector(2, DataType.FLOAT)

    @classmethod
    def vec3(cls):
        return Vector(3, DataType.FLOAT)

    @classmethod
    def vec4(cls):
        return Vector(4, DataType.FLOAT)

    @classmethod
    def dvec2(cls):
        return Vector(2, DataType.DOUBLE)

    @classmethod
    def dvec3(cls):
        return Vector(3, DataType.DOUBLE)

    @classmethod
    def dvec4(cls):
        return Vector(4, DataType.DOUBLE)

    def size(self):
        return self.scalar.size() * self.n

    def length(self):
        return self.n

    def alignment(self):
        if self.n == 2:
            # "A two-component vector, with components of size N, has a base alignment of 2 N."
            return self.scalar.size() * 2
        if self.n in (3, 4):
            # "A three- or four-component vector, with components of size N, has a base alignment of 4 N."
            return self.scalar.size() * 4
        return -1

    def __str__(self, name=None, indent=2):
        return "{} [{}]".format(self.glsl_dtype(), name or "?")

    def glsl_dtype(self):
        return "{}vec{}".format(self.dtype.lower()[0] if self.dtype is not DataType.FLOAT else "", self.n)

    def compare(self, other, path=(), quiet=True):
        if not self.compare_type(type(self), type(other), path, quiet):
            return False

        count_expected = self.n
        count_other = other.n

        if count_expected != count_other:
            if not quiet:
                raise RuntimeError("Expected vector length {}, but got length {} at {}".format(count_expected,
                                                                                               count_other,
                                                                                               " > ".join(path)))
            return False

        return True

    def to_bytes(self, array):
        if len(array) != self.n:
            raise RuntimeError("Array as length {}, expected {}".format(len(array), self.n))

        bytez = bytearray()

        for value in array:
            bytez += self.scalar.to_bytes(value)

        return bytez

    def from_bytes(self, bytez):
        return np.frombuffer(bytez, self.scalar.numpy_dtype())[:self.n]


class Array(ByteRepresentation):

    def __init__(self, definition, dims, layout):
        super(Array, self).__init__()
        self.definition = definition
        self.dims = tuple(dims) if isinstance(dims, (list, tuple)) else (dims,)
        self.layout = layout  # precomputes alignment

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        self.a = None
        self.precompute_alignment()

    def precompute_alignment(self):
        if self.layout in [Layout.STD140, Layout.STD430, Layout.STDXXX]:
            self.a = self.definition.alignment()

            if self.layout == Layout.STD140:
                self.a += (16 - self.a % 16) % 16

    def shape(self):
        return self.dims

    def size(self):
        s = self.definition.size()
        s += (self.a - s % self.a) % self.a  # pad to array stride
        return s * np.product(self.shape())

    def alignment(self):
        return self.a

    def __str__(self, name=None, indent=2):
        s = self.definition.glsl_dtype()  # "array"
        s += ("[{}]" * len(self.shape())).format(*self.shape())
        s += " [{}] {{ {} }}".format(name or "?", self.definition.__str__(indent=indent + 2))
        return s

    def glsl_dtype(self):
        return ("{}" + "[{}]" * len(self.shape())).format(self.definition.glsl_dtype(), *self.shape())

    def compare(self, other, path=(), quiet=True):
        if not self.compare_type(type(self), type(other), path, quiet):
            return False
        if not self.compare_layout(self.layout, other.layout, path, quiet):
            return False

        shape_expected = self.shape()
        shape_other = other.shape()

        if shape_expected != shape_other:
            if not quiet:
                raise RuntimeError("Expected array shape {}, but got shape {} at {}".format(shape_expected, shape_other,
                                                                                            " > ".join(path)))
            return False

        return self.definition.compare(other.definition,
                                       list(path) + ["array {}".format("x".join(map(str, shape_expected)))],
                                       quiet)

    def to_bytes(self, values):
        if isinstance(self.definition, Scalar):
            return self.to_bytes_for_scalars(values)

        # elif isinstance(self.definition, Vector):
        #     return self.to_bytes_for_vectors(values)
        #
        # elif isinstance(self.definition, Matrix):
        #     return self.to_bytes_for_matrices(values)

        else:
            bytez = bytearray()

            for value in self.iterate_over_nd_array(values, self.shape()):
                bytez += self.definition.to_bytes(value)
                padding = (self.a - len(bytez) % self.a) % self.a
                bytez += bytearray(padding)

            return bytez

    @classmethod
    def iterate_over_nd_array(cls, array, dims):
        for indices in itertools.product(*[range(d) for d in dims]):
            value = array
            for idx in indices:
                value = value[idx]
            yield value

    def to_bytes_for_scalars(self, array):
        if not isinstance(array, np.ndarray):
            raise RuntimeError("Incorrect datatype {}, expected {}".format(type(array), np.ndarray))
        # if array.dtype is not self.definition.numpy_dtype():
        #     raise RuntimeError("Incorrect datatype {}, expected {}".format(array.dtype, self.definition.numpy_dtype()))
        if tuple(array.shape) != self.shape():
            raise RuntimeError("Array has shape {}, expected {}".format(array.shape, self.shape()))

        p = (self.a - self.definition.alignment()) / self.definition.size()
        a = self.a / self.definition.size()

        array_padded = np.zeros(a * np.product(array.shape), dtype=array.dtype)
        mask = (np.arange(len(array_padded)) % a) < (a - p)
        array_padded[mask] = array.flatten()

        # for std430 the following would be sufficient: return array.flatten().tobytes()
        return array_padded.tobytes()

    def from_bytes(self, bytez):
        if isinstance(self.definition, Scalar):
            return self.from_bytes_for_scalars(bytez)

        # elif isinstance(self.definition, Vector):
        #     return self.to_bytes_for_vectors(values)
        #
        # elif isinstance(self.definition, Matrix):
        #     return self.to_bytes_for_matrices(values)

        else:
            values = np.zeros(self.shape()).tolist()
            offset = 0
            size = self.definition.size()

            for indices in itertools.product(*[range(s) for s in self.shape()]):
                _data = values

                for index in indices[:-1]:
                    _data = _data[index]

                _data[indices[-1]] = self.definition.from_bytes(bytez[offset:offset + size])
                offset += size
                offset += (self.a - offset % self.a) % self.a  # bytes

            return values

    def from_bytes_for_scalars(self, bytez):
        p = (self.a - self.definition.alignment()) / self.definition.size()
        a = self.a / self.definition.size()

        array_padded = np.frombuffer(bytez, dtype=self.definition.numpy_dtype())
        mask = (np.arange(a * np.product(self.shape())) % a) < (a - p)
        return array_padded[mask].reshape(self.shape())


class Struct(ByteRepresentation):

    def __init__(self, definitions, layout, member_names=None, type_name=None):
        super(Struct, self).__init__()
        self.definitions = definitions
        self.member_names = member_names or ([None] * len(definitions))
        self.type_name = type_name
        self.layout = layout  # precomputes alignment

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        self.a = None
        self.precompute_alignment()

    def precompute_alignment(self):
        if self.layout in [Layout.STD140, Layout.STD430, Layout.STDXXX]:
            self.a = max([d.alignment() for d in self.definitions])

            if self.layout == Layout.STD140:
                self.a += (16 - self.a % 16) % 16

    def size(self):
        return self.steps()[-1]

    def offsets(self):
        return self.steps()[:-1]

    def steps(self):
        steps = [0]

        for d in self.definitions:
            a = d.alignment()
            padding_before = (a - steps[-1] % a) % a
            steps[-1] = steps[-1] + padding_before  # update last step + size to next step
            steps.append(steps[-1] + d.size())
            padding_after = (a - steps[-1] % a) % a
            steps[-1] = steps[-1] + padding_after

        return steps

    def alignment(self):
        return self.a

    def __str__(self, name=None, indent=2):
        # s = "struct [{}] {{\n".format(name or "?")
        s = "{} [{}] {{\n".format(self.type_name or "<struct_type>", name or "?")
        for i, (definition, member_name) in enumerate(zip(self.definitions, self.member_names)):
            s += "{}({}) {}\n".format(" " * indent, i, definition.__str__(name=member_name, indent=indent + 2))
        s += "{}}}".format(" " * (indent - 2))
        return s

    def glsl_dtype(self):
        return self.type_name or "structType?"

    def compare(self, other, path=(), quiet=True):
        if not self.compare_type(type(self), type(other), path, quiet):
            return False
        if not self.compare_layout(self.layout, other.layout, path, quiet):
            return False

        definitions_expected = self.definitions
        definitions_other = other.definitions

        if len(definitions_expected) != len(definitions_other):
            if not quiet:
                raise RuntimeError("Expected {} members, but got {} members at {}".format(len(definitions_expected),
                                                                                          len(definitions_other),
                                                                                          " > ".join(path)))
            return False

        for i, (definition_expected, definition_other) in enumerate(zip(definitions_expected, definitions_other)):
            if not definition_expected.compare(definition_other, list(path) + ["member {}".format(i)], quiet):
                return False

        return True

    def to_bytes(self, values):
        bytez = bytearray()

        for d in self.definitions:
            a = d.alignment()
            padding_before = (a - len(bytez) % a) % a
            bytez += bytearray(padding_before)

            bytez += d.to_bytes(values[d])

            # TODO: figure out when the following is needed
            # padding_after = (a - len(bytez) % a) % a
            # bytez += bytearray(padding_after)

        return bytez

    def from_bytes(self, bytez):
        values = {}
        offset = 0

        for d in self.definitions:
            a = d.alignment()
            offset += (a - offset % a) % a  # padding before
            size = d.size()

            values[d] = d.from_bytes(bytez[offset:offset + size])

            # TODO: figure out when the following is needed
            # offset += size
            # offset += (a - offset % a) % a  # padding after

        return values


class ByteCache(object):

    def __init__(self, definition):
        if type(definition) != Struct:
            raise RuntimeError("ByteCaches can only be initialized with struct definitions")
        self.definition = definition
        self.values = {}

        for i, d in enumerate(self.definition.definitions):
            value = None

            if isinstance(d, Struct):
                value = ByteCache(d)

            if isinstance(d, Array):
                if isinstance(d.definition, Struct):
                    value = np.zeros(d.shape()).tolist()

                    for indices in itertools.product(*[range(s) for s in d.shape()]):
                        _data = value
                        for index in indices[:-1]:
                            _data = _data[index]
                        _data[indices[-1]] = ByteCache(d.definition)

            self.values[d] = value

    def get_as_dict(self):
        data = {}

        for d in self.definition.definitions:
            value = self.values[d]

            if isinstance(d, Struct):
                value = self.values[d].get_as_dict()

            if isinstance(d, Array):
                if isinstance(d.definition, Struct):
                    value = np.zeros(d.shape()).tolist()

                    for indices in itertools.product(*[range(s) for s in d.shape()]):
                        _data1 = value
                        _data2 = self.values[d]

                        for index in indices[:-1]:
                            _data1 = _data1[index]
                            _data2 = _data2[index]

                        _data1[indices[-1]] = _data2[indices[-1]].get_as_dict()

            data[d] = value

        return data

    def set_from_dict(self, values):
        pass

    def __str__(self):
        s = self.__class__.__name__ + " around\n"
        s += str(self.definition)
        return s

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.values[self.definition.definitions[key]]

        if isinstance(key, str):
            if key not in self.definition.member_names:
                pass
            index = self.definition.member_names.index(key)
            return self.values[self.definition.definitions[index]]

        if isinstance(key, ByteRepresentation):
            return self.values[key]

        raise RuntimeError("Invalid key")

    def __setitem__(self, key, value):
        pass


# class Matrix(ByteRepresentation):
#
#     def __init__(self, n=4, m=4, dtype=ByteRepresentation.FLOAT):
#         super(Matrix, self).__init__()
#         if dtype not in (self.FLOAT, self.DOUBLE):
#             raise RuntimeError("Matrices of type {} are not supported".format(dtype))
#         self.dtype = dtype
#         self.n = n  # columns
#         self.m = m  # rows
#         self.scalar = Scalar.of(dtype)
#
#     def size(self):
#         return self.scalar.size() * self.n * self.m
#
#     def shape(self):
#         return self.m, self.n
#
#     def alignment(self, layout, order):
#         # "A row-major matrix of C columns has a base alignment equal to the base alignment of a vector of C matrix
#         #  components."
#         if self.order == self.ROW_MAJOR:
#             return Vector(self.layout, self.n, self.dtype).alignment()
#
#         # "A column-major matrix has a base alignment equal to the base alignment of the matrix column type."
#         if self.order == self.COLUMN_MAJOR:
#             return self.scalar.alignment()
#         return -1
#
#     def glsl_dtype(self):
#         return "{}mat{}x{}".format("d" if self.dtype == self.DOUBLE else "", self.n, self.m)
#
#     def to_bytes(self, array, layout, order):
#         if isinstance(array, (list, tuple)):
#             array = np.array(array, dtype=self.scalar.numpy_dtype())
#
#         if array.shape != self.shape():
#             raise RuntimeError("Array as shape {}, expected {}".format(array.shape, self.shape()))
#
#         bytez = bytearray()
#
#         if self.order == self.ROW_MAJOR:
#             row_vector = Vector(self.layout, self.n, self.dtype)
#
#             for r in range(self.m):
#                 bytez += row_vector.to_bytes(array[r, :])
#
#         if self.order == self.COLUMN_MAJOR:
#             for value in array.transpose().flatten():
#                 bytez += self.scalar.to_bytes(value)
#
#         return bytez



# specs
# https://www.khronos.org/registry/vulkan/specs/1.1/html/chap14.html#interfaces-resources-layout
# https://github.com/KhronosGroup/glslang/issues/201#issuecomment-204785552 (example)
#
# https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt
#   CTRL+F "Sub-section 2.15.3.1.2"
#   CTRL+F "The following example illustrates the rules specified by the "std140" layout"
# https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_storage_buffer_object.txt
# https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt
#
# Wiki layout bindings etc
# https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)
# https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)
#
# more stuff to come
# http://www.paranormal-entertainment.com/idr/blog/posts/2014-01-29T17:08:42Z-GLSL_multidimensional_array_madness/


