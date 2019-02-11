# -*- coding: UTF-8 -*-

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


class ByteRepresentation(object):

    """Data structure implementation of Vulkan specification 14.5.4."""

    # References
    # https://www.khronos.org/registry/vulkan/specs/1.1/html/chap14.html#interfaces-resources-layout
    # https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt
    #   search document for "Sub-section 2.15.3.1.2"
    #   search document for "The following example illustrates the rules specified by the "std140" layout"

    LAYOUT_STD140 = "std140"
    LAYOUT_STD430 = "std430"
    LAYOUT_DEFAULT = LAYOUT_STD140
    LAYOUT_STDXXX = "stdxxx"  # can not be inferred / does not matter

    ORDER_COLUMN_MAJOR = "column_major"
    ORDER_ROW_MAJOR = "row_major"
    ORDER_DEFAULT = ORDER_ROW_MAJOR

    #BOOL = "bool"  # machine unit 1 byte? (probably not 1 bit)
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    DOUBLE = "double"

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
        if layout_expected == cls.LAYOUT_STDXXX:
            return True

        if layout_expected != layout_other and layout_other != cls.LAYOUT_STDXXX:
            if not quiet:
                raise RuntimeError(
                    "Expected layout {}, but got layout {} at {}".format(layout_expected, layout_other, " > ".join(path)))
            return False

        return True

    def compare(self, other, path, quiet=True):
        raise NotImplementedError()

    def to_bytes(self, values):
        raise NotImplementedError()

    # def from_bytes(self, bytez):
    #     raise NotImplementedError()


class Scalar(ByteRepresentation):

    def __init__(self, input_dtypes):
        super(Scalar, self).__init__()
        self.input_dtypes = input_dtypes

    @classmethod
    def int(cls):
        return cls.of(Scalar.INT)

    @classmethod
    def uint(cls):
        return cls.of(Scalar.UINT)

    @classmethod
    def float(cls):
        return cls.of(Scalar.FLOAT)

    @classmethod
    def double(cls):
        return cls.of(Scalar.DOUBLE)

    @classmethod
    def of(cls, dtype):
        if dtype == cls.INT:
            return ScalarInt()
        if dtype == cls.UINT:
            return ScalarUnsignedInt()
        if dtype == cls.FLOAT:
            return ScalarFloat()
        if dtype == cls.DOUBLE:
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

    def __init__(self, n=4, dtype=ByteRepresentation.FLOAT):
        super(Vector, self).__init__()
        self.dtype = dtype
        self.n = n
        self.scalar = Scalar.of(dtype)

    @classmethod
    def ivec2(cls):
        return Vector(2, cls.INT)

    @classmethod
    def ivec3(cls):
        return Vector(3, cls.INT)

    @classmethod
    def ivec4(cls):
        return Vector(4, cls.INT)

    @classmethod
    def uvec2(cls):
        return Vector(2, cls.UINT)

    @classmethod
    def uvec3(cls):
        return Vector(3, cls.UINT)

    @classmethod
    def uvec4(cls):
        return Vector(4, cls.UINT)

    @classmethod
    def vec2(cls):
        return Vector(2, cls.FLOAT)

    @classmethod
    def vec3(cls):
        return Vector(3, cls.FLOAT)

    @classmethod
    def vec4(cls):
        return Vector(4, cls.FLOAT)

    @classmethod
    def dvec2(cls):
        return Vector(2, cls.DOUBLE)

    @classmethod
    def dvec3(cls):
        return Vector(3, cls.DOUBLE)

    @classmethod
    def dvec4(cls):
        return Vector(4, cls.DOUBLE)

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
        return "{}vec{}".format(self.dtype.lower()[0] if self.dtype is not self.FLOAT else "", self.n)

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
        if self.layout in [self.LAYOUT_STD140, self.LAYOUT_STD430, self.LAYOUT_STDXXX]:
            self.a = self.definition.alignment()

            if self.layout == self.LAYOUT_STD140:
                self.a += (16 - self.a % 16) % 16

    def shape(self):
        return self.dims

    def size(self):
        s = self.definition.size()
        s += (self.a - s % self.a) % self.a  # pad to array stride
        return s * np.product(self.dims)

    def alignment(self):
        return self.a

    def __str__(self, name=None, indent=2):
        s = self.definition.glsl_dtype()  # "array"
        s += ("[{}]" * len(self.shape())).format(*self.dims)
        s += " [{}] {{ {} }}".format(name or "?", self.definition.__str__(indent=indent + 2))
        return s

    def glsl_dtype(self):
        return ("{}" + "[{}]" * len(self.shape())).format(self.definition.glsl_dtype(), *self.dims)

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
        else:
            bytez = bytearray()

            for value in self.iterate_over_nd_array(values, self.dims):
                bytez += self.definition.to_bytes(value)
                padding = (self.a - len(bytez) % self.a) % self.a
                bytez += bytearray(padding)

            return bytez

    def iterate_over_nd_array(self, array, dims):
        for indices in itertools.product(*[range(d) for d in dims]):
            value = array
            for idx in indices:
                value = value[idx]
            yield value

    def to_bytes_for_scalars(self, array, *args, **kwargs):
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
        if self.layout in [self.LAYOUT_STD140, self.LAYOUT_STD430, self.LAYOUT_STDXXX]:
            self.a = max([d.alignment() for d in self.definitions])

            if self.layout == self.LAYOUT_STD140:
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
            padding_after = (a - len(bytez) % a) % a
            bytez += bytearray(padding_after)

        return bytez






class Matrix(ByteRepresentation):

    def __init__(self, n=4, m=4, dtype=ByteRepresentation.FLOAT):
        super(Matrix, self).__init__()
        if dtype not in (self.FLOAT, self.DOUBLE):
            raise RuntimeError("Matrices of type {} are not supported".format(dtype))
        self.dtype = dtype
        self.n = n  # columns
        self.m = m  # rows
        self.scalar = Scalar.of(dtype)

    def size(self):
        return self.scalar.size() * self.n * self.m

    def shape(self):
        return self.m, self.n

    def alignment(self, layout, order):
        # "A row-major matrix of C columns has a base alignment equal to the base alignment of a vector of C matrix
        #  components."
        if self.order == self.ROW_MAJOR:
            return Vector(self.layout, self.n, self.dtype).alignment()

        # "A column-major matrix has a base alignment equal to the base alignment of the matrix column type."
        if self.order == self.COLUMN_MAJOR:
            return self.scalar.alignment()
        return -1

    def glsl_dtype(self):
        return "{}mat{}x{}".format("d" if self.dtype == self.DOUBLE else "", self.n, self.m)

    def to_bytes(self, array, layout, order):
        if isinstance(array, (list, tuple)):
            array = np.array(array, dtype=self.scalar.numpy_dtype())

        if array.shape != self.shape():
            raise RuntimeError("Array as shape {}, expected {}".format(array.shape, self.shape()))

        bytez = bytearray()

        if self.order == self.ROW_MAJOR:
            row_vector = Vector(self.layout, self.n, self.dtype)

            for r in range(self.m):
                bytez += row_vector.to_bytes(array[r, :])

        if self.order == self.COLUMN_MAJOR:
            for value in array.transpose().flatten():
                bytez += self.scalar.to_bytes(value)

        return bytez


# class Struct(ByteRepresentation):
#
#     def __init__(self, *components):
#         super(Struct, self).__init__()
#         for i, component in enumerate(components):
#             if component.layout != layout:
#                 raise RuntimeError("Layout mismatch: struct has {}, but component at index {} has {}".format(
#                     layout, i, component.layout
#                 ))
#
#         self.components = components
#
#     def size(self):
#         return sum([c.size() for c in self.components])
#
#     def alignment(self):
#         return max([c.alignment() for c in self.components])
#
#     def size_aligned(self):
#         size = self.size()
#         alignment = self.size()
#
#
#     def glsl(self):
#         raise NotImplementedError()
#
#     def convert_data_to_aligned_bytes(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def convert_aligned_bytes_to_data(self, bytez):
#         raise NotImplementedError()
#
#
#
# class Tensor(ByteRepresentation):
#
#     """ata structure implementation of ARB_arrays_of_arrays"""
#     # https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt
#
#     def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, order=ByteRepresentation.ROW_MAJOR, dims=(),
#                  dtype=ByteRepresentation.FLOAT):
#
#         pass
#
#     def shape(self):
#         return (4, 4, 4, 4)
#
#
# class DynamicArray(Struct):
#
#     def __init__(self):
#         pass


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


if __name__ == "__main__":
    scalar = Scalar.of(Scalar.INT)
    print scalar.glsl()

    vec = Vector(n=3, dtype=Scalar.UINT)
    print vec.glsl()

