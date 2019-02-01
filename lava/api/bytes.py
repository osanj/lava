# -*- coding: UTF-8 -*-

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ByteRepresentation(object):

    """Data structure implementation of Vulkan specification 14.5.4."""

    LAYOUT_STD140 = "std140"
    LAYOUT_STD430 = "std430"
    LAYOUT_DEFAULT = LAYOUT_STD140

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

    def alignment(self, layout, order):
        # called 'base alignment' in the specs
        raise NotImplementedError()

    def glsl(self, var_name):
        return "{} {};".format(self.glsl_dtype(), var_name)

    def glsl_dtype(self):
        raise NotImplementedError()

    def to_bytes(self, values, layout, order):
        raise NotImplementedError()

    # def from_bytes(self, bytez):
    #     raise NotImplementedError()


class Container(ByteRepresentation):

    def __init__(self, layout, order, *definitions):
        super(Container, self).__init__()
        self.layout = layout
        self.order = order
        self.definitions = definitions

    @classmethod
    def of(cls, *definitions):
        return cls(cls.LAYOUT_DEFAULT, cls.ORDER_DEFAULT, *definitions)

    def size(self):
        step = 0

        for d in self.definitions:
            a = d.alignment(self.layout, self.order)
            padding = (a - step % a) % a
            step += padding + d.size()

        return step

    def alignment(self, *args, **kwargs):
        raise RuntimeError()

    def glsl_dtype(self):
        raise NotImplementedError()

    def to_bytes(self, values, *args, **kwargs):
        bytez = bytearray()
        settings = [self.layout, self.order]

        for d in self.definitions:
            a = d.alignment(*settings)
            padding = (a - len(bytez) % a) % a
            bytez += bytearray(padding)
            bytez += d.to_bytes(values[d], *settings)

        return bytez


class Scalar(ByteRepresentation):

    def __init__(self, *input_dtypes):
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

    def alignment(self,  *args, **kwargs):
        # "A scalar of size N has a base alignment of N."
        return self.size()

    def glsl_dtype(self):
        raise NotImplementedError()

    def to_bytes(self, value, *args, **kwargs):
        if not isinstance(value, self.input_dtypes):
            raise RuntimeError("{} got dtype {} (expects one of the following {})".format(
                self.__class__.__name__, type(value), self.input_dtypes
            ))

    def numpy_dtype(self):
        raise NotImplementedError()


class ScalarInt(Scalar):

    def __init__(self):
        super(ScalarInt, self).__init__(int, np.int32)

    def numpy_dtype(self):
        return np.int32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "int"

    def to_bytes(self, value, *args, **kwargs):
        super(ScalarInt, self).to_bytes(value)
        if type(value) is int:
            if not (-(0x7FFFFFFF + 1) <= value <= 0x7FFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.int32(value)
        return bytearray(value.tobytes())


class ScalarUnsignedInt(Scalar):

    def __init__(self):
        super(ScalarUnsignedInt, self).__init__(int, np.uint32)

    def numpy_dtype(self):
        return np.uint32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "uint"

    def to_bytes(self, value, *args, **kwargs):
        super(ScalarUnsignedInt, self).to_bytes(value)
        if type(value) is int:
            if not (0 <= value <= 0xFFFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.uint32(value)
        return bytearray(value.tobytes())


class ScalarFloat(Scalar):

    def __init__(self):
        super(ScalarFloat, self).__init__(float, np.float32)

    def numpy_dtype(self):
        return np.float32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "float"

    def to_bytes(self, value, *args, **kwargs):
        super(ScalarFloat, self).to_bytes(value)
        if type(value) is float:
            # TODO: add range check
            value = np.float32(value)
        return bytearray(value.tobytes())


class ScalarDouble(Scalar):

    def __init__(self):
        super(ScalarDouble, self).__init__(float, np.float64)

    def numpy_dtype(self):
        return np.float64

    def size(self):
        return 8

    def glsl_dtype(self):
        return "double"

    def to_bytes(self, value, *args, **kwargs):
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

    def alignment(self,  *args, **kwargs):
        if self.n == 2:
            # "A two-component vector, with components of size N, has a base alignment of 2 N."
            return self.scalar.size() * 2
        if self.n in (3, 4):
            # "A three- or four-component vector, with components of size N, has a base alignment of 4 N."
            return self.scalar.size() * 4
        return -1

    def glsl_dtype(self):
        return "{}vec{}".format(self.dtype.lower()[0] if self.dtype is not self.FLOAT else "", self.n)

    def to_bytes(self, array, *args, **kwargs):
        if len(array) != self.n:
            raise RuntimeError("Array as length {}, expected {}".format(len(array), self.n))

        bytez = bytearray()

        for value in array:
            bytez += self.scalar.to_bytes(value)

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
# https://www.khronos.org/registry/vulkan/specs/1.1/html/chap14.html#interfaces-resources
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

