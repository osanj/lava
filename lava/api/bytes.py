# -*- coding: UTF-8 -*-

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ByteRepresentation(object):

    """Data structure implementation of Vulkan specification 14.5.4."""

    LAYOUT_STD140 = "std140"
    LAYOUT_STD430 = "std430"
    LAYOUT_DEFAULT = LAYOUT_STD140

    #BOOL = "bool"
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    DOUBLE = "double"

    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"

    def __init__(self, layout):
        self.layout = layout

    def size(self):
        raise NotImplementedError()

    def alignment(self):
        # called 'base alignment' in the specs
        raise NotImplementedError()

    def size_aligned(self):
        return self.size()

    def glsl(self):
        raise NotImplementedError()

    def convert_data_to_aligned_bytes(self, *args, **kwargs):
        raise NotImplementedError()

    def convert_aligned_bytes_to_data(self, bytez):
        raise NotImplementedError()


class Scalar(ByteRepresentation):

    def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, *input_dtypes):
        super(Scalar, self).__init__(layout)
        self.input_dtypes = input_dtypes

    @classmethod
    def of(cls, dtype, layout=ByteRepresentation.LAYOUT_DEFAULT):
        if dtype == cls.INT:
            return ScalarInt(layout)
        if dtype == cls.UINT:
            return ScalarUnsignedInt(layout)
        if dtype == cls.FLOAT:
            return ScalarFloat(layout)
        if dtype == cls.DOUBLE:
            return ScalarDouble(layout)
        raise RuntimeError("Unknown scalar type '{}'".format(dtype))

    def alignment(self):
        # "A scalar of size N has a base alignment of N."
        return self.size()

    def convert_data_to_aligned_bytes(self, value):
        if not isinstance(value, self.input_dtypes):
            raise RuntimeError("{} got dtype {} (expects one of the following {})".format(
                self.__class__.__name__, type(value), self.input_dtypes
            ))

    def numpy_dtype(self):
        raise NotImplementedError()


class ScalarInt(Scalar):

    def __init__(self, layout=ByteRepresentation.LAYOUT_STD140):
        super(ScalarInt, self).__init__(layout, int, np.int32)

    def numpy_dtype(self):
        return np.int32

    def size(self):
        return 4

    def glsl(self):
        return "int myInt;"

    def convert_data_to_aligned_bytes(self, value):
        super(ScalarInt, self).convert_data_to_aligned_bytes(value)
        if type(value) is int:
            if not (-(0x7FFFFFFF + 1) <= value <= 0x7FFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.int32(value)
        return bytearray(value.tobytes())

    def convert_aligned_bytes_to_data(self, bytez):
        return 0


class ScalarUnsignedInt(Scalar):

    def __init__(self, layout=ByteRepresentation.LAYOUT_STD140):
        super(ScalarUnsignedInt, self).__init__(layout, int, np.uint32)

    def numpy_dtype(self):
        return np.uint32

    def size(self):
        return 4

    def glsl(self):
        return "uint myUnsignedInt;"

    def convert_data_to_aligned_bytes(self, value):
        super(ScalarUnsignedInt, self).convert_data_to_aligned_bytes(value)
        if type(value) is int:
            if not (0 <= value <= 0xFFFFFFFF):
                raise RuntimeError("Value {} is out of memory bounds")
            value = np.uint32(value)
        return bytearray(value.tobytes())

    def convert_aligned_bytes_to_data(self, bytez):
        return 0


class ScalarFloat(Scalar):

    def __init__(self, layout=ByteRepresentation.LAYOUT_STD140):
        super(ScalarFloat, self).__init__(layout, float, np.float32)

    def numpy_dtype(self):
        return np.float32

    def size(self):
        return 4

    def glsl(self):
        return "float myFloat;"

    def convert_data_to_aligned_bytes(self, value):
        super(ScalarFloat, self).convert_data_to_aligned_bytes(value)
        if type(value) is float:
            # TODO: add range check
            value = np.float32(value)
        return bytearray(value.tobytes())

    def convert_aligned_bytes_to_data(self, bytez):
        return 0


class ScalarDouble(Scalar):

    def __init__(self, layout=ByteRepresentation.LAYOUT_STD140):
        super(ScalarDouble, self).__init__(layout, float, np.float64)

    def numpy_dtype(self):
        return np.float64

    def size(self):
        return 8

    def glsl(self):
        return "double myDouble;"

    def convert_data_to_aligned_bytes(self, value):
        super(ScalarDouble, self).convert_data_to_aligned_bytes(value)
        if type(value) is float:
            # TODO: add range check
            value = np.float64(value)
        return bytearray(value.tobytes())

    def convert_aligned_bytes_to_data(self, bytez):
        return 0


class Vector(ByteRepresentation):

    def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, n=4, dtype=ByteRepresentation.FLOAT):
        super(Vector, self).__init__(layout)
        self.dtype = dtype
        self.n = n
        self.scalar = Scalar.of(dtype, layout)

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

    def size_aligned(self):
        return self.alignment()

    def glsl(self):
        return "{}vec{} myVec;".format(self.dtype.lower()[0] if self.dtype is not self.FLOAT else "", self.n)

    def convert_data_to_aligned_bytes(self, array):
        if len(array) != self.n:
            raise RuntimeError("Array as length {}, expected {}".format(len(array), self.n))

        bytez = bytearray()

        for value in array:
            bytez += self.scalar.convert_data_to_aligned_bytes(value)

        if self.n == 3:
            bytez += bytearray(self.scalar.size())

        return bytez

    def convert_aligned_bytes_to_data(self, bytez):
        raise NotImplementedError()


class Matrix(ByteRepresentation):

    def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, order=ByteRepresentation.ROW_MAJOR, n=4, m=4,
                 dtype=ByteRepresentation.FLOAT):
        super(Matrix, self).__init__(layout)
        self.order = order

        if dtype not in (self.FLOAT, self.DOUBLE):
            raise RuntimeError("Matrices of type {} are not supported".format(dtype))
        self.dtype = dtype
        self.n = n  # columns
        self.m = m  # rows
        self.scalar = Scalar.of(dtype, layout)

    def size(self):
        return self.scalar.size() * self.n * self.m

    def shape(self):
        return self.m, self.n

    def alignment(self):
        # "A row-major matrix of C columns has a base alignment equal to the base alignment of a vector of C matrix
        #  components."
        if self.order == self.ROW_MAJOR:
            return Vector(self.layout, self.n, self.dtype).alignment()

        # "A column-major matrix has a base alignment equal to the base alignment of the matrix column type."
        if self.order == self.COLUMN_MAJOR:
            return self.scalar.alignment()
        return -1

    def size_aligned(self):
        if self.order == self.ROW_MAJOR:
            return Vector(self.layout, self.n, self.dtype).alignment() * self.m
        if self.order == self.COLUMN_MAJOR:
            return self.size()
        return -1

    def glsl(self):
        return "{}mat{}x{} myMat;".format("d" if self.dtype == self.DOUBLE else "", self.n, self.m)

    def convert_data_to_aligned_bytes(self, array):
        if isinstance(array, (list, tuple)):
            array = np.array(array, dtype=self.scalar.numpy_dtype())

        if array.shape != self.shape():
            raise RuntimeError("Array as shape {}, expected {}".format(array.shape, self.shape()))

        bytez = bytearray()

        if self.order == self.ROW_MAJOR:
            row_vector = Vector(self.layout, self.n, self.dtype)

            for r in range(self.m):
                bytez += row_vector.convert_data_to_aligned_bytes(array[r, :])

        if self.order == self.COLUMN_MAJOR:
            for value in array.transpose().flatten():
                bytez += self.scalar.convert_data_to_aligned_bytes(value)

        return bytez

    def convert_aligned_bytes_to_data(self, bytez):
        raise NotImplementedError()


class Struct(ByteRepresentation):

    def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, *components):
        super(Struct, self).__init__(layout)
        for i, component in enumerate(components):
            if component.layout != layout:
                raise RuntimeError("Layout mismatch: struct has {}, but component at index {} has {}".format(
                    layout, i, component.layout
                ))

        self.components = components

    def size(self):
        return sum([c.size() for c in self.components])

    def alignment(self):
        return max([c.alignment() for c in self.components])

    def size_aligned(self):
        size = self.size()
        alignment = self.size()


    def glsl(self):
        raise NotImplementedError()

    def convert_data_to_aligned_bytes(self, *args, **kwargs):
        raise NotImplementedError()

    def convert_aligned_bytes_to_data(self, bytez):
        raise NotImplementedError()



class Tensor(ByteRepresentation):

    """ata structure implementation of ARB_arrays_of_arrays"""
    # https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt

    def __init__(self, layout=ByteRepresentation.LAYOUT_DEFAULT, order=ByteRepresentation.ROW_MAJOR, dims=(),
                 dtype=ByteRepresentation.FLOAT):

        pass

    def shape(self):
        return (4, 4, 4, 4)


class DynamicArray(Struct):

    def __init__(self):
        pass


# more stuff to come
# http://www.paranormal-entertainment.com/idr/blog/posts/2014-01-29T17:08:42Z-GLSL_multidimensional_array_madness/



if __name__ == "__main__":
    scalar = Scalar.of(Scalar.INT)
    print scalar.glsl()

    vec = Vector(n=3, dtype=Scalar.UINT)
    print vec.glsl()

