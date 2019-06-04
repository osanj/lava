# -*- coding: UTF-8 -*-

import numpy as np

from lava.api.constants.spirv import DataType, Layout, Order
from lava.api.util import NdArray, LavaError


class BytesError(LavaError):

    def __init__(self, message):
        super(BytesError, self).__init__(message)

    @classmethod
    def out_of_bounds(cls, value, glsl_dtype):
        raise cls("Value {} is out of memory bounds for type {}".format(value, glsl_dtype))


class ByteRepresentation(object):

    """Data structure implementation of Vulkan specification 14.5.4."""

    # References
    # https://www.khronos.org/registry/vulkan/specs/1.1/html/chap14.html#interfaces-resources-layout
    # https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt
    #   search document for "Sub-section 2.15.3.1.2"
    #   search document for "The following example illustrates the rules specified by the "std140" layout"

    def __init__(self):
        pass

    def copy(self):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def alignment(self):
        # called 'base alignment' in the specs
        raise NotImplementedError()

    def __str__(self, indent=2):
        raise NotImplementedError()

    # def glsl(self, var_name):
    #     return "{} {};".format(self.glsl_dtype(), var_name)

    def glsl_dtype(self):
        raise NotImplementedError()

    @classmethod
    def path_to_str(cls, path):
        return " > ".join(path)

    @classmethod
    def compare_type(cls, type_expected, type_other, path, quiet=True):
        if type_expected != type_other:
            if not quiet:
                raise TypeError(
                    "Expected type {}, but got type {} at {}".format(type_expected, type_other, cls.path_to_str(path)))
            return False

        return True

    @classmethod
    def compare_layout(cls, layout_expected, layout_other, path, quiet=True):
        if layout_expected == Layout.STDXXX:
            return True

        if layout_expected != layout_other and layout_other != Layout.STDXXX:
            if not quiet:
                raise TypeError(
                    "Expected layout {}, but got layout {} at {}".format(layout_expected, layout_other,
                                                                         cls.path_to_str(path)))
            return False

        return True

    @classmethod
    def compare_order(cls, order_expected, order_other, path, quiet=True):
        if order_expected != order_other:
            if not quiet:
                raise TypeError(
                    "Expected layout {}, but got layout {} at {}".format(order_expected, order_other,
                                                                         cls.path_to_str(path)))
            return False
        return True

    @classmethod
    def compare_shape(cls, shape_expected, shape_other, path, quiet):
        if shape_expected != shape_other:
            if not quiet:
                raise TypeError("Expected shape {}, but got shape {} at {}".format(shape_expected, shape_other,
                                                                                   cls.path_to_str(path)))
            return False
        return True

    def compare(self, other, path, quiet=True):
        raise NotImplementedError()

    def to_bytes(self, values, path=()):
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
        raise ValueError("Unknown scalar type '{}'".format(dtype))

    def copy(self):
        return self.__class__()

    def alignment(self):
        # "A scalar of size N has a base alignment of N."
        return self.size()

    def __str__(self, name=None, indent=2):
        return "{} [{}]".format(self.glsl_dtype(), name or "?")

    def glsl_dtype(self):
        raise NotImplementedError()

    def compare(self, other, path=(), quiet=True):
        return self.compare_type(type(self), type(other), path, quiet)

    def to_bytes(self, value, path=()):
        if not isinstance(value, self.input_dtypes):
            raise TypeError("{} got dtype {}, expected one of the following {} at {}".format(
                self.__class__.__name__, type(value), self.input_dtypes, self.path_to_str(path)
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

    def to_bytes(self, value, path=()):
        super(ScalarInt, self).to_bytes(value, path)
        if type(value) is int:
            if not (-(0x7FFFFFFF + 1) <= value <= 0x7FFFFFFF):
                raise BytesError.out_of_bounds(value, self.glsl_dtype())
            value = np.int32(value)
        return bytearray(value.tobytes())


class ScalarUnsignedInt(Scalar):

    def __init__(self):
        super(ScalarUnsignedInt, self).__init__((int, np.uint32, bool, np.bool, np.bool_))

    def numpy_dtype(self):
        return np.uint32

    def size(self):
        return 4

    def glsl_dtype(self):
        return "uint"

    def to_bytes(self, value, path=()):
        super(ScalarUnsignedInt, self).to_bytes(value, path)
        if not (0 <= value <= 0xFFFFFFFF):
            raise BytesError.out_of_bounds(value, self.glsl_dtype())
        if type(value) != np.uint32:
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

    def to_bytes(self, value, path=()):
        super(ScalarFloat, self).to_bytes(value, path)
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

    def to_bytes(self, value, path=()):
        super(ScalarDouble, self).to_bytes(value, path)
        if type(value) is float:
            # TODO: add range check
            value = np.float64(value)
        return bytearray(value.tobytes())


class Vector(ByteRepresentation):

    def __init__(self, n, dtype):
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

    def copy(self):
        return Vector(self.n, self.dtype)

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
                raise TypeError("Expected vector length {}, but got length {} at {}".format(
                    count_expected, count_other, self.path_to_str(path)))
            return False

        return True

    def to_bytes(self, array, path=()):
        expected_types = (np.ndarray, list, tuple)
        if not isinstance(array, expected_types):
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(type(array), self.glsl_dtype(), expected_types, self.path_to_str(path)))
        if len(array) != self.n:
            raise TypeError("Got length {} for {} variable, expected {} at {}"
                            .format(len(array), self.glsl_dtype(), self.n, self.path_to_str(path)))

        bytez = bytearray()

        for value in array:
            bytez += self.scalar.to_bytes(value, path)

        return bytez

    def from_bytes(self, bytez):
        return np.frombuffer(bytez, self.scalar.numpy_dtype())[:self.n]


class Matrix(ByteRepresentation):

    DEFAULT_ORDER = Order.COLUMN_MAJOR

    def __init__(self, cols, rows, dtype, layout, order=DEFAULT_ORDER):
        super(Matrix, self).__init__()
        if dtype not in (DataType.FLOAT, DataType.DOUBLE):
            raise TypeError("Matrices of type {} are not supported".format(dtype))
        self.dtype = dtype
        self.cols = cols
        self.rows = rows
        self.order = order
        self.layout = layout

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order
        self.vector = Vector(self.rows if order == Order.COLUMN_MAJOR else self.cols, self.dtype)
        self.vector_count = self.cols if order == Order.COLUMN_MAJOR else self.rows

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
            self.a = self.vector.alignment()

            if self.layout == Layout.STD140:
                self.a += (16 - self.a % 16) % 16

    def copy(self):
        return Matrix(self.cols, self.rows, self.dtype, self.layout, self.order)

    def size(self):
        return self.step_size() * self.vector_count

    def stride(self):
        return self.step_size()

    def step_size(self):
        s = self.vector.size()
        s += (self.a - s % self.a) % self.a  # pad to array stride
        return s

    def shape(self):
        return self.rows, self.cols

    def alignment(self):
        return self.a

    def __str__(self, name=None, indent=2):
        return "{} [{}]".format(self.glsl_dtype(), name or "?")

    def glsl_dtype(self):
        return "{}mat{}x{}".format(self.dtype.lower()[0] if self.dtype is not DataType.FLOAT else "", self.cols, self.rows)

    def compare(self, other, path=(), quiet=True):
        if not self.compare_type(type(self), type(other), path, quiet):
            return False
        if not self.compare_layout(self.layout, other.layout, path, quiet):
            return False
        if not self.compare_shape(self.shape(), other.shape(), path, quiet):
            return False
        return True

    def to_bytes(self, array, path=()):
        expected_types = (np.ndarray, list, tuple)
        if not (isinstance(array, expected_types) or isinstance(array[0], expected_types)):
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(type(array), self.glsl_dtype(), expected_types, self.path_to_str(path)))
        shape = (len(array), len(array[0]))
        expected_shape = self.shape()
        if shape != expected_shape:
            raise TypeError("Got shape {} for {} variable, expected {} at {}"
                            .format(len(array), self.glsl_dtype(), expected_shape, self.path_to_str(path)))

        bytez = bytearray()

        if self.order == Order.COLUMN_MAJOR:
            for col in range(self.cols):
                bytez += self.vector.to_bytes([array[row][col] for row in range(self.rows)], path)
                padding = (self.a - len(bytez) % self.a) % self.a
                bytez += bytearray(padding)

        if self.order == Order.ROW_MAJOR:
            for row in range(self.rows):
                bytez += self.vector.to_bytes([array[row][col] for col in range(self.cols)], path)
                padding = (self.a - len(bytez) % self.a) % self.a
                bytez += bytearray(padding)

        return bytez

    def from_bytes(self, bytez):
        array = np.zeros((self.rows, self.cols), dtype=self.vector.scalar.numpy_dtype())
        offset = 0
        size = self.vector.size()

        if self.order == Order.COLUMN_MAJOR:
            for col in range(self.cols):
                array[:, col] = self.vector.from_bytes(bytez[offset:offset + size])
                offset += size
                offset += (self.a - offset % self.a) % self.a  # bytes

            return array

        if self.order == Order.ROW_MAJOR:
            for row in range(self.rows):
                array[row, :] = self.vector.from_bytes(bytez[offset:offset + size])
                offset += size
                offset += (self.a - offset % self.a) % self.a  # bytes

            return array

        return None


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

        # set children first
        if isinstance(self.definition, (Matrix, Struct)):
            self.definition.layout = layout

        self.a = None
        self.precompute_alignment()

    def precompute_alignment(self):
        if self.layout in [Layout.STD140, Layout.STD430, Layout.STDXXX]:
            self.a = self.definition.alignment()

            if self.layout == Layout.STD140:
                self.a += (16 - self.a % 16) % 16

    def shape(self):
        return self.dims

    def copy(self):
        return Array(self.definition.copy(), self.dims, self.layout)

    def size(self):
        return self.step_size() * np.product(self.shape())

    def strides(self):
        strides = [self.step_size()]
        shape = self.shape()

        for i in reversed(range(1, len(shape))):
            strides.insert(0, shape[i] * strides[0])

        return strides

    def step_size(self):
        s = self.definition.size()
        s += (self.a - s % self.a) % self.a  # pad to array stride
        return s

    def alignment(self):
        return self.a

    @classmethod
    def is_array_of_structs(cls, definition):
        if isinstance(definition, Array):
            if isinstance(definition.definition, Struct):
                return True
        return False

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
        if not self.compare_shape(self.shape(), other.shape(), path, quiet):
            return False

        return self.definition.compare(other.definition,
                                       list(path) + ["array {}".format("x".join(map(str, self.shape())))], quiet)

    def to_bytes(self, values, path=()):
        if isinstance(self.definition, Scalar):
            return self.to_bytes_for_scalars(values, path)

        elif isinstance(self.definition, Vector):
            return self.to_bytes_for_vectors(values, path)

        elif isinstance(self.definition, Matrix):
            return self.to_bytes_for_matrices(values, path)

        else:
            bytez = bytearray()

            for indices in NdArray.iterate(self.shape()):
                bytez += self.definition.to_bytes(NdArray.get(values, indices),
                                                  list(path) + ["array" + "".join(["[{}]".format(i) for i in indices])])
                padding = (self.a - len(bytez) % self.a) % self.a
                bytez += bytearray(padding)

            return bytez

    def to_bytes_for_scalars(self, array, path):
        transfer_dtype = self.definition.numpy_dtype()

        if not isinstance(array, np.ndarray):
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(type(array), self.glsl_dtype(), np.ndarray, self.path_to_str(path)))
        if array.dtype.type not in self.definition.input_dtypes:
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(array.dtype, self.glsl_dtype(), self.definition.input_dtypes, self.path_to_str(path)))
        if tuple(array.shape) != self.shape():
            raise TypeError("Got shape {} for {} variable, expected {} at {}"
                            .format(array.shape, self.glsl_dtype(), self.shape(), self.path_to_str(path)))

        if self.layout == Layout.STD430:
            return array.astype(transfer_dtype).flatten().tobytes()

        else:
            p = (self.a - self.definition.alignment()) // self.definition.size()
            a = self.a // self.definition.size()

            array_padded = np.zeros(a * np.product(array.shape), dtype=transfer_dtype)
            mask = (np.arange(len(array_padded)) % a) < (a - p)
            array_padded[mask] = array.flatten()

            return array_padded.tobytes()

    def to_bytes_for_vectors(self, array, path):
        numpy_dtypes = self.definition.scalar.input_dtypes
        transfer_dtype = self.definition.scalar.numpy_dtype()
        shape = tuple(list(self.shape()) + [self.definition.length()])

        if not isinstance(array, np.ndarray):
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(type(array), self.glsl_dtype(), np.ndarray, self.path_to_str(path)))
        if array.dtype.type not in numpy_dtypes:
            raise TypeError("Got datatype {} for {} variable, expected {} at {}"
                            .format(array.dtype, self.glsl_dtype(), numpy_dtypes, self.path_to_str(path)))
        if tuple(array.shape) != shape:
            raise TypeError("Got shape {} for {} variable, expected {} at {}"
                            .format(array.shape, self.glsl_dtype(), shape, self.path_to_str(path)))

        p = (self.a - self.definition.size()) // self.definition.scalar.size()
        a = self.a // self.definition.scalar.size()

        array_padded = np.zeros(a * np.product(shape[:-1]), dtype=transfer_dtype)
        mask = (np.arange(len(array_padded)) % a) < (a - p)
        array_padded[mask] = array.flatten()

        return array_padded.tobytes()

    def to_bytes_for_matrices(self, array, path):
        numpy_dtype = self.definition.vector.scalar.numpy_dtype()
        shape = tuple(list(self.shape()) + list(self.definition.shape()))

        if not isinstance(array, np.ndarray):
            raise RuntimeError("Got datatype {} for {} variable, expected {} at {}"
                               .format(type(array), self.glsl_dtype(), np.ndarray, self.path_to_str(path)))
        if array.dtype != numpy_dtype:
            raise RuntimeError("Got datatype {} for {} variable, expected {} at {}"
                               .format(array.dtype, self.glsl_dtype(), numpy_dtype, self.path_to_str(path)))
        if tuple(array.shape) != shape:
            raise RuntimeError("Got shape {} for {} variable, expected {} at {}"
                               .format(array.shape, self.glsl_dtype(), shape, self.path_to_str(path)))

        # swap the last two dimensions if necessary
        if self.definition.order == Order.COLUMN_MAJOR:
            array = np.swapaxes(array, -2, -1)
            shape = array.shape

        p = (self.a - self.definition.vector.size()) // self.definition.vector.scalar.size()
        a = self.a // self.definition.vector.scalar.size()

        array_padded = np.zeros(a * np.product(shape[:-1]), dtype=array.dtype)
        mask = (np.arange(len(array_padded)) % a) < (a - p)
        array_padded[mask] = array.flatten()

        return array_padded.tobytes()

    def from_bytes(self, bytez):
        if isinstance(self.definition, Scalar):
            return self.from_bytes_for_scalars(bytez)

        elif isinstance(self.definition, Vector):
            return self.from_bytes_for_vectors(bytez)

        elif isinstance(self.definition, Matrix):
            return self.from_bytes_for_matrices(bytez)

        else:
            values = np.zeros(self.shape()).tolist()
            offset = 0
            size = self.definition.size()

            for indices in NdArray.iterate(self.shape()):
                NdArray.assign(values, indices, self.definition.from_bytes(bytez[offset:offset + size]))
                offset += size
                offset += (self.a - offset % self.a) % self.a  # bytes

            return values

    def from_bytes_for_scalars(self, bytez):
        if self.layout == Layout.STD430:
            array_flat = np.frombuffer(bytez, dtype=self.definition.numpy_dtype())

        else:
            p = (self.a - self.definition.alignment()) // self.definition.size()
            a = self.a // self.definition.size()

            array_padded = np.frombuffer(bytez, dtype=self.definition.numpy_dtype())
            mask = (np.arange(a * np.product(self.shape())) % a) < (a - p)
            array_flat = array_padded[mask]

        return array_flat.reshape(self.shape())

    def from_bytes_for_vectors(self, bytez):
        shape = tuple(list(self.shape()) + [self.definition.length()])
        p = (self.a - self.definition.size()) // self.definition.scalar.size()
        a = self.a // self.definition.scalar.size()

        array_padded = np.frombuffer(bytez, dtype=self.definition.scalar.numpy_dtype())
        mask = (np.arange(a * np.product(shape[:-1])) % a) < (a - p)
        return array_padded[mask].reshape(shape)

    def from_bytes_for_matrices(self, bytez):
        shape = tuple(list(self.shape()) + list(self.definition.shape()))
        p = (self.a - self.definition.vector.size()) // self.definition.vector.scalar.size()
        a = self.a // self.definition.vector.scalar.size()

        if self.definition.order == Order.COLUMN_MAJOR:
            shape = tuple(list(shape[:-2]) + [shape[-1], shape[-2]])

        array_padded = np.frombuffer(bytez, dtype=self.definition.vector.scalar.numpy_dtype())
        mask = (np.arange(a * np.product(shape[:-1])) % a) < (a - p)
        array = array_padded[mask].reshape(shape)

        if self.definition.order == Order.COLUMN_MAJOR:
            array = np.swapaxes(array, -2, -1)

        return array


class Struct(ByteRepresentation):

    def __init__(self, definitions, layout, member_names=None, type_name=None):
        super(Struct, self).__init__()
        self.definitions = definitions
        self.member_names = member_names or ([None] * len(definitions))
        self.type_name = type_name
        self.layout = layout  # precomputes alignment

        if len(set(definitions)) != len(definitions):
            raise BytesError("For struct members a definition object can not be used more than once")

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout

        # set children first
        for definition in self.definitions:
            if isinstance(definition, (Array, Matrix, Struct)):
                definition.layout = layout

        self.a = None
        self.precompute_alignment()

    def precompute_alignment(self):
        if self.layout in [Layout.STD140, Layout.STD430, Layout.STDXXX]:
            self.a = max([d.alignment() for d in self.definitions])

            if self.layout == Layout.STD140:
                self.a += (16 - self.a % 16) % 16

    def copy(self):
        return Struct([d.copy() for d in self.definitions], self.layout, self.member_names, self.type_name)

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

        a = self.alignment()
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
        if self.type_name is None:
            raise BytesError("Type name was not defined for structure")
        return self.type_name

    def __extend_path(self, path, member_index):
        if self.member_names[member_index]:
            step = "'{}'".format(self.member_names[member_index])
        else:
            step = str(member_index)
        return list(path) + ["member " + step]

    def compare(self, other, path=(), quiet=True):
        if not self.compare_type(type(self), type(other), path, quiet):
            return False
        if not self.compare_layout(self.layout, other.layout, path, quiet):
            return False

        definitions_expected = self.definitions
        definitions_other = other.definitions

        if len(definitions_expected) != len(definitions_other):
            if not quiet:
                raise TypeError("Expected {} members, but got {} members at {}".format(
                    len(definitions_expected), len(definitions_other), self.path_to_str(path)
                ))
            return False

        for i, (definition_expected, definition_other) in enumerate(zip(definitions_expected, definitions_other)):
            if not definition_expected.compare(definition_other, self.__extend_path(path, i), quiet):
                return False

        return True

    def to_bytes(self, values, path=()):
        bytez = bytearray()

        for i, d in enumerate(self.definitions):
            a = d.alignment()
            padding_before = (a - len(bytez) % a) % a
            bytez += bytearray(padding_before)
            bytez += d.to_bytes(values[d], self.__extend_path(path, i))

        # padding at the end
        a = self.alignment()
        padding_after = (a - len(bytez) % a) % a
        bytez += bytearray(padding_after)

        return bytez

    def from_bytes(self, bytez):
        values = {}
        offset = 0

        for i, d in enumerate(self.definitions):
            a = d.alignment()
            offset += (a - offset % a) % a  # padding before
            size = d.size()
            values[d] = d.from_bytes(bytez[offset:offset + size])
            offset += size

        # nothing needs to be done about the padding at the end
        return values


class ByteCache(object):

    def __init__(self, definition):
        if type(definition) != Struct:
            raise BytesError("ByteCaches can only be initialized with struct definitions")
        self.definition = definition
        self.values = {}
        self.dirty = False

        for i, d in enumerate(self.definition.definitions):
            value = None

            if isinstance(d, Struct):
                value = ByteCache(d)

            if Array.is_array_of_structs(d):
                value = np.zeros(d.shape()).tolist()

                for indices in NdArray.iterate(d.shape()):
                    NdArray.assign(value, indices, ByteCache(d.definition))

            self.values[d] = value

    def get_as_dict(self):
        data = {}

        for d in self.definition.definitions:
            value = self.values[d]

            if isinstance(d, Struct):
                value = self.values[d].get_as_dict()

            if Array.is_array_of_structs(d):
                value = np.zeros(d.shape()).tolist()

                for indices in NdArray.iterate(d.shape()):
                    cache = NdArray.get(self.values[d], indices)
                    NdArray.assign(value, indices, cache.get_as_dict())

            data[d] = value

        return data

    def set_from_dict(self, values):
        for d in self.definition.definitions:
            if isinstance(d, Struct):
                self.values[d].set_from_dict(values[d])

            elif isinstance(d, Array) and isinstance(d.definition, Struct):
                for indices in NdArray.iterate(d.shape()):
                    value = NdArray.get(values[d], indices)
                    cache = NdArray.get(self.values[d], indices)
                    cache.set_from_dict(value)

            else:
                self.values[d] = values[d]

    def set_dirty(self, dirty, include_children=True):
        self.dirty = dirty

        if include_children:
            for d in self.definition.definitions:
                value = self.values[d]

                if isinstance(d, Struct):
                    value.set_dirty(dirty, include_children)

                if Array.is_array_of_structs(d):
                    for indices in NdArray.iterate(d.shape()):
                        NdArray.get(value, indices).set_dirty(dirty, include_children)

    def is_dirty(self, include_children=True):
        if not include_children:
            return self.dirty

        dirty = self.dirty

        for d in self.definition.definitions:
            if dirty:
                return True

            value = self.values[d]

            if isinstance(d, Struct):
                dirty = dirty or value.is_dirty(include_children)

            if Array.is_array_of_structs(d):
                for indices in NdArray.iterate(d.shape()):
                    dirty = dirty or NdArray.get(value, indices).is_dirty(include_children)

        return dirty

    def __str__(self):
        s = self.__class__.__name__ + " around\n"
        s += str(self.definition)
        return s

    def __definition_from_key(self, key):
        if isinstance(key, int):
            return self.definition.definitions[key]

        if isinstance(key, str):
            if key not in self.definition.member_names:
                possible_reasons = [
                    "the definition was defined manually and the member names were not specified",
                    "the definition was taken from shader bytecode, but it did not contain the member names "
                    "(they are optional, the compiler must not put them in)"
                ]
                raise ValueError(
                    "Did not find {} in member names. In case this is not a typo please reference by index, member "
                    "names can be empty in following scenarios:\n{}".format(
                        key, "".join(["* {}\n".format(reason) for reason in possible_reasons])
                    )
                )
            index = self.definition.member_names.index(key)
            return self.definition.definitions[index]

        if isinstance(key, ByteRepresentation):
            return key

        raise ValueError("Invalid key")

    def __getitem__(self, key):
        return self.values[self.__definition_from_key(key)]

    def __setitem__(self, key, value):
        self.values[self.__definition_from_key(key)] = value
        self.dirty = True


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
