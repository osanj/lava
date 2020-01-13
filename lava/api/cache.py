# -*- coding: UTF-8 -*-

import numpy as np

from lava.api.bytes import BytesError, ByteRepresentation, Scalar, Vector, Matrix, Array, Struct
from lava.api.util import NdArray


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

    def set_defaults(self):
        for d in self.definition.definitions:
            if isinstance(d, Struct):
                self.values[d].set_defaults()

            elif isinstance(d, Array):
                if isinstance(d.definition, Struct):
                    for indices in NdArray.iterate(d.shape()):
                        NdArray.get(self.values[d], indices).set_defaults()

                elif isinstance(d.definition, (Scalar, Vector, Matrix)):
                    self.values[d] = np.zeros(d.shape_extended(), dtype=d.definition.numpy_dtype())

                else:
                    raise RuntimeError()

            elif isinstance(d, Matrix):
                self.values[d] = np.zeros(d.shape(), dtype=d.numpy_dtype())

            elif isinstance(d, Vector):
                self.values[d] = np.zeros(d.length(), dtype=d.numpy_dtype())

            elif isinstance(d, Scalar):
                self.values[d] = d.numpy_dtype()(0)

            else:
                raise RuntimeError()

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
