# -*- coding: UTF-8 -*-

import logging

from lava.api.bytes import ByteCache, Struct
from lava.api.memory import Buffer as _Buffer
from lava.api.shader import Shader


class Buffer(object):

    LOCATION_CPU = "CPU"
    LOCATION_GPU = "GPU"

    TYPE_STORAGE = "STORAGE"
    TYPE_UNIFORM = "UNIFORM"

    def __init__(self, block_definition, buffer_type, location):
        if not isinstance(block_definition, Struct):
            raise RuntimeError("Block definitions must be structs")
        self.definition = block_definition
        self.buffer_type = buffer_type
        self.location = location
        self.cache = ByteCache(self.definition)

        self.session = None
        self.vulkan_buffer = None

    @classmethod
    def from_shader(cls, shader, binding, location):
        block_definition = shader.get_block_definition(binding)
        block_type = shader.get_block_buffer_type(binding)
        return cls(block_definition, block_type, location)

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

    def size(self):
        return self.definition.size()

    def __bind_to_session(self, session, vulkan_buffer):
        if self.session is not None:
            raise RuntimeError("Session should only be bound once")
        self.session = session
        self.vulkan_buffer = vulkan_buffer

    def __bind_to_shader(self, shader):
        pass

    def __write(self):
        data = self.cache.get_as_dict()
        bytez = self.definition.to_bytes(data)
        self.vulkan_buffer.map(bytez)

    def __read(self):
        with self.vulkan_buffer.mapped() as bytebuffer:
            bytez = bytebuffer[:]

        data = self.definition.from_bytes(bytez)
        self.cache.set_from_dict(data)


class PushConstants(object):

    def __init__(self):
        pass

