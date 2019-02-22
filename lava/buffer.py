# -*- coding: UTF-8 -*-

import logging

from lava.api.bytes import ByteCache, Struct
from lava.api.constants.vk import BufferUsage, MemoryType
from lava.api.memory import Buffer as _Buffer
from lava.api.shader import Shader


class Buffer(object):

    LOCATION_CPU = "CPU"
    LOCATION_GPU = "GPU"

    USAGE_STORAGE = BufferUsage.STORAGE_BUFFER
    USAGE_UNIFORM = BufferUsage.UNIFORM_BUFFER

    def __init__(self, session, block_definition, block_usage, location):
        if not isinstance(block_definition, Struct):
            raise RuntimeError("Block definitions must be structs")
        self.block_definition = block_definition
        self.block_usage = block_usage
        self.location = location

        self.session = session
        self.vulkan_buffer = None
        self.vulkan_memory = None
        self.session.register_buffer(self)
        self.in_sync = True

    def __del__(self):
        del self.vulkan_memory
        del self.vulkan_buffer

    def size(self):
        return self.block_definition.size()

    def get_block_definition(self):
        return self.block_definition

    def get_block_usage(self):
        return self.block_usage

    def get_location(self):
        return self.location

    def allocate(self):
        if self.vulkan_buffer is not None:
            raise RuntimeError("Buffer is already allocated")

        self.vulkan_buffer = _Buffer(self.session.device, self.size(), self.block_usage, self.session.queue_index)

        minimum_size = self.vulkan_buffer.get_memory_requirements()[0]
        memory_types = {self.LOCATION_CPU: MemoryType.CPU, self.LOCATION_GPU: MemoryType.GPU}[self.location]

        self.vulkan_memory = self.session.device.allocate_memory(memory_types, minimum_size)
        self.vulkan_buffer.bind_memory(self.vulkan_memory)


class BufferCPU(Buffer):

    def __init__(self, session, block_definition, block_usage):
        super(BufferCPU, self).__init__(session, block_definition, block_usage, Buffer.LOCATION_CPU)
        self.cache = ByteCache(self.block_definition)
        self.buffer_gpu = None

    def __del__(self):
        super(BufferCPU, self).__del__()

    @classmethod
    def from_shader(cls, session, shader, binding):
        block_definition = shader.get_block_definition(binding)
        block_usage = shader.get_block_usage(binding)
        return cls(session, block_definition, block_usage)

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.in_sync = False

    def is_synced(self):
        return self.in_sync

    def write(self):
        data = self.cache.get_as_dict()
        bytez = self.block_definition.to_bytes(data)
        self.vulkan_buffer.map(bytez)
        self.in_sync = True

    def read(self):
        with self.vulkan_buffer.mapped() as bytebuffer:
            bytez = bytebuffer[:]

        data = self.block_definition.from_bytes(bytez)
        self.cache.set_from_dict(data)
        self.in_sync = True

    def gpu(self):
        if self.buffer_gpu is None:
            self.buffer_gpu = BufferGPU(self.session, self.block_definition, self.block_usage)
        raise NotImplementedError()


class BufferGPU(Buffer):

    def __init__(self, session, block_definition, block_usage):
        super(BufferGPU, self).__init__(session, block_definition, block_usage, Buffer.LOCATION_CPU)
        self.buffer_cpu = None

    def cpu(self):
        if self.buffer_cpu is None:
            buffer_cpu = BufferCPU(self.session, self.block_definition, self.block_usage)
        raise NotImplementedError()
