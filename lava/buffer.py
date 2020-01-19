# -*- coding: UTF-8 -*-

import warnings

import numpy as np

from lava.api.bytes import Struct
from lava.api.cache import ByteCache
from lava.api.constants.spirv import Access
from lava.api.constants.vk import BufferUsage, MemoryType
from lava.api.memory import Buffer as _Buffer
from lava.api.pipeline import CopyOperation
from lava.api.util import Destroyable, LavaError, LavaUnsupportedError, mask_to_bounds

__all__ = ["BufferCPU", "BufferGPU", "StagedBuffer"]


class BufferInterface(Destroyable):

    USAGE_STORAGE = BufferUsage.STORAGE_BUFFER
    USAGE_UNIFORM = BufferUsage.UNIFORM_BUFFER

    SYNC_EAGER = "eager"
    SYNC_LAZY = "lazy"
    SYNC_MANUAL = "manual"
    SYNC_DEFAULT = SYNC_LAZY

    FLUSH_FULL = "full"
    FLUSH_PARTIAL = "partial"
    FLUSH_DEFAULT = FLUSH_FULL

    def __init__(self, session, block_definition, block_usage):
        super(BufferInterface, self).__init__()
        if not isinstance(block_definition, Struct):
            raise LavaError("Block definitions must be structs")
        self.session = session
        self.block_definition = block_definition
        self.block_usage = block_usage

    def _destroy(self):
        raise NotImplementedError()

    def size(self):
        return self.block_definition.size()

    def get_block_definition(self):
        return self.block_definition

    def get_block_usage(self):
        return self.block_usage

    def get_vulkan_buffer(self):
        raise NotImplementedError()

    def before_stage(self, stage, binding, access_modifier):
        pass

    def after_stage(self, stage, binding, access_modifier):
        pass

    def flush(self):
        pass

    def fetch(self):
        pass

    def __getitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()


class Buffer(BufferInterface):

    LOCATION_CPU = "CPU"
    LOCATION_GPU = "GPU"

    def __init__(self, session, block_definition, block_usage, location):
        super(Buffer, self).__init__(session, block_definition, block_usage)
        self.location = location
        self.vulkan_buffer = None
        self.vulkan_memory = None
        self.session.register_buffer(self)
        self.allocate()
        self.copy_operation = CopyOperation(self.session.device, self.session.queue_index)

    def _destroy(self):
        self.vulkan_memory.destroy()
        self.vulkan_buffer.destroy()
        self.copy_operation.destroy()

    def allocate(self):
        if self.vulkan_buffer is not None:
            raise LavaError("Buffer is already allocated")

        self.vulkan_buffer = _Buffer(self.session.device, self.size(), self.block_usage, self.session.queue_index)

        minimum_size, _, supported_memory_indices = self.vulkan_buffer.get_memory_requirements()
        memory_types = {self.LOCATION_CPU: MemoryType.CPU, self.LOCATION_GPU: MemoryType.GPU}[self.location]

        self.vulkan_memory = self.session.device.allocate_memory(memory_types, minimum_size, supported_memory_indices)
        self.vulkan_buffer.bind_memory(self.vulkan_memory)

    def get_vulkan_buffer(self):
        return self.vulkan_buffer

    def get_location(self):
        return self.location

    def copy_to(self, other, bounds=None):
        self.copy_operation.record(self.vulkan_buffer, other.vulkan_buffer, bounds)
        self.copy_operation.run_and_wait()


class BufferCPU(Buffer):

    def __init__(self, session, block_definition, block_usage, sync_mode=BufferInterface.SYNC_DEFAULT,
                 flush_mode=BufferInterface.FLUSH_DEFAULT):
        super(BufferCPU, self).__init__(session, block_definition, block_usage, Buffer.LOCATION_CPU)
        self.cache = ByteCache(self.block_definition)
        self.cache.set_defaults()
        self.cache.set_dirty(False)

        self.sync_mode = sync_mode
        self.flush_mode = flush_mode
        self.fresh_bytez = False
        self.first_flush = True

    @classmethod
    def from_shader(cls, session, shader, binding, sync_mode=BufferInterface.SYNC_DEFAULT,
                    flush_mode=BufferInterface.FLUSH_DEFAULT):
        block_definition = shader.get_block_definition(binding)
        block_usage = shader.get_block_usage(binding)
        return cls(session, block_definition, block_usage, sync_mode, flush_mode)

    def before_stage(self, stage, binding, access_modifier):
        if access_modifier in [Access.READ_ONLY, Access.READ_WRITE]:
            if self.sync_mode is self.SYNC_LAZY and self.cache.is_dirty():
                self.flush()

    def after_stage(self, stage, binding, access_modifier):
        if access_modifier in [Access.WRITE_ONLY, Access.READ_WRITE]:
            self.fresh_bytez = True
            if self.sync_mode is self.SYNC_EAGER:
                self.fetch()

    def __getitem__(self, key):
        if self.fresh_bytez and self.sync_mode is self.SYNC_LAZY:
            self.fetch()
        return self.cache[key]

    def __setitem__(self, key, value):
        self.cache[key] = value

        if self.sync_mode is self.SYNC_EAGER:
            self.flush()

    def is_synced(self):
        return not self.fresh_bytez and not self.cache.is_dirty()

    def flush(self):
        if self.fresh_bytez:
            warnings.warn("Buffer contains (probably) data which is not in the cache, it will be overwritten",
                          RuntimeWarning)

        data = self.cache.get_as_dict()
        bytez = self.block_definition.to_bytes(data)

        if self.flush_mode is self.FLUSH_PARTIAL and not self.first_flush:
            with self.vulkan_buffer.mapped() as bytez_current:
                dirty_mask = np.array(bytez) != bytez_current
                bytez_current[:] = bytez
        else:
            dirty_mask = np.ones(len(bytez), dtype=bool)
            self.vulkan_buffer.map(bytez)

        self.first_flush = False
        self.cache.set_dirty(False)
        self.fresh_bytez = False

        return dirty_mask

    def fetch(self):
        if self.cache.is_dirty():
            warnings.warn("Cache contains data which is not in the buffer, it will by overwritten", RuntimeWarning)

        with self.vulkan_buffer.mapped() as bytebuffer:
            bytez = bytebuffer[:]
        data = self.block_definition.from_bytes(bytez)
        self.cache.set_from_dict(data)
        self.cache.set_dirty(False)
        self.fresh_bytez = False


class BufferGPU(Buffer):

    def __init__(self, session, block_definition, block_usage):
        super(BufferGPU, self).__init__(session, block_definition, block_usage, Buffer.LOCATION_GPU)
        self.buffer_cpu = None
        self.copy_operation = CopyOperation(self.session.device, self.session.queue_index)

    @classmethod
    def from_shader(cls, session, shader, binding):
        block_definition = shader.get_block_definition(binding)
        block_usage = shader.get_block_usage(binding)
        return cls(session, block_definition, block_usage)

    def __getitem__(self, key):
        raise LavaUnsupportedError("Unsupported, the only way to read gpu buffers, is to copy them to a cpu buffer")

    def __setitem__(self, key, value):
        raise LavaUnsupportedError("Unsupported, the only way to write gpu buffers, is to copy from a cpu buffer")


class StagedBuffer(BufferInterface):

    def __init__(self, session, block_definition, block_usage, sync_mode=BufferInterface.SYNC_DEFAULT,
                 flush_mode=BufferInterface.FLUSH_DEFAULT):
        super(StagedBuffer, self).__init__(session, block_definition, block_usage)
        self.buffer_cpu = BufferCPU(session, block_definition, block_usage, sync_mode, flush_mode)
        self.buffer_gpu = BufferGPU(session, block_definition, block_usage)
        self.sync_mode = sync_mode
        self.flush_mode = flush_mode
        self.fresh_bytez = False

    def _destroy(self):
        self.buffer_cpu.destroy()
        self.buffer_gpu.destroy()

    @classmethod
    def from_shader(cls, session, shader, binding, sync_mode=BufferInterface.SYNC_DEFAULT,
                    flush_mode=BufferInterface.FLUSH_DEFAULT):
        block_definition = shader.get_block_definition(binding)
        block_usage = shader.get_block_usage(binding)
        return cls(session, block_definition, block_usage, sync_mode, flush_mode)

    def get_vulkan_buffer(self):
        return self.buffer_gpu.get_vulkan_buffer()

    def before_stage(self, stage, binding, access_modifier):
        if access_modifier in [Access.READ_ONLY, Access.READ_WRITE]:
            if self.sync_mode is self.SYNC_LAZY and self.buffer_cpu.cache.is_dirty():
                self.flush()

    def after_stage(self, stage, binding, access_modifier):
        if access_modifier in [Access.WRITE_ONLY, Access.READ_WRITE]:
            self.fresh_bytez = True
            if self.sync_mode is self.SYNC_EAGER:
                self.fetch()

    def __getitem__(self, key):
        if self.fresh_bytez:
            self.fetch()
        return self.buffer_cpu[key]

    def __setitem__(self, key, value):
        self.buffer_cpu[key] = value

        if self.sync_mode is self.SYNC_EAGER:
            self.flush()

    def is_synced(self):
        return not self.fresh_bytez and self.buffer_cpu.is_synced()

    def flush(self):
        bounds = None
        if self.buffer_cpu.cache.is_dirty():
            mask = self.buffer_cpu.flush()
            if self.flush_mode is self.FLUSH_PARTIAL:
                bounds = mask_to_bounds(mask)
        self.buffer_cpu.copy_to(self.buffer_gpu, bounds=bounds)
        self.fresh_bytez = False

    def fetch(self):
        self.buffer_gpu.copy_to(self.buffer_cpu)
        self.buffer_cpu.fetch()
        self.fresh_bytez = False
