# -*- coding: UTF-8 -*-

import contextlib

import vulkan as vk

from lava.api.constants import BufferUsage, DescriptorType

# PushConstants
# SampledImage
# UniformBuffer
# StorageImage
# StorageBuffer

# http://vulkan-spec-chunked.ahcox.com/ch09.html


#         desired_flag = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
#         mem_requirements_ping = vk.vkGetImageMemoryRequirements(self.device.handle, self.handle_ping)
#         mem_requirements_pong = vk.vkGetImageMemoryRequirements(self.device.handle, self.handle_pong)
#         mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.device.physical_device.handle)
#         mem_idx_invalid = 0xFFFFFFFF
#         mem_idx = mem_idx_invalid
#
#         for i in range(mem_properties.memoryTypeCount):
#             if ((mem_requirements_ping.memoryTypeBits & (2 ** i))
#                     and (mem_requirements_pong.memoryTypeBits & (2 ** i))
#                     and ((mem_properties.memoryTypes[i].propertyFlags & desired_flag) == desired_flag)):
#                 mem_idx = i
#
#         if mem_idx == mem_idx_invalid:
#             raise RuntimeError("No suitable memory found for allocation")
#
#         ping_size = mem_requirements_ping.size
#         pong_align = mem_requirements_pong.alignment
#
#         offset = ping_size + pong_align - ping_size % pong_align
#         size = offset + mem_requirements_pong.size
#
#         self.memory_handle = vk.vkAllocateMemory(self.device.handle, vk.VkMemoryAllocateInfo(
#             allocationSize=size, memoryTypeIndex=mem_idx
#         ), None)


class Memory(object):

    def __init__(self, device, memory_type_index, size):
        self.device = device
        self.memory_type_index = memory_type_index
        self.size = size
        self.memory_obj = None

        allocate_info = vk.VkMemoryAllocateInfo(allocationSize=size, memoryTypeIndex=memory_type_index)
        self.handle = vk.vkAllocateMemory(self.device.handle, allocate_info, None)

        # try:
        #     self.handle = vk.vkAllocateMemory(self.device.handle, allocate_info, None)
        # except vk.VkErrorOutOfDeviceMemory:
        #     raise RuntimeError("Device {} is out of memory".format(self.device.get_physical_device().get_name()))
        # except vk.VkErrorOutOfHostMemory:
        #     raise RuntimeError("Host is out of memory")
        # except vk.VkErrorTooManyObjects:
        #     raise RuntimeError("Too many allocations")

        # VK_ERROR_INVALID_EXTERNAL_HANDLE

    def __del__(self):
        vk.vkFreeMemory(self.device.handle, self.handle, None)

    def get_size(self):
        return self.size

    def get_device(self):
        return self.device

    def get_memory_obj(self):
        return self.memory_obj

    @contextlib.contextmanager
    def mapped(self, size=-1, offset=0):
        if size == -1:
            size = self.size
        # https://cffi.readthedocs.io/en/latest/ref.html#ffi-buffer-ffi-from-buffer
        bytebuffer = vk.vkMapMemory(self.device.handle, self.handle, offset, size, flags=0)
        yield bytebuffer
        vk.vkUnmapMemory(self.device.handle, self.handle)


class MemoryObject(object):

    def __init__(self, device, size):
        self.device = device
        self.size = size
        self.memory = None
        self.memory_offset = None

    def get_memory(self):
        return self.memory

    def get_size(self):
        return self.size

    def bind_memory(self, memory, memory_offset):
        if memory.get_memory_obj() is not None:
            raise RuntimeError("Memory is already bound")
        self.memory = memory
        self.memory_offset = memory_offset

    def descriptor_type(self):
        raise NotImplementedError()

    def descriptor_set_layout(self, binding):
        raise NotImplementedError()

    def write_descriptor_set(self, descriptor_set_handle, binding):
        raise NotImplementedError()


class Buffer(MemoryObject):

    def __init__(self, device, size, usage, queue_index):
        super(Buffer, self).__init__(device, size)
        self.usage = usage
        self.queue_index = queue_index

        create_info = vk.VkBufferCreateInfo(
            size=self.size, usage=BufferUsage.to_vulkan(usage), sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            pQueueFamilyIndices=[queue_index]
        )

        self.handle = vk.vkCreateBuffer(self.device.handle, create_info, None)

    def __del__(self):
        vk.vkDestroyBuffer(self.device.handle, self.handle, None)

    def get_memory_requirements(self):
        requirements = vk.vkGetBufferMemoryRequirements(self.device.handle, self.handle)
        return requirements.size, requirements.alignment

    def bind_memory(self, memory, offset=0):
        super(Buffer, self).bind_memory(memory, offset)
        vk.vkBindBufferMemory(device=self.device.handle, buffer=self.handle, memory=self.memory.handle,
                              memoryOffset=offset)

    @contextlib.contextmanager
    def mapped(self):
        with self.memory.mapped(self.size, self.memory_offset) as bytebuffer:
            yield bytebuffer

    def map(self, bytez):
        with self.mapped() as bytebuffer:
            bytebuffer[:] = bytez

    def descriptor_type(self):
        if self.usage == BufferUsage.STORAGE_BUFFER:
            descriptor_type = DescriptorType.STORAGE_BUFFER
        elif self.usage == BufferUsage.UNIFORM_BUFFER:
            descriptor_type = DescriptorType.UNIFORM_BUFFER
        else:
            raise NotImplementedError("Buffer usage {} is not supported".format(self.usage))

        return descriptor_type

    def descriptor_set_layout(self, binding):
        descriptor_type = self.descriptor_type()
        return vk.VkDescriptorSetLayoutBinding(binding=binding,
                                               descriptorType=DescriptorType.to_vulkan(descriptor_type),
                                               descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)

    def write_descriptor_set(self, descriptor_set_handle, binding):
        descriptor_type = self.descriptor_type()
        # buffer_info = vk.VkDescriptorBufferInfo(self.handle, 0, vk.VK_WHOLE_SIZE)
        buffer_info = vk.VkDescriptorBufferInfo(self.handle, 0, self.size)
        return vk.VkWriteDescriptorSet(dstSet=descriptor_set_handle, dstBinding=binding,
                                       descriptorType=DescriptorType.to_vulkan(descriptor_type),
                                       pBufferInfo=[buffer_info])


