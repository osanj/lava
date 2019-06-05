# -*- coding: UTF-8 -*-

import lava.api.vulkan as vk


VALIDATION_LAYERS = ["VK_LAYER_LUNARG_standard_validation"]
SHADER_ENTRY = "main"
TIMEOUT_FOREVER = 0xFFFFFFFFFFFFFFFF


class VulkanEnum(object):

    _MAP = ()

    @classmethod
    def map(cls, key, keys, values):
        return values[keys.index(key)]

    @classmethod
    def to_vulkan(cls, key):
        keys, values = zip(*cls._MAP)
        return cls.map(key, keys, values)

    @classmethod
    def from_vulkan(cls, key):
        values, keys = zip(*cls._MAP)
        return cls.map(key, keys, values)

    @classmethod
    def keys(cls):
        return list(zip(*cls._MAP))[0]


class DeviceType(VulkanEnum):

    CPU = "CPU"
    DISCRETE_GPU = "DISCRETE_CPU"
    INTEGRATED_GPU = "INTEGRATED_GPU"
    VIRTUAL_GPU = "VIRTUAL_GPU"
    OTHER = "OTHER"

    _MAP = (
        (CPU, vk.VK_PHYSICAL_DEVICE_TYPE_CPU),
        (DISCRETE_GPU, vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU),
        (INTEGRATED_GPU, vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU),
        (VIRTUAL_GPU, vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
    )


class QueueType(VulkanEnum):

    GRAPHICS = "GRAPHICS"
    COMPUTE = "COMPUTE"
    TRANSFER = "TRANSFER"
    SPARSE_BINDING = "SPARSE_BINDING"

    _MAP = (
        (GRAPHICS, vk.VK_QUEUE_GRAPHICS_BIT),
        (COMPUTE, vk.VK_QUEUE_COMPUTE_BIT),
        (TRANSFER, vk.VK_QUEUE_TRANSFER_BIT),
        (SPARSE_BINDING, vk.VK_QUEUE_SPARSE_BINDING_BIT),
    )


class MemoryType(VulkanEnum):
    # https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkMemoryPropertyFlagBits.html

    DEVICE_LOCAL = "DEVICE_LOCAL"
    HOST_VISIBLE = "HOST_VISIBLE"
    HOST_COHERENT = "HOST_COHERENT"
    HOST_CACHED = "HOST_CACHED"
    LAZILY_ALLOCATED = "LAZILY_ALLOCATED"
    # PROTECTED = "PROTECTED"

    CPU = [HOST_COHERENT, HOST_VISIBLE]
    GPU = [DEVICE_LOCAL]

    _MAP = (
        (DEVICE_LOCAL, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        (HOST_VISIBLE, vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
        (HOST_COHERENT, vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        (HOST_CACHED, vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
        (LAZILY_ALLOCATED, vk.VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT),
    )


class BufferUsage(VulkanEnum):
    # https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkBufferUsageFlagBits.html

    UNIFORM_BUFFER = "UNIFORM_BUFFER"
    STORAGE_BUFFER = "STORAGE_BUFFER"
    TRANSFER_SRC = "TRANSFER_SRC"
    TRANSFER_DST = "TRANSFER_DST"

    _MAP = (
        (UNIFORM_BUFFER, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
        (STORAGE_BUFFER, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        (TRANSFER_SRC, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
        (TRANSFER_DST, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    )


class DescriptorType(VulkanEnum):
    # https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkDescriptorType.html

    SAMPLER = "SAMPLER"
    UNIFORM_BUFFER = "UNIFORM_BUFFER"
    STORAGE_BUFFER = "STORAGE_BUFFER"

    _MAP = (
        (SAMPLER, vk.VK_DESCRIPTOR_TYPE_SAMPLER),
        (UNIFORM_BUFFER, vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
        (STORAGE_BUFFER, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
    )


class CommandBufferUsage(VulkanEnum):
    # https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkCommandBufferUsageFlagBits.html

    ONE_TIME_SUBMIT = "ONE_TIME_SUBMIT"
    SIMULTANEOUS_USE = "SIMULTANEOUS"

    _MAP = (
        (ONE_TIME_SUBMIT, vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),
        (SIMULTANEOUS_USE, vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)
    )




