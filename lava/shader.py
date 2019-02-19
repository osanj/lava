# -*- coding: UTF-8 -*-

import logging

from lava.api.constants.vk import BufferUsage
from lava.api.shader import Shader as _Shader
from lava.buffer import Buffer


class Shader(object):

    def __init__(self):
        self.vulkan_shader = _Shader()

    def get_bindings(self):
        return self.vulkan_shader.get_bindings()

    def get_block_definition(self, binding):
        return self.vulkan_shader.get_block_definition(binding)

    def get_block_buffer_type(self, binding):
        usage = self.vulkan_shader.get_block_usage(binding)
        if usage == BufferUsage.STORAGE_BUFFER:
            return Buffer.TYPE_STORAGE
        if usage == BufferUsage.UNIFORM_BUFFER:
            return Buffer.TYPE_UNIFORM
        raise RuntimeError("ApiError")
