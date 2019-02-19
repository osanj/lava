# -*- coding: UTF-8 -*-

import logging

from lava.api.shader import Shader as _Shader


class Shader(object):

    def __init__(self, session, path, entry_point=None):
        self.session = session
        self.vulkan_shader = _Shader(self.session.device, path, entry_point)
        self.vulkan_shader.inspect()

    def get_bindings(self):
        return self.vulkan_shader.get_bindings()

    def get_block_definition(self, binding):
        return self.vulkan_shader.get_block_definition(binding)

    def get_block_usage(self, binding):
        return self.vulkan_shader.get_block_usage(binding)
