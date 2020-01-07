# -*- coding: UTF-8 -*-

import lava.api.vulkan as vk
from lava.api.bytecode.instruction import ByteCodeData
from lava.api.bytecode.interpreter import ByteCode
from lava.api.util import Destroyable


class Shader(Destroyable):

    def __init__(self, device, bytez):
        super(Shader, self).__init__()
        self.device = device
        self.handle = vk.vkCreateShaderModule(self.device.handle, vk.VkShaderModuleCreateInfo(codeSize=len(bytez),
                                                                                              pCode=bytez), None)
        self.byte_code_data = ByteCodeData(bytez)
        self.byte_code = None

    @classmethod
    def from_file(cls, device, path):
        with open(path, "rb") as f:
            bytez = f.read()
        return cls(device, bytez)

    def inspect(self, entry_point=None):
        self.byte_code = ByteCode(self.byte_code_data, entry_point)

    def get_entry_point(self):
        return self.byte_code.get_entry_point()

    def get_bindings(self):
        return self.byte_code.get_bindings()

    def get_block_definition(self, binding):
        return self.byte_code.get_block_definition(binding)

    def get_block_usage(self, binding):
        return self.byte_code.get_block_usage(binding)

    def get_local_size(self):
        return self.byte_code.get_local_size()

    def get_block_access(self, binding):
        return self.byte_code.get_block_access(binding)

    def _destroy(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)
