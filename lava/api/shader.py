# -*- coding: UTF-8 -*-

import lava.api.vulkan as vk
from lava.api.bytecode.logical import ByteCode
from lava.api.bytecode.physical import ByteCodeData
from lava.api.util import Destroyable


class Shader(Destroyable):

    def __init__(self, device, bytez, entry_point=None):
        super(Shader, self).__init__()
        self.device = device
        self.handle = vk.vkCreateShaderModule(self.device.handle, vk.VkShaderModuleCreateInfo(codeSize=len(bytez),
                                                                                              pCode=bytez), None)
        self.byte_code_data = ByteCodeData(bytez)
        self.byte_code = None

        self.entry_point, self.entry_point_index = ByteCode.check_entry_point(self.byte_code_data, entry_point)
        self.local_size = ByteCode.check_local_size(self.byte_code_data, self.entry_point_index)

    @classmethod
    def from_file(cls, device, path, entry_point=None):
        with open(path, "rb") as f:
            bytez = f.read()
        return cls(device, bytez, entry_point)

    def inspect(self):
        self.byte_code = ByteCode(self.byte_code_data, self.entry_point)

    @property
    def code(self):
        if self.byte_code is None:
            raise RuntimeError("Shader bytecode was not inspected yet")
        return self.byte_code

    def get_entry_point(self):
        return self.entry_point

    def get_local_size(self):
        return self.local_size

    def _destroy(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)
