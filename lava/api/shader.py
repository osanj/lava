# -*- coding: UTF-8 -*-

import logging
import os
import platform
import subprocess

import vulkan as vk

logger = logging.getLogger(__name__)


class Shader(object):

    def __init__(self, device, path, entry_point="main"):
        self.device = device
        self.entry_point = entry_point
        with open(path, "rb") as f:
            self.byte_code = f.read()

            create_info = vk.VkShaderModuleCreateInfo(codeSize=len(self.byte_code), pCode=self.byte_code)
            self.handle = vk.vkCreateShaderModule(self.device.handle, create_info, None)

    def __del__(self):
        vk.vkDestroyShaderModule(self.device.handle, self.handle, None)

    def get_entry_point(self):
        return self.entry_point


class ByteCodeAnalyzer(object):

    def __init__(self, byte_code):
        self.byte_code = byte_code

    # get entry point
    # get binding
    # get std140
    # get struct layout ?


EXT_SPIR_V = ".spv"
ENV_SDK_NAME_LINUX = "VULKAN_SDK"


def compile_glsl(path, verbose=False):
    path_output = path + EXT_SPIR_V

    # Linux
    if platform.system() == "Linux":
        if ENV_SDK_NAME_LINUX not in os.environ:
            raise RuntimeError("Could not find environment variable {}".format(ENV_SDK_NAME_LINUX))

        path_compiler = os.path.join(os.environ["VULKAN_SDK"], "bin", "glslangValidator")
        cmd = [path_compiler, "-V", path, "-o", path_output]

        if verbose:
            result = subprocess.call(cmd, stderr=subprocess.STDOUT)
        else:
            dev_null = open(os.devnull, "w")
            result = subprocess.call(cmd, stdout=dev_null, stderr=subprocess.STDOUT)

        if result != 0:
            raise RuntimeError("Could not compile shader {}, try yourself with\n{}".format(path, " ".join(cmd)))

    # Windows
    elif platform.system() == "Windows":
        raise NotImplementedError()

    else:
        raise NotImplementedError()

    logger.info("Compiled shader {}".format(path))
    return path_output
