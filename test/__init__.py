# -*- coding: UTF-8 -*-

import os
import platform
import tempfile


from lava.session import Session


class TestSession(Session):

    def __init__(self):
        pass


class TestUtil(object):

    @classmethod
    def set_vulkan_environment_variables(cls):
        env_data = {
            "PATH": "/home/jonas/Documents/VulkanSdk/1.1.92.1/x86_64/bin",
            "LD_LIBRARY_PATH": "/home/jonas/Documents/VulkanSdk/1.1.92.1/x86_64/lib",
            "VULKAN_SDK": "/home/jonas/Documents/VulkanSdk/1.1.92.1/x86_64",
            "VK_LAYER_PATH": "/home/jonas/Documents/VulkanSdk/1.1.92.1/x86_64/etc/explicit_layer.d"
        }

        for key, value in env_data.iteritems():
            if key in os.environ:
                if value not in os.environ[key]:
                    os.environ[key] += os.pathsep + value
            else:
                os.environ[key] = value

    @classmethod
    def write_to_temp_file(cls, txt, mode="w", prefix="lavatest-", suffix=""):
        if platform.system() != "Linux":
            raise NotImplementedError()

        with tempfile.NamedTemporaryFile(mode=mode, prefix=prefix, suffix=suffix, delete=False) as f:
            f.write(txt)
            return f.name
