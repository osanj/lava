# -*- coding: UTF-8 -*-

import lava.api.vulkan as vk
from lava.api.constants.vk import VALIDATION_LAYERS
from lava.api.util import Debugger, Destroyable


class Instance(Destroyable):

    def __init__(self, extensions=(), validation_lvl=None):
        super(Instance, self).__init__()
        self.validation_lvl = validation_lvl
        extensions = list(extensions)

        app_info = vk.VkApplicationInfo(
            pApplicationName=self.__class__.__name__,
            apiVersion=vk.VK_API_VERSION_1_0
        )

        if validation_lvl:
            create_info = vk.VkInstanceCreateInfo(
                pApplicationInfo=app_info,
                ppEnabledExtensionNames=extensions + [vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME],
                ppEnabledLayerNames=VALIDATION_LAYERS,
            )

        else:
            create_info = vk.VkInstanceCreateInfo(
                pApplicationInfo=app_info,
                ppEnabledExtensionNames=extensions,
                enabledLayerCount=0
            )

        self.handle = vk.vkCreateInstance(create_info, None)
        self.debugger = Debugger(self, lvl=validation_lvl) if validation_lvl else None

    def _destroy(self):
        if self.debugger:
            self.debugger.destroy()
        vk.vkDestroyInstance(self.handle, None)

    def __getattr__(self, item):
        # syntactic sugar to call vulkan instance functions as "pseudo methods" on this object, i.e.
        if item in vk._vulkan._instance_ext_funcs:
            def wrapper(*args, **kwargs):
                return vk.vkGetInstanceProcAddr(self.handle, item)(*args, **kwargs)
            return wrapper
        else:
            super(Instance, self).__getattribute__(item)
