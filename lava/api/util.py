# -*- coding: UTF-8 -*-

from functools import reduce
import logging
import operator

import vulkan as vk

logger = logging.getLogger(__name__)


class Debugger(object):

    def __init__(self, instance, lvl=logging.INFO):
        self.instance = instance
        self.lvl = lvl
        self.handle = None
        self.attach()

    def __del__(self):
        self.detach()

    @staticmethod
    def log(flags, object_type, object, location, message_code, layer, message, user_data):
        lvl = logging.DEBUG

        if flags & vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
            lvl = logging.INFO
        if flags & vk.VK_DEBUG_REPORT_WARNING_BIT_EXT:
            lvl = logging.WARNING
        if flags & vk.VK_DEBUG_REPORT_ERROR_BIT_EXT:
            lvl = logging.ERROR

        logger.log(lvl, message[::])
        return 0

    def attach(self):
        all_bits = [vk.VK_DEBUG_REPORT_DEBUG_BIT_EXT, vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT,
                    vk.VK_DEBUG_REPORT_WARNING_BIT_EXT, vk.VK_DEBUG_REPORT_ERROR_BIT_EXT]
        all_lvls = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]

        create_info = vk.VkDebugReportCallbackCreateInfoEXT(
            flags=reduce(operator.or_, all_bits[all_lvls.index(self.lvl):]),
            pfnCallback=self.log
        )

        self.handle = self.instance.vkCreateDebugReportCallbackEXT(self.instance.handle, create_info, None)

    def detach(self):
        self.instance.vkDestroyDebugReportCallbackEXT(self.instance.handle, self.handle)


class Event(object):

    def __init__(self, device):
        self.device = device
        self.handle = vk.vkCreateEvent(self.device.handle, vk.VkEventCreateInfo(), None)

    def __del__(self):
        vk.vkDestroyEvent(self.device.handle, self.handle, None)


class Fence(object):

    def __init__(self, device, signalled):
        self.device = device

        if signalled:
            create_info = vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        else:
            create_info = vk.VkFenceCreateInfo()

        self.handle = vk.vkCreateFence(self.device.handle, create_info, None)

    def __del__(self):
        vk.vkDestroyFence(self.device.handle, self.handle, None)
