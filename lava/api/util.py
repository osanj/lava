# -*- coding: UTF-8 -*-

from functools import reduce
import itertools
import logging
import operator

import lava.api.vulkan as vk


class Destroyable(object):

    def __init__(self):
        self.__destroyed = False

    def __del__(self):
        if not self.__destroyed:
            self.destroy()

    def destroy(self):
        if not self.__destroyed:
            self._destroy()
            self.__destroyed = True

    def _destroy(self):
        raise NotImplementedError()


class LavaError(Exception):

    def __init__(self, message):
        super(LavaError, self).__init__(message)


class LavaUnsupportedError(LavaError):

    def __init__(self, message):
        super(LavaUnsupportedError, self).__init__(message)


class Debugger(Destroyable):

    def __init__(self, instance, lvl=logging.INFO):
        super(Debugger, self).__init__()
        self.instance = instance
        self.lvl = lvl
        self.handle = None
        self.history = []
        self.history_size = 100
        self.attach()

    def _destroy(self):
        self.detach()

    def log(self, flags, object_type, object, location, message_code, layer, message, user_data):
        lvl = logging.DEBUG

        if flags & vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
            lvl = logging.INFO
        if flags & vk.VK_DEBUG_REPORT_WARNING_BIT_EXT:
            lvl = logging.WARNING
        if flags & vk.VK_DEBUG_REPORT_ERROR_BIT_EXT:
            lvl = logging.ERROR

        message_str = message[::]
        self.history.insert(0, message_str)
        if len(self.history) > self.history_size:
            self.history = self.history[:self.history_size]

        print("[VULKAN] " + message_str)
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
        self.instance.vkDestroyDebugReportCallbackEXT(self.instance.handle, self.handle, None)


class Event(Destroyable):

    def __init__(self, device):
        super(Event, self).__init__()
        self.device = device
        self.handle = vk.vkCreateEvent(self.device.handle, vk.VkEventCreateInfo(), None)

    def _destroy(self):
        vk.vkDestroyEvent(self.device.handle, self.handle, None)


class Fence(Destroyable):

    def __init__(self, device, signalled):
        super(Fence, self).__init__()
        self.device = device

        if signalled:
            create_info = vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT)
        else:
            create_info = vk.VkFenceCreateInfo()

        self.handle = vk.vkCreateFence(self.device.handle, create_info, None)

    def _destroy(self):
        vk.vkDestroyFence(self.device.handle, self.handle, None)


class NdArray(object):

    @classmethod
    def iterate(cls, dims):
        return itertools.product(*[range(d) for d in dims])

    @classmethod
    def assign(cls, nd_array, indices, value):
        data = nd_array

        for index in indices[:-1]:
            data = data[index]

        data[indices[-1]] = value

    @classmethod
    def get(cls, nd_array, indices):
        data = nd_array

        for index in indices:
            data = data[index]

        return data
