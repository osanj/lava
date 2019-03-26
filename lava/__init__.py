# -*- coding: UTF-8 -*-

import atexit
import logging
import os
import warnings

from future.utils import raise_with_traceback


__version__ = "0.2.0"


ENV_VAR_SDK = "VULKAN_SDK"
ENV_VAR_LAYER_PATH = "VK_LAYER_PATH"

if ENV_VAR_SDK not in os.environ:
    raise ImportError("{} environment variable not found".format(ENV_VAR_SDK))

if ENV_VAR_LAYER_PATH not in os.environ:
    raise ImportError("{} environment variable not found".format(ENV_VAR_LAYER_PATH))


VALIDATION_LEVEL_DEBUG = logging.DEBUG
VALIDATION_LEVEL_INFO = logging.INFO
VALIDATION_LEVEL_WARNING = logging.WARNING
VALIDATION_LEVEL_ERROR = logging.ERROR

VALIDATION_LEVEL = None


__instance = None
__instance_usages = 0
__devices = []


def __initialize():
    from .api.constants.vk import QueueType
    from .api.device import PhysicalDevice
    from .api.instance import Instance
    global __instance, __instance_usages, __devices, VALIDATION_LEVEL

    try:
        __instance = Instance(validation_lvl=VALIDATION_LEVEL)
        __instance_usages = 0
        __devices = []

        for candidate in PhysicalDevice.all(__instance):
            if candidate.supports_queue_type(QueueType.COMPUTE):
                __devices.append(candidate)

        if len(__devices) == 0:
            warnings.warn("Did not find any suitable device", RuntimeWarning)

    except:
        raise_with_traceback(ImportError("Could not initialize {}".format(__name__)))


def instance():
    global __instance, __instance_usages, __devices, VALIDATION_LEVEL

    if __instance.validation_lvl != VALIDATION_LEVEL:
        if __instance_usages > 0:
            __instance_usages = 0
            warnings.warn("Recreating Vulkan instance with new validation level, any previously created sessions, "
                          "devices, etc. will be no longer usable", UserWarning)

        __cleanup()
        __initialize()

    __instance_usages += 1
    return __instance


def devices():
    instance()  # validation level might has been changed
    return __devices


@atexit.register
def __cleanup():
    global __instance, __devices

    if __instance is not None:
        from .session import sessions
        for sess in sessions:
            sess.destroy()

        del __devices
        __instance.destroy()


__initialize()

from .buffer import *
from .session import *
from .shader import *
from .util import *
