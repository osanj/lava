# -*- coding: UTF-8 -*-

import atexit
import logging
import os
import warnings


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
    from lava.api.constants.vk import QueueType
    from lava.api.device import PhysicalDevice
    from lava.api.instance import Instance
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

        del QueueType, PhysicalDevice, Instance

    except:
        return False
    else:
        return True


def instance():
    global __instance, __instance_usages, __devices, VALIDATION_LEVEL

    if __instance.validation_lvl != VALIDATION_LEVEL:
        if __instance_usages > 0:
            __instance_usages = 0
            warnings.warn("Recreating Vulkan instance with new validation level, any previously created sessions, "
                          "devices, etc. will be no longer usable", UserWarning)

        del __devices
        del __instance
        __initialize()

    __instance_usages += 1
    return __instance


def devices():
    instance()  # validation level might has been changed
    return __devices


@atexit.register
def __cleanup():
    global __instance, __devices
    del __devices
    del __instance


if not __initialize():
    raise ImportError("Could not initialize {}".format(__name__))

from lava.buffer import *
from lava.session import *
from lava.shader import *
from lava.util import *
