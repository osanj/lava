# -*- coding: UTF-8 -*-

import atexit
import logging
import os
import platform
import warnings


__version__ = "0.3.1"

ENV_VAR_SDK = "VULKAN_SDK"
ENV_VAR_LAYER_PATH = "VK_LAYER_PATH"

VALIDATION_LEVEL_DEBUG = logging.DEBUG
VALIDATION_LEVEL_INFO = logging.INFO
VALIDATION_LEVEL_WARNING = logging.WARNING
VALIDATION_LEVEL_ERROR = logging.ERROR

VALIDATION_LEVEL = None


__instance = None
__instance_usages = 0
__devices = []


try:
    import vulkan as vk
    __error = None
except OSError as e:
    __error = e

if ENV_VAR_SDK not in os.environ:
    __error = ImportError("{} environment variable not found".format(ENV_VAR_SDK))


def __initialize():
    global __error, __instance, __instance_usages, __devices, VALIDATION_LEVEL

    if __error:
        return

    from .api.constants.vk import QueueType
    from .api.device import PhysicalDevice
    from .api.instance import Instance

    if VALIDATION_LEVEL is not None and platform.system() == "Linux":
        if ENV_VAR_LAYER_PATH not in os.environ:
            __error = ImportError("{} environment variable not found (required for validations)"
                                  .format(ENV_VAR_LAYER_PATH))
            return

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
        __error = ImportError("Could not initialize {}".format(__name__))


def initialized():
    global __error
    return __error is None


def instance():
    global __error, __instance, __instance_usages, __devices, VALIDATION_LEVEL

    if __error:
        raise __error

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
