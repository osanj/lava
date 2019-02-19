# -*- coding: UTF-8 -*-

import logging

from lava.api.constants.vk import MemoryType, QueueType
from lava.api.device import PhysicalDevice, Device
from lava.api.instance import Instance

logger = logging.getLogger(__name__)


class Session(object):

    def __init__(self, instance, physical_device, queue_index):
        self.instance = instance
        self.queue_index = queue_index
        self.device = Device(physical_device, [(QueueType.COMPUTE, self.queue_index)])

    def __del__(self):
        del self.device
        del self.instance

    @classmethod
    def discover(cls, validation_lvl=None):
        instance = Instance(validation_lvl=validation_lvl)
        physical_device = None

        for candidate in PhysicalDevice.all(instance):
            if candidate.supports_queue_type(QueueType.COMPUTE):
                physical_device = candidate
                break

        if physical_device is None:
            raise RuntimeError("Could not find a suitable device")

        queue_index = physical_device.get_queue_indices(QueueType.COMPUTE)[0]
        logger.info("Using {} for session".format(physical_device.get_name()))
        return cls(instance, physical_device, queue_index)

    def allocate_buffer(self, buffer):
        pass

    def allocate_push_constants(self, push_constants):
        pass
    #
    # def allocate_buffer(self, size):
    #     memory = self.device.allocate([MemoryType.HOST_VISIBLE, MemoryType.HOST_COHERENT], size)
    #     pass
    #
    # def allocate_gpu_buffer(self, size):
    #     memory = self.device.allocate([MemoryType.DEVICE_LOCAL], size)
    #     pass
    #
    # def allocate_uniform_buffer(self):
    #     pass
