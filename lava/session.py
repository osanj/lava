# -*- coding: UTF-8 -*-

import logging

import lava
from lava.api.constants.vk import QueueType
from lava.api.device import Device

logger = logging.getLogger(__name__)

__all__ = ["Session"]


class Session(object):

    def __init__(self, physical_device, queue_index=None):
        self.instance = lava.instance()  # validation level might has been changed
        self.queue_index = queue_index or physical_device.get_queue_indices(QueueType.COMPUTE)[0]
        self.device = Device(physical_device, [(QueueType.COMPUTE, self.queue_index)],
                             validation_lvl=lava.VALIDATION_LEVEL)
        self.buffers = []

    def __del__(self):
        del self.buffers
        del self.device

    def register_buffer(self, buffer):
        if buffer not in self.buffers:
            self.buffers.append(buffer)
