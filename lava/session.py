# -*- coding: UTF-8 -*-

import lava
from lava.api.constants.vk import QueueType
from lava.api.device import Device
from lava.api.util import Destroyable

__all__ = ["Session"]

sessions = set()


class Session(Destroyable):

    def __init__(self, physical_device, queue_index=None):
        super(Session, self).__init__()

        self.instance = lava.instance()  # validation level might has been changed
        if physical_device not in lava.devices():
            raise RuntimeError("Provided invalid / outdated device object")

        self.queue_index = queue_index or physical_device.get_queue_indices(QueueType.COMPUTE)[0]
        self.device = Device(physical_device, [(QueueType.COMPUTE, self.queue_index)],
                             validation_lvl=lava.VALIDATION_LEVEL)

        self.buffers = set()
        self.shaders = set()
        self.stages = set()

        sessions.add(self)

    def _destroy(self):
        for stage in self.stages:
            stage.destroy()
        for shader in self.shaders:
            shader.destroy()
        for buffer in self.buffers:
            buffer.destroy()
        self.device.destroy()

    def register_buffer(self, buffer):
        self.buffers.add(buffer)

    def register_shader(self, shader):
        self.shaders.add(shader)

    def register_stage(self, stage):
        self.stages.add(stage)
