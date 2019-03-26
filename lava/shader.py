# -*- coding: UTF-8 -*-

import warnings

from future.utils import raise_with_traceback

from lava.api.constants.vk import BufferUsage
from lava.api.pipeline import ShaderOperation, Pipeline
from lava.api.shader import Shader as _Shader
from lava.api.util import Destroyable, LavaError

__all__ = ["Shader", "Stage"]


class Shader(Destroyable):

    def __init__(self, session, path, entry_point=None):
        super(Shader, self).__init__()
        self.session = session
        self.session.register_shader(self)
        self.vulkan_shader = _Shader(self.session.device, path, entry_point)
        self.vulkan_shader.inspect()

    def _destroy(self):
        self.vulkan_shader.destroy()

    def get_bindings(self):
        return self.vulkan_shader.get_bindings()

    def get_block_definition(self, binding):
        return self.vulkan_shader.get_block_definition(binding)

    def get_block_usage(self, binding):
        return self.vulkan_shader.get_block_usage(binding)

    def get_local_size(self):
        return self.vulkan_shader.get_local_size()

    def get_block_access(self, binding):
        return self.vulkan_shader.get_block_access(binding)


class Stage(Destroyable):

    def __init__(self, shader, bindings):
        super(Stage, self).__init__()
        self.session = shader.session
        self.session.register_stage(self)
        self.shader = shader
        self.bindings = bindings

        self.checked = False
        self.check_workgroups()
        memory_object_binding = self.check_block_definitions()
        self.checked = True

        self.pipeline = Pipeline(self.session.device, shader.vulkan_shader, memory_object_binding)
        self.operation = ShaderOperation(self.session.device, self.pipeline, self.session.queue_index)

    def _destroy(self):
        if self.checked:
            self.operation.destroy()
            self.pipeline.destroy()

    def check_workgroups(self):
        physical_device = self.session.device.physical_device

        x, y, z = self.shader.vulkan_shader.get_local_size()
        x_max, y_max, z_max = physical_device.get_maximum_work_group_sizes()
        group_invocations_max = physical_device.get_maximum_work_group_invocations()
        if x > x_max or y > y_max or z > z_max:
            msg = "Device supports work group sizes up to x={} y={} z={}, but shader defines x={} y={} z={}".format(
                x_max, y_max, z_max, x, y, z)
            raise LavaError(msg)

        if x * y * z > group_invocations_max:
            msg = "Device supports work group invocations up to {}, but shader defines {}*{}*{}={}".format(
                group_invocations_max, x, y, z, x*y*z)
            raise LavaError(msg)

    def check_block_definitions(self):
        memory_object_binding = {}
        bindings_shader = self.shader.get_bindings()
        max_uniform_size = self.session.device.physical_device.get_maximum_uniform_size()

        for binding, buffer in self.bindings.items():
            if binding not in bindings_shader:
                raise LavaError("Shader does not define binding {}".format(binding))

            usage_shader = self.shader.get_block_usage(binding)
            usage_buffer = buffer.get_block_usage()

            if usage_buffer != usage_shader:
                raise LavaError("Shader defines binding {} as {}, but got {}".format(
                    binding, usage_shader, usage_buffer))

            definition_shader = self.shader.get_block_definition(binding)
            definition_buffer = buffer.get_block_definition()

            try:
                definition_shader.compare(definition_buffer, quiet=False)
            except TypeError as e:
                msg = "Block definition mismatch of buffer and shader at binding {}".format(binding)
                msg += "\n" + e.args[0]
                raise_with_traceback(LavaError(msg))

            if usage_buffer == BufferUsage.UNIFORM_BUFFER:
                if definition_buffer.size() > max_uniform_size:
                    msg = "Uniform at binding {} will be larger than maximum of {}, weird things might happen".format(
                        binding, max_uniform_size)
                    warnings.warn(msg, UserWarning)

            memory_object_binding[binding] = buffer.get_vulkan_buffer()
        return memory_object_binding

    def record(self, x, y, z, after_stages=(),):
        x_max, y_max, z_max = self.session.device.physical_device.get_maximum_work_group_counts()
        if x > x_max or y > y_max or z > z_max:
            msg = "Device supports work group counts up to x={} y={} z={}, but requested are x={} y={} z={}".format(
                x_max, y_max, z_max, x, y, z)
            raise LavaError(msg)

        wait_events = [stage.executor.event for stage in after_stages]
        self.operation.record(x, y, z, wait_events)

    def run(self):
        for binding, buffer in self.bindings.items():
            access_modifier = self.shader.get_block_access(binding)
            buffer.before_stage(self, binding, access_modifier)

        self.operation.run()

    def wait(self):
        self.operation.wait()

        for binding, buffer in self.bindings.items():
            access_modifier = self.shader.get_block_access(binding)
            buffer.after_stage(self, binding, access_modifier)

    def run_and_wait(self):
        self.run()
        self.wait()


# class Flow(object):
#
#     def __init__(self, session):
#         pass

# buffers either have
#   unlimited reads
# or
#   one write and unlimited reads afterwards
