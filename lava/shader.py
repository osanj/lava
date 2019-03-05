# -*- coding: UTF-8 -*-

import warnings

from lava.api.constants.vk import BufferUsage
from lava.api.pipeline import ShaderOperation, Pipeline
from lava.api.shader import Shader as _Shader

__all__ = ["Shader", "Stage"]


class Shader(object):

    def __init__(self, session, path, entry_point=None):
        self.session = session
        self.vulkan_shader = _Shader(self.session.device, path, entry_point)
        self.vulkan_shader.inspect()

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


class Stage(object):

    def __init__(self, shader, bindings):
        self.session = shader.session
        self.shader = shader
        self.bindings = bindings

        self.check_workgroups()
        memory_object_binding = self.check_block_definitions()

        self.pipeline = Pipeline(self.session.device, shader.vulkan_shader, memory_object_binding)
        self.operation = ShaderOperation(self.session.device, self.pipeline, self.session.queue_index)

    def check_workgroups(self):
        physical_device = self.session.device.physical_device

        x, y, z = self.shader.vulkan_shader.get_local_size()
        x_max, y_max, z_max = physical_device.get_maximum_work_group_sizes()
        group_invocations_max = physical_device.get_maximum_work_group_invocations()
        if x > x_max or y > y_max or z > z_max:
            msg = "Device supports work group sizes up to x={} y={} z={}, but shader defines x={} y={} z={}".format(
                x_max, y_max, z_max, x, y, z)
            raise RuntimeError(msg)

        if x * y * z > group_invocations_max:
            msg = "Device supports work group invocations up to {}, but shader defines {}*{}*{}={}".format(
                group_invocations_max, x, y, z, x*y*z)
            raise RuntimeError(msg)

    def check_block_definitions(self):
        memory_object_binding = {}
        bindings_shader = self.shader.get_bindings()
        max_uniform_size = self.session.device.physical_device.get_maximum_uniform_size()

        for binding, buffer in self.bindings.items():
            if binding not in bindings_shader:
                raise RuntimeError("Shader does not define binding {}".format(binding))

            usage_shader = self.shader.get_block_usage(binding)
            usage_buffer = buffer.get_block_usage()

            if usage_buffer != usage_shader:
                raise RuntimeError("Shader defines binding {} as {}, but got {}".format(
                    binding, usage_shader, usage_buffer))

            definition_shader = self.shader.get_block_definition(binding)
            definition_buffer = buffer.get_block_definition()

            try:
                definition_shader.compare(definition_buffer, quiet=False)
            except RuntimeError as e:
                msg = "Block definition mismatch of buffer and shader at binding {}".format(binding)
                args = list(e.args)
                args[0] = msg + "\n" + e.args[0]
                e.args = tuple(args)
                raise e

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
            raise RuntimeError(msg)

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
