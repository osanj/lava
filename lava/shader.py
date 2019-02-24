# -*- coding: UTF-8 -*-

import logging
import time
import warnings

from lava.buffer import BufferCPU
from lava.api.constants.spirv import Access
from lava.api.constants.vk import BufferUsage
from lava.api.pipeline import ShaderOperation, Pipeline
from lava.api.shader import Shader as _Shader


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
        self.executor = ShaderOperation(self.session.device, self.pipeline, self.session.queue_index)

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

        for binding, buffer in self.bindings.iteritems():
            if binding not in bindings_shader:
                raise RuntimeError("Shader does not define binding {}".format(binding))

            usage_shader = self.shader.get_block_usage(binding)
            usage_buffer = self.shader.get_block_usage(binding)

            if usage_buffer != usage_shader:
                raise RuntimeError("Shader defines binding {} as {}, but got {}".format(
                    binding, usage_shader, usage_buffer))

            definition_shader = self.shader.get_block_definition(binding)
            definition_buffer = self.shader.get_block_definition(binding)

            try:
                definition_shader.compare(definition_buffer, quiet=False)
            except RuntimeError as e:
                raise e  # TODO: add message

            if usage_buffer == BufferUsage.UNIFORM_BUFFER:
                if definition_buffer.size() > max_uniform_size:
                    msg = "Uniform at binding {} will be larger than maximum of {}, weird things might happen".format(
                        binding, max_uniform_size)
                    warnings.warn(msg, UserWarning)

            memory_object_binding[binding] = buffer.vulkan_buffer
        return memory_object_binding

    def record(self, x, y, z, after_stages=(),):
        x_max, y_max, z_max = self.session.device.physical_device.get_maximum_work_group_counts()
        if x > x_max or y > y_max or z > z_max:
            msg = "Device supports work group counts up to x={} y={} z={}, but requested are x={} y={} z={}".format(
                x_max, y_max, z_max, x, y, z)
            raise RuntimeError(msg)

        wait_events = [stage.executor.event for stage in after_stages]
        self.executor.record(x, y, z, wait_events)

    def run(self):
        t00 = time.time()

        for binding, buffer in self.bindings.iteritems():
            access = self.shader.get_block_access(binding)

            if isinstance(buffer, BufferCPU):
                if not buffer.is_synced() and access in [Access.READ_ONLY, Access.READ_WRITE]:
                    a = time.time()
                    buffer.write()
                    print "buffer write", time.time() - a

        print "prestage", time.time() - t00
        self.t0 = time.time()
        self.executor.run()

    def wait(self):
        self.executor.wait()
        t1 = time.time()
        print ""
        print "stage", t1 - self.t0
        print ""

        for binding, buffer in self.bindings.iteritems():
            access = self.shader.get_block_access(binding)

            if isinstance(buffer, BufferCPU):
                if access in [Access.WRITE_ONLY, Access.READ_WRITE]:
                    if not buffer.is_synced():
                        msg = "Cache of cpu buffer backing binding {} contains unmapped data, it will be overwritten"\
                            .format(binding)
                        warnings.warn(msg, RuntimeWarning)

                    a = time.time()
                    buffer.read()
                    print "buffer read", time.time() - a

        t11 = time.time()
        print "poststage",  t11 - t1

    def run_and_wait(self):
        self.run()
        self.wait()




