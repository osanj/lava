# -*- coding: UTF-8 -*-

import logging
import time
import warnings

from lava.buffer import BufferCPU
from lava.api.constants.spirv import Access
from lava.api.pipeline import Executor, Pipeline
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

        # check definitions
        memory_object_binding = {}
        bindings_shader = self.shader.get_bindings()

        for binding, buffer in self.bindings.iteritems():
            if binding not in bindings_shader:
                raise RuntimeError("Shader does not define binding {}".format(binding))

            usage_shader = shader.get_block_usage(binding)
            usage_buffer = shader.get_block_usage(binding)

            if usage_buffer != usage_shader:
                raise RuntimeError("Shader defines binding as {}, but got {}".format(usage_shader, usage_buffer))

            definition_shader = shader.get_block_definition(binding)
            definition_buffer = shader.get_block_definition(binding)

            try:
                definition_shader.compare(definition_buffer, quiet=False)
            except RuntimeError as e:
                raise e  # add message

            memory_object_binding[binding] = buffer.vulkan_buffer

        self.pipeline = Pipeline(self.session.device, shader.vulkan_shader, memory_object_binding)
        self.executor = Executor(self.session.device, self.pipeline, self.session.queue_index)

    def record(self, x, y, z, after_stages=(),):
        wait_events = [stage.executor.event for stage in after_stages]
        self.executor.record(x, y, z, wait_events)

    def run(self):
        self.t00 = time.time()

        for binding, buffer in self.bindings.iteritems():
            access = self.shader.get_block_access(binding)

            if isinstance(buffer, BufferCPU):
                if not buffer.is_synced() and access in [Access.READ_ONLY, Access.READ_WRITE]:
                    buffer.write()

        self.t0 = time.time()
        print self.t0 - self.t00
        self.executor.execute()

    def wait(self):
        self.executor.wait()
        t1 = time.time()
        print t1 - self.t0

        for binding, buffer in self.bindings.iteritems():
            a = time.time()
            access = self.shader.get_block_access(binding)
            print "block access", time.time() - a

            if isinstance(buffer, BufferCPU):
                if access in [Access.WRITE_ONLY, Access.READ_WRITE]:
                    if not buffer.is_synced():
                        warnings.warn("Cache of cpu buffer contains unmapped data, it will be overwritten"
                                      .format(buffer.__class__.__name__), RuntimeWarning)

                    a = time.time()
                    buffer.read()
                    print "buffer read", time.time() - a

        t11 = time.time()
        print t11 - t1

    def run_and_wait(self):
        self.run()
        self.wait()




