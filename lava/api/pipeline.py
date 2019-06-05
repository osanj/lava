# -*- coding: UTF-8 -*-

import ctypes

import lava.api.vulkan as vk
from lava.api.constants.vk import DescriptorType, QueueType, TIMEOUT_FOREVER
from lava.api.util import Destroyable, Event, Fence


class Pipeline(Destroyable):

    def __init__(self, device, shader, memory_objects, push_constants=()):
        super(Pipeline, self).__init__()
        self.device = device
        self.shader = shader
        self.memory_objects = memory_objects
        self.descriptor_set_handle = None
        self.descriptor_pool_handle = None
        bindings = []

        for i, memory_obj in memory_objects.items():
            bindings.append(memory_obj.descriptor_set_layout(i))

        descriptor_layout_create_info = vk.VkDescriptorSetLayoutCreateInfo(flags=None, pBindings=bindings)
        self.descriptor_set_layout_handle = vk.vkCreateDescriptorSetLayout(self.device.handle,
                                                                           descriptor_layout_create_info, None)

        shader_stage_create_info = vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                                                                      module=shader.handle,
                                                                      pName=shader.get_entry_point().encode("ascii"))

        if len(push_constants) > 0:
            if len(push_constants) > 1:
                raise NotImplementedError("Using more than one push constant is not implemented")

            push_constant_range = vk.VkPushConstantRange(stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT, offset=0,
                                                         size=ctypes.sizeof(push_constants[0]))

            pipeline_layout_create_info = vk.VkPipelineLayoutCreateInfo(pSetLayouts=[self.descriptor_set_layout_handle],
                                                                        pPushConstantRanges=[push_constant_range])

        else:
            pipeline_layout_create_info = vk.VkPipelineLayoutCreateInfo(pSetLayouts=[self.descriptor_set_layout_handle])

        self.pipeline_layout_handle = vk.vkCreatePipelineLayout(self.device.handle, pipeline_layout_create_info, None)
        pipeline_create_info = vk.VkComputePipelineCreateInfo(stage=shader_stage_create_info,
                                                              layout=self.pipeline_layout_handle)

        self.handle = vk.vkCreateComputePipelines(self.device.handle, None, 1, [pipeline_create_info], None)

    def _destroy(self):
        if self.descriptor_pool_handle is not None:
            vk.vkDestroyDescriptorPool(self.device.handle, self.descriptor_pool_handle, None)
        vk.vkDestroyDescriptorSetLayout(self.device.handle, self.descriptor_set_layout_handle, None)
        vk.vkDestroyPipelineLayout(self.device.handle, self.pipeline_layout_handle, None)
        vk.vkDestroyPipeline(self.device.handle, self.handle, None)

    def allocate_descriptor_sets(self):
        pool_sizes = []
        descriptor_types = {DescriptorType.UNIFORM_BUFFER: 0, DescriptorType.STORAGE_BUFFER: 0}

        for memory_obj in self.memory_objects.values():
            descriptor_type = memory_obj.descriptor_type()
            descriptor_types[descriptor_type] += 1

        for descriptor_type, count in descriptor_types.items():
            if count > 0:
                pool_sizes.append(vk.VkDescriptorPoolSize(DescriptorType.to_vulkan(descriptor_type), count))

        pool_create_info = vk.VkDescriptorPoolCreateInfo(maxSets=1, pPoolSizes=pool_sizes)
        self.descriptor_pool_handle = vk.vkCreateDescriptorPool(self.device.handle, pool_create_info, None)

        allocate_info = vk.VkDescriptorSetAllocateInfo(descriptorPool=self.descriptor_pool_handle,
                                                       pSetLayouts=[self.descriptor_set_layout_handle])

        self.descriptor_set_handle = vk.vkAllocateDescriptorSets(self.device.handle, allocate_info)[0]

    def update_descriptor_sets(self):
        write_data = []

        for i, memory_obj in self.memory_objects.items():
            write_data.append(memory_obj.write_descriptor_set(self.descriptor_set_handle, i))

        vk.vkUpdateDescriptorSets(self.device.handle, len(write_data), write_data, 0, None)

    def get_memory_objects(self):
        return self.memory_objects

    def get_descriptor_set_handle(self):
        if self.descriptor_set_handle is None:
            raise RuntimeError("Descriptor set was not allocated yet")
        return self.descriptor_set_handle

    def get_pipeline_layout_handle(self):
        return self.pipeline_layout_handle


class ShaderOperation(Destroyable):

    def __init__(self, device, pipeline, queue_index):
        super(ShaderOperation, self).__init__()
        self.device = device
        self.pipeline = pipeline
        self.queue_index = queue_index
        self.event = Event(self.device)
        self.fence = Fence(self.device, signalled=False)

        self.pipeline.allocate_descriptor_sets()
        self.pipeline.update_descriptor_sets()

        # VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        command_pool_create_info = vk.VkCommandPoolCreateInfo(flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                                              queueFamilyIndex=queue_index)
        self.command_pool_handle = vk.vkCreateCommandPool(self.device.handle, command_pool_create_info, None)

        allocate_info = vk.VkCommandBufferAllocateInfo(commandPool=self.command_pool_handle,
                                                       level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1)
        self.command_buffer_handle = vk.vkAllocateCommandBuffers(self.device.handle, allocate_info)[0]

    def _destroy(self):
        vk.vkDestroyCommandPool(self.device.handle, self.command_pool_handle, None)
        self.event.destroy()
        self.fence.destroy()

    def record(self, count_x, count_y, count_z, wait_events=()):
        vk.vkBeginCommandBuffer(self.command_buffer_handle,
                                vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))

        vk.vkCmdBindPipeline(self.command_buffer_handle, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.handle)

        vk.vkCmdBindDescriptorSets(self.command_buffer_handle, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                   self.pipeline.get_pipeline_layout_handle(), firstSet=0, descriptorSetCount=1,
                                   pDescriptorSets=[self.pipeline.get_descriptor_set_handle()], dynamicOffsetCount=0,
                                   pDynamicOffsets=None)

        vk.vkCmdResetEvent(self.command_buffer_handle, self.event.handle, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)

        if len(wait_events) > 0:
            # https://stackoverflow.com/questions/45680135/pipeline-barriers-across-multiple-shaders
            vk.vkCmdWaitEvents(self.command_buffer_handle, len(wait_events), [event.handle for event in wait_events],
                               srcStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               dstStageMask=vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               memoryBarrierCount=0, pMemoryBarriers=None, bufferMemoryBarrierCount=0,
                               pBufferMemoryBarriers=None, imageMemoryBarrierCount=0, pImageMemoryBarriers=None)

        # vk.vkCmdPushConstants()

        vk.vkCmdDispatch(self.command_buffer_handle, count_x, count_y, count_z)

        vk.vkCmdSetEvent(self.command_buffer_handle, self.event.handle, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)

        vk.vkEndCommandBuffer(self.command_buffer_handle)

    def run(self):
        queue_handle = self.device.get_queue_handle(QueueType.COMPUTE)

        submit_info = vk.VkSubmitInfo(pCommandBuffers=[self.command_buffer_handle])

        vk.vkQueueSubmit(queue_handle, 1, [submit_info], self.fence.handle)

    def wait(self):
        vk.vkWaitForFences(self.device.handle, 1, [self.fence.handle], True, TIMEOUT_FOREVER)
        vk.vkResetFences(self.device.handle, 1, [self.fence.handle])

    def run_and_wait(self):
        self.run()
        self.wait()


class CopyOperation(Destroyable):

    def __init__(self, device, queue_index):
        super(CopyOperation, self).__init__()
        self.device = device
        self.queue_index = queue_index
        self.fence = Fence(self.device, signalled=False)

        command_pool_create_info = vk.VkCommandPoolCreateInfo(flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                                              queueFamilyIndex=queue_index)
        self.command_pool_handle = vk.vkCreateCommandPool(self.device.handle, command_pool_create_info, None)

        allocation_info = vk.VkCommandBufferAllocateInfo(commandPool=self.command_pool_handle, commandBufferCount=1,
                                                         level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        self.command_buffer_handle = vk.vkAllocateCommandBuffers(self.device.handle, allocation_info)[0]

    def _destroy(self):
        vk.vkDestroyCommandPool(self.device.handle, self.command_pool_handle, None)
        self.fence.destroy()

    def record(self, src_buffer, dst_buffer):
        vk.vkBeginCommandBuffer(self.command_buffer_handle,
                                vk.VkCommandBufferBeginInfo(flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))

        region = vk.VkBufferCopy(0, 0, src_buffer.get_size())
        vk.vkCmdCopyBuffer(self.command_buffer_handle, src_buffer.handle, dst_buffer.handle, 1, [region])

        vk.vkEndCommandBuffer(self.command_buffer_handle)

    def run(self):
        queue_handle = self.device.get_queue_handle(QueueType.COMPUTE)

        submit_info = vk.VkSubmitInfo(pCommandBuffers=[self.command_buffer_handle])

        vk.vkQueueSubmit(queue_handle, 1, [submit_info], self.fence.handle)

    def wait(self):
        vk.vkWaitForFences(self.device.handle, 1, [self.fence.handle], True, TIMEOUT_FOREVER)
        vk.vkResetFences(self.device.handle, 1, [self.fence.handle])

    def run_and_wait(self):
        self.run()
        self.wait()
