# -*- coding: UTF-8 -*-

import itertools
import os
import unittest

import numpy as np

import lava as lv
from lava.api.bytes import Array, ScalarFloat, Struct
from lava.api.constants.spirv import Layout

from test.util import write_to_temp_file


class BufferTest(unittest.TestCase):

    def setUp(self):
        super(BufferTest, self).setUp()
        self.session = lv.Session(lv.devices()[0])

    def tearDown(self):
        super(BufferTest, self).tearDown()
        self.session.destroy()

    def shader_from_txt(self, txt, verbose=True, clean_up=True):
        path_shader = write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = lv.compile_glsl(path_shader, verbose)
        shader = lv.Shader(self.session, shader_path_spirv)
        if clean_up:
            os.remove(path_shader)
            os.remove(shader_path_spirv)
        return shader

    def test_copy(self):
        data = np.arange(128, dtype=np.float32)
        definition = Struct([Array(ScalarFloat(), len(data), Layout.STD140)], Layout.STD140)

        buffer_a = lv.BufferCPU(self.session, definition, lv.BufferCPU.USAGE_STORAGE)
        buffer_b = lv.BufferCPU(self.session, definition, lv.BufferCPU.USAGE_STORAGE)

        buffer_a[0] = data
        buffer_a.flush()
        buffer_a.copy_to(buffer_b)
        buffer_b.fetch()

        self.assertTrue((buffer_a[0] == buffer_b[0]).all())

    def test_copy_over_gpu(self):
        data = np.arange(128, dtype=np.float32)
        definition = Struct([Array(ScalarFloat(), len(data), Layout.STD140)], Layout.STD140)

        buffer_a = lv.BufferCPU(self.session, definition, lv.BufferCPU.USAGE_STORAGE)
        buffer_gpu = lv.BufferGPU(self.session, definition, lv.BufferCPU.USAGE_STORAGE)
        buffer_b = lv.BufferCPU(self.session, definition, lv.BufferCPU.USAGE_STORAGE)

        buffer_a[0] = data
        buffer_a.flush()
        buffer_a.copy_to(buffer_gpu)
        buffer_gpu.copy_to(buffer_b)
        buffer_b.fetch()

        self.assertTrue((buffer_a[0] == buffer_b[0]).all())

    def test_sync_modes(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std140, binding = 0) buffer readonly BufferA {
                float[72][128][3] imageIn;
            };

            layout(std140, binding = 1) buffer writeonly BufferB {
                float[72][128][3] imageOut;
            };

            void main() {
                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);
                int c = int(pixel.z);

                imageOut[h][w][c] = imageIn[h][w][c];
            }
            """

        shader = self.shader_from_txt(glsl, verbose=False)
        classes = [lv.BufferCPU, lv.StagedBuffer]
        sync_modes = [lv.BufferCPU.SYNC_LAZY, lv.BufferCPU.SYNC_EAGER]

        data = np.ones((72, 128, 3), dtype=np.float32)

        for cls_in, cls_out, sync_mode in itertools.product(classes, classes, sync_modes):
            buffer_in = cls_in.from_shader(self.session, shader, binding=0, sync_mode=sync_mode)
            buffer_out = cls_in.from_shader(self.session, shader, binding=1, sync_mode=sync_mode)

            stage = lv.Stage(shader, {0: buffer_in, 1: buffer_out})
            stage.record(*data.shape)

            buffer_in[0] = data

            if sync_mode is lv.BufferCPU.SYNC_LAZY:
                self.assertTrue(not buffer_in.is_synced())
            if sync_mode is lv.BufferCPU.SYNC_EAGER:
                self.assertTrue(buffer_in.is_synced())

            stage.run_and_wait()

            if sync_mode is lv.BufferCPU.SYNC_LAZY:
                self.assertTrue(buffer_in.is_synced())
                self.assertTrue(not buffer_out.is_synced())

            if sync_mode is lv.BufferCPU.SYNC_EAGER:
                self.assertTrue(buffer_out.is_synced())

            self.assertTrue((buffer_in[0] == buffer_out[0]).all())

            if sync_mode is lv.BufferCPU.SYNC_LAZY:
                self.assertTrue(buffer_out.is_synced())

    def test_partially_modified_buffer(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
            
            struct Struct1 {
                uint var1;
                uint var2;
            };
            
            struct Struct2 {
                double var1;
                Struct1[2] var2;
            };

            layout(std140, binding = 0) buffer readonly BufferA {
                float var1;
                Struct2 var2;
                vec2[3] var3;
            } inputData;
            
            layout(std140, binding = 1) buffer writeonly BufferB {
                float var1;
                Struct2 var2;
                vec2[3] var3;
            } outputData;

            void main() {
                outputData.var1 = inputData.var1;
                outputData.var2 = inputData.var2;
                outputData.var3 = inputData.var3;
            }
            """

        shader = self.shader_from_txt(glsl, verbose=False)
        classes = [lv.BufferCPU, lv.StagedBuffer]
        sync_mode = lv.BufferCPU.SYNC_LAZY

        for cls_in, cls_out in itertools.product(classes, classes):
            buffer_in = cls_in.from_shader(self.session, shader, binding=0, sync_mode=sync_mode)
            buffer_out = cls_in.from_shader(self.session, shader, binding=1, sync_mode=sync_mode)

            stage = lv.Stage(shader, {0: buffer_in, 1: buffer_out})
            stage.record(1, 1, 1)

            buffer_in["var1"] = 0.0
            buffer_in["var2"]["var1"] = 1.0
            buffer_in["var2"]["var2"][0]["var1"] = 11
            buffer_in["var2"]["var2"][0]["var2"] = 12
            buffer_in["var2"]["var2"][1]["var1"] = 21
            buffer_in["var2"]["var2"][1]["var2"] = 22
            buffer_in["var3"] = np.array(((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)), dtype=np.float32)

            self.assertFalse(buffer_in.is_synced())

            stage.run_and_wait()

            # change in nested struct
            new_value = 9.0
            buffer_in["var2"]["var1"] = new_value
            self.assertFalse(buffer_in.is_synced())

            stage.run_and_wait()

            self.assertEqual(buffer_in["var2"]["var1"], new_value)
            self.assertEqual(buffer_out["var2"]["var1"], new_value)

            # change in array of structs
            new_value = 99
            buffer_in["var2"]["var2"][0]["var2"] = new_value
            self.assertFalse(buffer_in.is_synced())

            stage.run_and_wait()

            self.assertEqual(buffer_in["var2"]["var2"][0]["var2"], new_value)
            self.assertEqual(buffer_out["var2"]["var2"][0]["var2"], new_value)

            # change on highest level
            new_value = 999.0
            buffer_in["var1"] = new_value
            self.assertFalse(buffer_in.is_synced())

            stage.run_and_wait()

            self.assertEqual(buffer_in["var1"], new_value)
            self.assertEqual(buffer_out["var1"], new_value)

# more tests:
# missing readonly, writeonly decorations


if __name__ == "__main__":
    unittest.main()
