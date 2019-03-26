# -*- coding: UTF-8 -*-

import os
import unittest

import numpy as np

import lava as lv
from lava.api.bytes import Array, ScalarFloat, Struct
from lava.api.constants.spirv import Layout
from lava.api.util import LavaError

from test.util import write_to_temp_file


class ShaderTest(unittest.TestCase):

    def setUp(self):
        super(ShaderTest, self).setUp()
        self.session = lv.Session(lv.devices()[0])

    def tearDown(self):
        super(ShaderTest, self).tearDown()
        self.session.destroy()

    def shader_from_txt(self, txt, verbose=True, clean_up=True):
        path_shader = write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = lv.compile_glsl(path_shader, verbose)
        shader = lv.Shader(self.session, shader_path_spirv)
        if clean_up:
            os.remove(path_shader)
            os.remove(shader_path_spirv)
        return shader

    def test_definition_check(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(std140, binding = 0) buffer BufferA {
                float[720][1280][3] image;
            };

            void main() {}
            """
        shader = self.shader_from_txt(glsl, verbose=False)
        incompatible_definition = Struct([Array(ScalarFloat(), (721, 1281, 4), Layout.STD140)], Layout.STD140)
        incompatible_buffer = lv.BufferCPU(self.session, incompatible_definition, lv.BufferCPU.USAGE_STORAGE)
        self.assertRaises(LavaError, lv.Stage, shader=shader, bindings={0: incompatible_buffer})

    def test_convolution(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std430, binding = 0) buffer readonly BufferA {
                float[72][128][3] imageIn;
            };

            layout(std430, binding = 1) buffer readonly BufferB {
                float[5][5] convWeights;
            };

            layout(std430, binding = 2) buffer writeonly BufferC {
                float[72][128][3] imageOut;
            };

            int height = imageIn.length();
            int width = imageIn[0].length();
            int channels = imageIn[0][0].length();

            bool inputHasPixel(int y, int x, int c) {
                return !(y < 0 || y >= height || x < 0 || x >= width || c < 0 || c >= channels);
            }

            float inputPixel(int y, int x, int c, float defaultValue) {
                if (inputHasPixel(y, x, c)) {
                    return imageIn[y][x][c];
                }

                return defaultValue;
            }

            void main() {
                int filterHeight = convWeights.length();
                int filterWidth = convWeights[0].length();
                int fdh = (filterWidth - 1) / 2;
                int fdw = (filterHeight - 1) / 2;

                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);
                int c = int(pixel.z);

                if (!inputHasPixel(h, w, c)) {
                    return;
                }

                float filteredPixel = 0;

                for (int i = 0; i < filterHeight; i++) {
                    for (int j = 0; j < filterWidth; j++) {
                        float a = inputPixel(h - fdh + i, w - fdw + j, c, 0.0);
                        float b = convWeights[i][j];
                        filteredPixel += a * b;
                    }
                }

                imageOut[h][w][c] = filteredPixel;
            }
            """

        shader = self.shader_from_txt(glsl, verbose=False)

        buf_in = lv.StagedBuffer.from_shader(self.session, shader, binding=0)
        buf_weights = lv.StagedBuffer.from_shader(self.session, shader, binding=1)
        buf_out = lv.StagedBuffer.from_shader(self.session, shader, binding=2)

        im = np.random.randint(0, 255, size=(72, 128, 3)).astype(np.float32)
        weights = np.ones((5, 5), dtype=np.float32) / (5 * 5)

        buf_in["imageIn"] = im
        buf_weights["convWeights"] = weights

        stage = lv.Stage(shader, {0: buf_in, 1: buf_weights, 2: buf_out})
        stage.record(*im.shape)
        stage.run_and_wait()

        im_filtered = buf_out["imageOut"]

        # convolution on cpu
        padding = 2
        h, w, c = im.shape
        im_padded = np.zeros((h + 2 * padding, w + 2 * padding, c), dtype=np.float32)
        im_padded[padding:-padding, padding:-padding, :] = im
        im_filtered_expected = np.zeros(im.shape, dtype=np.float32)

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    im_filtered_expected[i, j, k] = np.sum(
                        im_padded[i:i+2*padding+1, j:j+2*padding+1, k] * weights)

        self.assertTrue(np.mean(np.abs(im_filtered - im_filtered_expected)) < 1e-3)


if __name__ == "__main__":
    unittest.main()
