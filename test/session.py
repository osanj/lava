# -*- coding: UTF-8 -*-

import unittest
import logging
import os

import numpy as np

from lava.buffer import BufferCPU, BufferGPU, StagedBuffer
from lava.session import Session
from lava.shader import Shader, Stage
from lava.util import compile_glsl

from test import TestUtil

logger = logging.getLogger(__name__)


class SessionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig()
        TestUtil.set_vulkan_environment_variables()

    def setUp(self):
        super(SessionTest, self).setUp()
        self.session = Session.discover(validation_lvl=logging.INFO)

    def tearDown(self):
        super(SessionTest, self).tearDown()
        del self.session

    def shader_from_txt(self, txt, verbose=True, clean_up=True):
        path_shader = TestUtil.write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = compile_glsl(path_shader, verbose)
        shader = Shader(self.session, shader_path_spirv)
        if clean_up:
            os.remove(path_shader)
            os.remove(shader_path_spirv)
        return shader

    def test_limits(self):
        import vulkan as vk

        properties = vk.vkGetPhysicalDeviceProperties(self.session.device.physical_device.handle)
        # for x in dir(properties.limits):
        #     print x, getattr(properties.limits, x)

        # print ""
        print "max compute work group count      ", properties.limits.maxComputeWorkGroupCount[0], properties.limits.maxComputeWorkGroupCount[1], properties.limits.maxComputeWorkGroupCount[2]
        print "max compute work group invocations", properties.limits.maxComputeWorkGroupInvocations
        print "max compute work group size       ", properties.limits.maxComputeWorkGroupSize[0], properties.limits.maxComputeWorkGroupSize[1], properties.limits.maxComputeWorkGroupSize[2]

    def test_image_through(self):
        glsl_template = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout({}, binding = 0) buffer BufferA {{
                {}[720][1280][3] imageIn;
            }};

            layout({}, binding = 1) buffer BufferB {{
                {}[720][1280][3] imageOut;
            }};

            void main() {{
                vec3 pixel = gl_GlobalInvocationID;
                int h = int(pixel.x);
                int w = int(pixel.y);
                int c = int(pixel.z);
                
                imageOut[h][w][c] = imageIn[h][w][c];
            }}
            """

        glsl = glsl_template.format()
        shader = self.shader_from_txt(glsl)

        buffer_im_in = BufferCPU.from_shader(self.session, shader, binding=0)
        buffer_im_out = BufferCPU.from_shader(self.session, shader, binding=1)

    def test_convolution(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            //layout(std140, binding = 0) buffer readonly BufferA {
            layout(std430, binding = 0) buffer readonly BufferA {
            //layout(std140, binding = 0) uniform BufferA {
                float[720][1280][3] imageIn;
                //float[352][626][3] imageIn;
            };

            //layout(std140, binding = 1) buffer readonly BufferB {
            layout(std430, binding = 1) buffer readonly BufferB {
            //layout(std140, binding = 1) uniform BufferB {
                float[5][5] convWeights;
            };
            
            
            layout(std430, binding = 2) buffer writeonly BufferC {
                float[720][1280][3] imageOut;
                //float[352][626][3] imageOut;
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
                //imageOut[h][w][c] = 128;
                //imageOut[h][w][c] = imageIn[h][w][c];
            }
            """

        shader = self.shader_from_txt(glsl, clean_up=False)

        buffer_im_in = StagedBuffer.from_shader(self.session, shader, binding=0)
        buffer_weights_in = StagedBuffer.from_shader(self.session, shader, binding=1)
        buffer_im_out = StagedBuffer.from_shader(self.session, shader, binding=2)

        import cv2 as cv
        import time

        im = cv.imread("image.jpg", cv.IMREAD_COLOR)

        buffer_im_in["imageIn"] = im.astype(np.float32)
        buffer_weights_in["convWeights"] = np.ones((5, 5), dtype=np.float32) / (5 * 5)

        stage = Stage(shader, {0: buffer_im_in, 1: buffer_weights_in, 2: buffer_im_out})

        stage.record(*im.shape)
        #stage.record(int(im.shape[0] / 38. + 0.5), int(im.shape[1] / 38. + 0.5), 3)
        #stage.record(1, 1, 1)

        t0 = time.time()
        stage.run_and_wait()
        print "run_and_wait", time.time() - t0

        print ""
        print ""
        print ""

        im_filtered = buffer_im_out["imageOut"]

        cv.imshow("input", im)
        cv.imshow("output", np.around(im_filtered).astype(np.uint8))
        cv.waitKey()
        cv.destroyAllWindows()
