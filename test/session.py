# -*- coding: UTF-8 -*-

import unittest
import logging
import os

import numpy as np

from lava.api.pipeline import Executor, Pipeline
from lava.buffer import Buffer
from lava.session import Session
from lava.shader import Shader
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
        for x in dir(properties.limits):
            print x, getattr(properties.limits, x)

    def test_convolution(self):
        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            
            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std430, binding = 0) buffer BufferA {
                //float[30][30][3] imageIn;
                //float[300][300][3] imageIn;
                float[720][1280][3] imageIn;
            };

            layout(std430, binding = 1) buffer BufferB {
                float[5][5] convWeights;
            };
            
            
            layout(std430, binding = 2) buffer BufferC {
                //float[30][30][3] imageOut;
                //float[300][300][3] imageOut;
                float[720][1280][3] imageOut;
            };
            
            bool inputHasPixel(int y, int x, int c) {
                int height = imageIn.length();
                int width = imageIn[0].length();
                int channels = imageIn[0][0].length();
                
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
                
                imageOut[h][w][c] = 0;

                for (int i = 0; i < filterHeight; i++) {
                    for (int j = 0; j < filterWidth; j++) {
                        float a = inputPixel(h - fdh + i, w - fdw + j, c, 0.0);
                        float b = convWeights[i][j];
                        imageOut[h][w][c] += a * b;
                    }
                }
            }
            """

        shader = self.shader_from_txt(glsl, clean_up=False)

        buffer_im_in = Buffer.from_shader(self.session, shader, binding=0, location=Buffer.LOCATION_CPU)
        buffer_weights_in = Buffer.from_shader(self.session, shader, binding=1, location=Buffer.LOCATION_CPU)
        buffer_im_out = Buffer.from_shader(self.session, shader, binding=2, location=Buffer.LOCATION_CPU)

        print buffer_im_in.definition
        print ""
        print buffer_weights_in.definition
        print ""
        print buffer_im_out.definition
        print ""

        buffer_im_in.allocate()
        buffer_weights_in.allocate()
        buffer_im_out.allocate()

        import cv2 as cv

        im = cv.imread("image.jpg", cv.IMREAD_COLOR)

        buffer_im_in["imageIn"] = im.astype(np.float32)
        buffer_weights_in["convWeights"] = np.ones((5, 5), dtype=np.float32) / (5 * 5)

        buffer_im_in.write()
        buffer_weights_in.write()

        pipeline = Pipeline(self.session.device, shader.vulkan_shader, [buffer_im_in.vulkan_buffer, buffer_weights_in.vulkan_buffer, buffer_im_out.vulkan_buffer])
        executor = Executor(self.session.device, pipeline, self.session.queue_index)

        executor.record(*im.shape)
        executor.execute_and_wait()

        buffer_im_out.read()

        im_filtered = buffer_im_out["imageOut"]

        cv.imshow("input", im)
        cv.imshow("output", np.around(im_filtered).astype(np.uint8))
        cv.waitKey()
        cv.destroyAllWindows()
