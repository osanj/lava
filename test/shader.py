# -*- coding: UTF-8 -*-

import logging
import os
import unittest

import lava as lv
from lava.api.bytes import Array, ScalarFloat, Struct
from lava.api.constants.spirv import Layout

from test.util import write_to_temp_file

logger = logging.getLogger(__name__)


class ShaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig()

    def setUp(self):
        super(ShaderTest, self).setUp()
        self.session = lv.Session(lv.devices()[0])

    def tearDown(self):
        super(ShaderTest, self).tearDown()
        del self.session

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

        shader = self.shader_from_txt(glsl)
        incompatible_definition = Struct([Array(ScalarFloat(), (721, 1281, 4), Layout.STD140)], Layout.STD140)
        incompatible_buffer = lv.BufferCPU(self.session, incompatible_definition, lv.BufferCPU.USAGE_STORAGE)
        self.assertRaises(RuntimeError, lv.Stage, shader=shader, bindings={0: incompatible_buffer})


if __name__ == "__main__":
    unittest.main()
