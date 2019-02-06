# -*- coding: UTF-8 -*-

import itertools
import logging
import unittest

from lava.api.shader import *
from lava.session import Session
from lava.util import compile_glsl

from test import TestUtil
#from test import TestSession

logger = logging.getLogger(__name__)


class TestByteCodeInspection(unittest.TestCase):

    SESSION = None
    MEMORY = None

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.DEBUG)
        TestUtil.set_vulkan_environment_variables()
        cls.SESSION = Session.discover()  # TestSession() ?
        cls.MEMORY = {}

    @classmethod
    def tearDownClass(cls):
        del cls.SESSION

    # Util

    @classmethod
    def shader_from_txt(cls, txt):
        path_shader = TestUtil.write_to_temp_file(txt, suffix=".comp")
        shader_path_spirv = compile_glsl(path_shader, verbose=True)
        return Shader(cls.SESSION.device, shader_path_spirv)

    # Test

    def test1(self):
        glsl = """
        #version 450
        #extension GL_ARB_separate_shader_objects : enable
        
        layout(local_size_x=1, local_size_y=1, local_size_z=1) in;
        //layout(local_size_x=100, local_size_y=1, local_size_z=1) in;
        
        struct Data1 {
            vec3 var1;
            ivec2 var2;
        };
        
        struct Data2 {
            double var1;
            double var2;
            Data1 var3;
        };
        
        layout(std140, binding = 0) uniform uniIn
        //layout(std430, binding = 0) readonly buffer bufIn
        {
            bool flag2;
            vec2[5][2][3] abc;
            //float bufferIn[5];
            bool flag;
            mat4x4 model;
            Data1[2] datas1;
            Data2 datas2;
        };
        
        //layout(std140, binding = 1) writeonly buffer bufOut
        layout(std430, binding = 1) writeonly buffer bufOut
        {
            uint width;
            uint height;
            float bufferOut[];
            uint asd;
        };
        
        void main() {
            uint index = gl_GlobalInvocationID.x;
            //bufferOut[index] = bufferIn[index] + 1;
        
        }
        """

        byte_code = ByteCode(self.shader_from_txt(glsl).byte_code)

        print ""
        print byte_code
