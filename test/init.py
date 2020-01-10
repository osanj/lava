# -*- coding: UTF-8 -*-

import builtins
import contextlib
import os
import sys
import unittest

from test.util import write_to_temp_file


class InitializationTest(unittest.TestCase):

    PKG_NAME = "lava"

    @staticmethod
    @contextlib.contextmanager
    def env_backup():
        environ_backup = dict(os.environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(environ_backup)

    @classmethod
    def clear_lava(cls):
        if cls.PKG_NAME in sys.modules:
            del sys.modules[cls.PKG_NAME]

    def setUp(self):
        self.clear_lava()

    def tearDown(self):
        self.clear_lava()

    def test_validation_level(self):
        import lava as lv
        lv.VALIDATION_LEVEL = lv.VALIDATION_LEVEL_INFO

        debugger = lv.instance().debugger
        self.assertTrue(len(debugger.history) > 0)
        self.assertTrue("CREATE PhysicalDevice object" in " ".join(debugger.history))

        lv.VALIDATION_LEVEL = None

    def test_import_missing_variable(self):
        error_on_import = None
        error_on_instance = None
        error_on_devices = None

        with self.env_backup():
            del os.environ["VULKAN_SDK"]

            try:
                import lava as lv
            except Exception as e:
                error_on_import = e

            try:
                lv.instance()
            except Exception as e:
                error_on_instance = e

            try:
                lv.devices()
            except Exception as e:
                error_on_devices = e

        self.assertIsNone(error_on_import)
        self.assertTrue(isinstance(error_on_instance, ImportError))
        self.assertTrue(isinstance(error_on_devices, ImportError))
        self.assertFalse(lv.initialized())

    def test_import_vulkan_unavailable(self):
        errors_on_module_import = []
        error_on_instance = None
        error_on_devices = None

        # mock vulkan initialization error
        # https://stackoverflow.com/a/2481588
        import_original = builtins.__import__

        def import_mock(name, *args, **kwargs):
            if name == "vulkan":
                # https://github.com/realitix/vulkan/blob/1.1.99.0/generator/vulkan.template.py#L105
                raise OSError("Cannot find Vulkan SDK version. Please ensure...")
            return import_original(name, *args, **kwargs)

        builtins.__import__ = import_mock

        # import should not fail
        import lava as lv

        # find and try all possible module imports, e.g. import lava.api.shader, ...
        pkg_name = self.PKG_NAME
        pkg_path = lv.__path__[0]
        ext = ".py"
        paths = []

        for root, dirs, files in os.walk(pkg_path):
            for name in files:
                if name.endswith(ext):
                    paths.append([pkg_name] + root[len(pkg_path):].split(os.sep)[1:] + [name[:-len(ext)]])

        for path in paths:
            try:
                __import__(".".join(path))
            except Exception as e:
                errors_on_module_import.append(e)

        try:
            lv.instance()
        except Exception as e:
            error_on_instance = e

        try:
            lv.devices()
        except Exception as e:
            error_on_devices = e

        self.assertTrue(len(errors_on_module_import) == 0, "\n".join([str(e) for e in errors_on_module_import]))
        self.assertTrue(isinstance(error_on_instance, OSError))
        self.assertTrue(isinstance(error_on_devices, OSError))
        self.assertFalse(lv.initialized())

        builtins.__import__ = import_original

    def test_bytecode_parsing_without_gpu(self):
        with self.env_backup():
            del os.environ["VULKAN_SDK"]

            import lava as lv
            from lava.api.bytecode.logical import ByteCode
            from lava.api.bytecode.physical import ByteCodeData
            from lava.api.bytes import Vector, Scalar, Struct
            from lava.api.constants.spirv import Layout

            self.assertTrue(not lv.initialized())

        glsl = """
            #version 450
            #extension GL_ARB_separate_shader_objects : enable

            layout(local_size_x=1, local_size_y=1, local_size_z=1) in;

            layout(std430, binding = 0) readonly buffer bufIn {
                vec3 var1;
            };

            layout(std430, binding = 1) writeonly buffer bufOut {
                float var2;
            };

            void main() {
                var2 = var1.x + var1.y + var1.z;
            }
            """

        path_shader = write_to_temp_file(glsl, suffix=".comp")
        path_shader_spirv = lv.compile_glsl(path_shader, verbose=True)

        with self.env_backup():
            del os.environ["VULKAN_SDK"]

            self.assertTrue(not lv.initialized())

            byte_code_data = ByteCodeData.from_file(path_shader_spirv)
            byte_code = ByteCode(byte_code_data, None)

            quiet = True
            container0 = Struct([Vector.vec3()], Layout.STD430)
            container1 = Struct([Scalar.float()], Layout.STD430)
            self.assertTrue(container0.compare(byte_code.get_block_definition(0), quiet=quiet))
            self.assertTrue(container1.compare(byte_code.get_block_definition(1), quiet=quiet))

        os.remove(path_shader)
        os.remove(path_shader_spirv)
