# -*- coding: UTF-8 -*-

import builtins
import contextlib
import os
import sys
import unittest


class InitializationTest(unittest.TestCase):

    @staticmethod
    @contextlib.contextmanager
    def env_backup():
        environ_backup = dict(os.environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(environ_backup)

    @staticmethod
    def clear_package():
        if "lava" in sys.modules:
            del sys.modules["lava"]

    def setUp(self):
        self.clear_package()

    def tearDown(self):
        self.clear_package()

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
        error_on_import = None
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

        try:
            import lava as lv
        except Exception as e:
            error_on_import = e


        from lava.buffer import BufferCPU, StagedBuffer

        print("abc")
        buf = BufferCPU(None, None, None)
        print("abc")


        try:
            lv.instance()
        except Exception as e:
            error_on_instance = e

        try:
            lv.devices()
        except Exception as e:
            error_on_devices = e

        self.assertIsNone(error_on_import)
        self.assertTrue(isinstance(error_on_instance, OSError))
        self.assertTrue(isinstance(error_on_devices, OSError))
        self.assertFalse(lv.initialized())

        builtins.__import__ = import_original
