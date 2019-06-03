# -*- coding: UTF-8 -*-

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

    def setUp(self) -> None:
        self.clear_package()

    def tearDown(self) -> None:
        self.clear_package()

    def test_validation_level(self):
        import lava as lv
        lv.VALIDATION_LEVEL = lv.VALIDATION_LEVEL_INFO

        debugger = lv.instance().debugger
        self.assertTrue(len(debugger.history) > 0)
        self.assertTrue("CREATE PhysicalDevice object" in " ".join(debugger.history))

        lv.VALIDATION_LEVEL = None

    def test_delayed_import_errors(self):
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

            initialization_status = lv.initialized()

        self.assertIsNone(error_on_import)
        self.assertIsNotNone(error_on_instance)
        self.assertIsNotNone(error_on_devices)
        self.assertFalse(initialization_status)
