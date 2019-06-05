# -*- coding: UTF-8 -*-

import builtins
import contextlib
import os
import sys
import unittest


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
