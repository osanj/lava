# -*- coding: UTF-8 -*-

import unittest

import lava as lv


class InitializationTest(unittest.TestCase):

    def test_validation_level(self):
        lv.VALIDATION_LEVEL = lv.VALIDATION_LEVEL_INFO

        debugger = lv.instance().debugger
        self.assertTrue(len(debugger.history) > 0)
        self.assertTrue("CREATE PhysicalDevice object" in " ".join(debugger.history))

        lv.VALIDATION_LEVEL = None
