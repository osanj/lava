# -*- coding: UTF-8 -*-

import os
import platform
import tempfile


from lava.session import Session


class TestSession(Session):

    def __init__(self):
        pass


class TestUtil(object):

    @classmethod
    def write_to_temp_file(cls, txt, mode="w", prefix="lavatest-", suffix=""):
        if platform.system() != "Linux":
            raise NotImplementedError()

        with tempfile.NamedTemporaryFile(mode=mode, prefix=prefix, suffix=suffix, delete=False) as f:
            f.write(txt)
            return f.name
