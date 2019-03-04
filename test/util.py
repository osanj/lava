# -*- coding: UTF-8 -*-

import platform
import tempfile


def write_to_temp_file(txt, mode="w", prefix="lavatest-", suffix=""):
    if platform.system() != "Linux":
        raise NotImplementedError()

    with tempfile.NamedTemporaryFile(mode=mode, prefix=prefix, suffix=suffix, delete=False) as f:
        f.write(txt)
        return f.name
