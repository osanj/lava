# -*- coding: UTF-8 -*-

import tempfile


def write_to_temp_file(txt, mode="w", prefix="lavatest-", suffix=""):
    with tempfile.NamedTemporaryFile(mode=mode, prefix=prefix, suffix=suffix, delete=False) as f:
        f.write(txt)
        return f.name
