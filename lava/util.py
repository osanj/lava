# -*- coding: UTF-8 -*-

import os
import platform
import subprocess

from lava import ENV_VAR_SDK

__all__ = ["compile_glsl"]

EXT_SPIR_V = ".spv"


def compile_glsl(path, verbose=True):
    path_output = path + EXT_SPIR_V

    if platform.system() not in ("Windows", "Linux"):
        raise NotImplementedError()

    if ENV_VAR_SDK not in os.environ:
        raise RuntimeError("Could not find environment variable {}".format(ENV_VAR_SDK))

    if platform.system() == "Linux":
        path_compiler = os.path.join(os.environ[ENV_VAR_SDK], "bin", "glslangValidator")
    else:
        path_compiler = os.path.join(os.environ[ENV_VAR_SDK], "Bin", "glslangValidator.exe")

    cmd = [path_compiler, "-V", path, "-o", path_output]

    if verbose:
        result = subprocess.call(cmd, stderr=subprocess.STDOUT)
    else:
        dev_null = open(os.devnull, "w")
        result = subprocess.call(cmd, stdout=dev_null, stderr=subprocess.STDOUT)
        dev_null.close()

    if result != 0:
        raise RuntimeError("Could not compile shader {}, try yourself with\n{}".format(path, " ".join(cmd)))

    return path_output
