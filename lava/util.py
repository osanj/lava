# -*- coding: UTF-8 -*-

import logging
import os
import platform
import subprocess

from lava import ENV_VAR_SDK

logger = logging.getLogger(__name__)

__all__ = ["compile_glsl"]

EXT_SPIR_V = ".spv"


def compile_glsl(path, verbose=True):
    path_output = path + EXT_SPIR_V

    # Linux
    if platform.system() == "Linux":
        if ENV_VAR_SDK not in os.environ:
            raise RuntimeError("Could not find environment variable {}".format(ENV_VAR_SDK))

        path_compiler = os.path.join(os.environ["VULKAN_SDK"], "bin", "glslangValidator")
        cmd = [path_compiler, "-V", path, "-o", path_output]

        if verbose:
            result = subprocess.call(cmd, stderr=subprocess.STDOUT)
        else:
            dev_null = open(os.devnull, "w")
            result = subprocess.call(cmd, stdout=dev_null, stderr=subprocess.STDOUT)
            dev_null.close()

        if result != 0:
            raise RuntimeError("Could not compile shader {}, try yourself with\n{}".format(path, " ".join(cmd)))

    # Windows
    elif platform.system() == "Windows":
        raise NotImplementedError()

    else:
        raise NotImplementedError()

    if verbose:
        logger.info("Compiled shader {}".format(path))
    return path_output
