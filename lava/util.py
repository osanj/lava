# -*- coding: UTF-8 -*-

import logging
import os
import platform
import subprocess

logger = logging.getLogger(__name__)


EXT_SPIR_V = ".spv"
ENV_SDK_NAME_LINUX = "VULKAN_SDK"


def compile_glsl(path, verbose=True):
    path_output = path + EXT_SPIR_V

    # Linux
    if platform.system() == "Linux":
        if ENV_SDK_NAME_LINUX not in os.environ:
            raise RuntimeError("Could not find environment variable {}".format(ENV_SDK_NAME_LINUX))

        path_compiler = os.path.join(os.environ["VULKAN_SDK"], "bin", "glslangValidator")
        cmd = [path_compiler, "-V", path, "-o", path_output]

        if verbose:
            result = subprocess.call(cmd, stderr=subprocess.STDOUT)
        else:
            dev_null = open(os.devnull, "w")
            result = subprocess.call(cmd, stdout=dev_null, stderr=subprocess.STDOUT)

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
