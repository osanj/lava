# -*- coding: UTF-8 -*-

from setuptools import setup

import lava as lv


setup(
    name="lava",
    version=lv.__version__,
    description="Highlevel Wrapper for Vulkan's Compute API",
    author="Jonas Schuepfer",
    author_email="jonasschuepfer@gmail.com",
    packages=["lava", "lava.api", "lava.api.constants"],
    include_package_data=True,
    install_requires=["vulkan", "numpy", "future"],
    url="https://github.com/osanj/lava",
    keywords=["Vulkan", "Parallel Computing", "Numpy"]
)
