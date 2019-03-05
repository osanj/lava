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
    # install_requires=['cffi>=1.10'],
    # setup_requires=["vulkan"],
    # url="https://github.com/osanj/lava",
    # keywords="Vulkan",
    # classifiers=[
    #     "Development Status :: 5 - Production/Stable",
    #     "Intended Audience :: Developers",
    #     "License :: OSI Approved :: Apache Software License",
    #     "Operating System :: Android",
    #     "Operating System :: Microsoft :: Windows",
    #     "Operating System :: POSIX :: Linux",
    #     "Natural Language :: English",
    #     "Topic :: Multimedia :: Graphics",
    #     "Topic :: Scientific/Engineering",
    #     "Topic :: Software Development :: Libraries :: Python Modules",
    # ]
)
