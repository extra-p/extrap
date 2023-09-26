# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2023, Technical University of Darmstadt, Germany
#
# This software may be modified and distributed under the terms of a BSD-style license.
# See the LICENSE file in the base directory for details.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

info = {}
with open("extrap/__init__.py") as fp:
    exec(fp.read(), info)

setup(
    name="extrap",
    version=info['__version__'],
    packages=find_packages(include=('extrap', 'extrap.*')),
    author="Extra-P project",
    author_email="extra-p@lists.parallel.informatik.tu-darmstadt.de",
    description=info['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/extra-p/extrap",
    entry_points={
        "console_scripts": [
            "extrap = extrap.extrap.extrapcmd:main",
        ],

        "gui_scripts": [
            "extrap-gui = extrap.extrap.extrapgui:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    install_requires=[
        "pyside6~=6.4",
        "numpy~=1.18",
        "matplotlib~=3.5",
        "tqdm~=4.47",
        "pycubexr>=1.1,<3",
        "marshmallow~=3.7",
        "packaging>=20.0,<22",
        "kaitaistruct~=0.9",
        "protobuf~=3.14",
        "itanium_demangler~=1.0",
        "sympy~=1.8",
        "typing-extensions~=4.3.0",
        "msgpack~=1.0.3",
        "scipy~=1.5",
        "pyobjc-framework-Cocoa~=9.0; sys_platform == 'darwin'"
    ]
)
