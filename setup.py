# This file is part of the Extra-P software (http://www.scalasca.org/software/extra-p)
#
# Copyright (c) 2020-2024, Technical University of Darmstadt, Germany
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
        "Programming Language :: Python :: 3.12",
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
        "PySide6-Essentials~=6.4",
        "numpy~=1.18",
        "matplotlib~=3.6",
        "tqdm~=4.47",
        "pycubexr>=1.2.3,<3,!=2.0.0",
        "marshmallow~=3.7",
        "packaging>=20.0",
        "pyobjc-framework-Cocoa~=9.0; sys_platform == 'darwin'",
        "scikit-learn>=1.2.2"
    ],
    extras_require={
        "adaptive_modeling": ["extrap-adaptive-modeler>=1.0.2,<3"],
    },
)
