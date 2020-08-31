from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

info = {}
with open("src/__info__/__init__.py") as fp:
    exec(fp.read(), info)

setup(
    name="extrap",
    version=info['__version__'],
    package_dir={"": "src"},
    packages=find_packages("src"),
    author="Extra-P project",
    author_email="extra-p@lists.parallel.informatik.tu-darmstadt.de",
    description=info['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/extra-p/extrap",
    entry_points={
        "console_scripts": [
            "extrap = extrap.extrap:main",
        ],

        "gui_scripts": [
            "extrap-gui = extrap.extrapgui:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7',
    install_requires=["pyside2~=5.15", "numpy~=1.18", "matplotlib~=3.2", "tqdm~=4.47", "pycubexr~=1.1",
                      "marshmallow~=3.7"],
)
