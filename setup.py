from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

info = {}
with open("src/__info__.py") as fp:
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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=["pyside2", "numpy", "matplotlib", "tqdm", "pycubexr", "marshmallow"],
)
