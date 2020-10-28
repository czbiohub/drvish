#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import io
import glob
import os

import setuptools


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ).read()


setuptools.setup(
    name="drvish",
    version="0.0.2",
    license="MIT License",
    description="scRNA + deep learning + drug response",
    long_description=read("README.md"),
    author="James Webber",
    author_email="james.webber@czbiohub.org",
    url="https://github.com/czbiohub/drvish",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")
    ],
    zip_safe=False,
    install_requires=["numpy", "torch", "cupy", "pyro-ppl", "simscity"],
    extras_require={"dev": ["pytest", "black"], "plot": ["altair"]},
)
