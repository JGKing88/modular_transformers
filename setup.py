#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "accelerate",
    "torch==1.13.1",
    "numpy==1.23.5",
    "pytest==7.2.2",
    "black==23.1.0",
    "tiktoken==0.3.1",
    "xarray",
    "torchvision",
    "torchaudio",
    "transformers==4.27.0",
    "sklearn",
    "netCDF4",
    "h5netcdf",
    "wandb",
    "openpyxl",
    "datasets",
    "chardet",
    "cchardet",
    "deepspeed",
]

test_requirements = [
    "pytest",
    "pytest-timeout",
]

setup(
    name='modular_transformers',
    version='0.0.0',
    description="Using modular transformers to explore human language processing",
    long_description=readme,
    author="Jack King",
    author_email='jackking@mit.edu',
    url='https://github.com/JGKing88/modular_transformers',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='computational neuroscience, human language, '
             'machine learning, deep neural networks, transformers',
    test_suite='tests',
    tests_require=test_requirements
)