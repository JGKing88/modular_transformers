#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch",
    "numpy",
    "pytest",
    "black",
    "tiktoken"
]

test_requirements = [
    "pytest",
    "pytest-timeout",
]

setup(
    name='modular_transformers',
    version='0.1.0',
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