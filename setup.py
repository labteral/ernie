#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages
import ernie

setup(
    name='ernie',
    version=ernie.__version__,
    description=(
        'An Accessible Python Library for State-of-the-art '
        'Natural Language Processing. Built with HuggingFace\'s Transformers.'
    ),
    url='https://github.com/brunneis/ernie',
    author='Rodrigo Martínez Castaño',
    author_email='rodrigo@martinez.gal',
    license='Apache License (Version 2.0)',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[
        'transformers>=2.4.1, < 2.5.0',
        'scikit-learn>=0.22.1, < 1.0.0',
        'pandas>=0.25.3, < 1.0.0',
        'tensorflow>=2.5.1, < 2.6.0',
        'py-cpuinfo>=5.0.0, < 6.0.0'
    ])
