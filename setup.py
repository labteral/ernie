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
    url='https://github.com/labteral/ernie',
    author='Rodrigo Martínez Castaño',
    author_email='dev@brunneis.com',
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
        'transformers>=4.24.0, <5.0.0',
        'scikit-learn>=1.2.1, <2.0.0',
        'pandas>=1.5.3, <2.0.0',
        'tensorflow>=2.5.1, <2.11.0',
        'py-cpuinfo>=9.0.0, <10.0.0',
    ])
