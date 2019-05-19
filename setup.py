import os
import sys
import subprocess

import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# All directories containing Cython code
extensions_dir = ['pymaat/', 'pymaat/garch/spec/', ]

setup(
    name="pymaat",
    version="0.0.0",
    author="Hugo Lamarre",
    author_email="hugolamarre.phd@gmail.com",
    description="Financial volatility modelling and hedging toolbox",
    license="MIT",
    keywords="quantitative finance garch hedging risk",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers="""
            Development Status :: 1 - Planning
            Intended Audience :: Science/Research
            License :: OSI Approved :: MIT License
            Programming Language :: Python :: 3 :: Only
            Programming Language :: C
            Programming Language :: Cython
            Topic :: Scientific/Engineering :: Mathematics
            """,
    project_urls={
            "Source Code": "https://github.com/hugolamarrephd/pymaat"
    },
    setup_requires='cython',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'zignor',
        'scipy',
        'tabulate',
        'matplotlib',
        'cachetools'
    ],
    ext_modules=cythonize(
        [x + '*.pyx' for x in extensions_dir],
    ),
    include_dirs=[np.get_include()]
)
