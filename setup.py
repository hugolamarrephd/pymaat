import os, sys, subprocess
import numpy as np
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# All directories containing Cython code
extensions_dir = ['pymaat/','pymaat/garch/spec/',]

args = sys.argv[1:]

setup(
        name="pymaat",
        version="0.0.0",
        author="Hugo Lamarre",
        author_email="hugolamarre.phd@gmail.com",
        description="Financial portfolio management toolbox",
        license="MIT",
        keywords="quantitative finance garch hedging risk allocation",
        packages=find_packages(),
        long_description=read('README.md'),
        classifiers="""
            Development Status :: 1 - Planning
            Intended Audience :: Science/Research
            License :: OSI Approved :: MIT License
            Programming Language :: Python :: 3 :: Only
            Programming Language :: C
            Programming Language :: Cython
            Topic :: Scientific/Engineering :: Artificial Intelligence
            Topic :: Scientific/Engineering :: Mathematics
            Topic :: Scientific/Engineering :: Visualization
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

# Clean
if "clean" in args:
    subprocess.Popen("rm -rf prof/",
            shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf htmlcov/",
            shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf build/",
            shell=True, executable="/bin/bash")
    for x in extensions_dir:
        subprocess.Popen("rm " + x + "*.html",
                shell=True, executable="/bin/bash")
        subprocess.Popen("rm " + x + "*.c",
                shell=True, executable="/bin/bash")
        subprocess.Popen("rm " + x + "*.so",
                shell=True, executable="/bin/bash")
