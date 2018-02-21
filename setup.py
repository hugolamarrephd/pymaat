import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extensions = ['pymaat/garch/spec/*_core.pyx',]

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
                'scipy',
                'tabulate',
                'matplotlib',
                'cachetools'
            ],
        ext_modules=cythonize(extensions),
    )

