# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:14:19 2024

@author: Julian
"""

# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "gumbel_barnett",
    ext_modules=cythonize("gumbel_barnett.pyx"),
    include_dirs = [numpy.get_include()]
)
