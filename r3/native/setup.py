from setuptools import setup, Extension
import pybind11
setup(name='r3_diagnostic',version='1.0',ext_modules=[Extension('r3_diagnostic',
    ['two_opt_diagnostic.cpp'],include_dirs=[pybind11.get_include()],language='c++',
    extra_compile_args=['-O3','-std=c++17'])])
