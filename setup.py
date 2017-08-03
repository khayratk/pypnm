from distutils.core import setup, find_packages

import os
import sys
import pypnm

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='pypnm',
    version='0.1',
    packages=find_packages(),
    url='https://bitbucket.org/kkhayrat/porenetwork',
    license='',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pyevtk',
        'scikit-umfpack',
        'petsc4py',
        'colorama',
        'python-igraph',
        'pymetis'
    ],
    author='Karim Khayrat',
    author_email='kkhayrat@gmail.com',
    description='Pore Network Model'
)
