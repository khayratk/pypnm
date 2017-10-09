from setuptools import setup, find_packages

import os
import sys

sys.path.append(os.getcwd())

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
        'petsc',
        'petsc4py',
        'colorama',
        'python-igraph',
        'pymetis',
        'pandas',
        'mpi4py',
        'h5py',
        'pandas',
        'cython',
    ],
    package_data={'': ['*.c', '*.pyx'], },
    author='Karim Khayrat',
    author_email='kkhayrat@gmail.com',
    description='Pore Network Modeling package'
)
