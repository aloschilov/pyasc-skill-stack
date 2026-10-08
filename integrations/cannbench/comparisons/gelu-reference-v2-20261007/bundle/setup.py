"""The build entrypoint installs the verified self-contained CPython 3.12 wheel."""
from setuptools import setup, find_packages

setup(name='cann_bench', version='1.1.0', packages=find_packages(),
      description='Reference-DAG GeLU on pyasc v2 9069108e', python_requires='>=3.12')
