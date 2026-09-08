"""Source layout metadata; build.sh supplies the byte-verified runtime wheel."""
from setuptools import find_packages, setup

setup(name='cann_bench', version='1.1.0',
      description='Corrected target-style GeLU, pyasc v2 adadd7d66ed0ee16d33d79487bf584899a26ef1e',
      packages=find_packages(include=['cann_bench','cann_bench.*']),
      python_requires='>=3.12')
