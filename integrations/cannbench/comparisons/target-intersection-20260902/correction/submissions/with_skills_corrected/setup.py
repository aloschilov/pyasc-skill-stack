from setuptools import find_packages, setup


setup(
    name="cann_bench",
    version="1.1.0",
    description="Self-contained pyasc v2 operators for CANN Bench",
    packages=find_packages(include=("cann_bench", "cann_bench.*")),
    python_requires=">=3.10",
)
