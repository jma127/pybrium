from setuptools import setup, find_packages

from pybrium import __version__


setup(
    name="pybrium",
    version=__version__,
    description="Python library for calculations involving strategies and equilibria",
    author="Jerry Ma",
    url="https://github.com/jma127/pybrium",
    license="BSD-3-Clause",
    packages=find_packages(exclude=["test_*"]),
    data_files=[("source_docs/qhoptim", ["LICENSE", "README.rst"])],
    install_requires=["torch>=1"],
    zip_safe=True,
)
