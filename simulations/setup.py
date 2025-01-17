from setuptools import setup, find_packages

setup(
    name="simulations",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.15.0",
        "tqdm",
        "tskit",
        "msprime",
        "pydantic",
        "oyaml",
        "numpy==1.26.4",
        "fire",
    ]
)