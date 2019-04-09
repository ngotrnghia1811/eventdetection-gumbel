from setuptools import setup, find_packages

setup(
    name="eventdetection-gumbel",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.0.0",
        "numpy>=1.16.0",
        "scipy>=1.2.0",
        "gensim>=3.7.0",
        "pyyaml>=3.13",
        "tqdm>=4.31.0",
    ],
)
