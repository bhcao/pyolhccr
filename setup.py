from setuptools import find_packages, setup

# Package metadata
NAME = "pyolhccr"
VERSION = "0.0.1"
DESCRIPTION = "Python library and tools for online handwritten Chinese character recognition"
URL = "https://github.com/bhcao/pyolhccr"
AUTHOR = "Bohan Cao"
AUTHOR_EMAIL = "2110313@mail.nankai.edu.cn"
LICENSE = "GPL-3.0"

# Read the contents of README file
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "matplotlib>=3.7.0"
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages("./src"),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0"
)