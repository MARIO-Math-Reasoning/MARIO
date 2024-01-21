import os
from typing import List, Set
import setuptools


ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="math_evaluation",
    description="A utility for determining whether 2 answers for a problem in the MATH dataset are equivalent.",
    url="https://code.alibaba-inc.com/mti_llm/math_evaluation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=("math_evaluation", "math_evaluation.*")),
    install_requires=get_requirements(),
    py_modules = ["math_evaluation"],
    version='0.0.1',
)