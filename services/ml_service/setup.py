from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements(path: Path):
    """
    Parses the requirements.txt file and returns a list of installed
    package names.

    Args:
        path (pathlib.Path):
            The path to the requirements.txt file.

    Returns:
        list[str]:
            A list of installed package names, excluding commented-out
            lines.
    """
    with open(path, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="ml_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=parse_requirements(Path("requirements.txt")),
    package_data={
        "ml_service": ["config/*.yaml", "params/*.json"],
    },
    include_package_data=True,
    author="Egor Udalov",
    author_email="egor.udalov13@gmail.com",
    description="Scripts for a service with Recommendation System.",
    url="https://github.com/GitHub-User228/mle-project-sprint-3-v001",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10, <3.11",
)
