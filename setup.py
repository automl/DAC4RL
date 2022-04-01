from setuptools import setup

packages = ["dac4rl"]

package_data = {"": ["*"]}

AUTHORS = (
    ", ".join(
        [
            "Raghu Rajan",
            "Aditya Mohan",
            "Theresa Eimer",
        ]
    ),
)

AUTHOR_EMAIL = (
    ", ".join(
        [
            "rajanr@cs.uni-freiburg.de",
            "mohan@tnt.uni-hannover.de",
            "eimer@tnt.uni-hannover.de",
        ]
    ),
)

# TODO
LICENSE = "Apache License, Version 2.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dac4rl",
    version="0.0.1",
    package_dir={"carl": "carl"},
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    description="A python package for the DAC4RL competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    # url="https://github.com/automl/", #TODO
    project_urls={
        "Bug Tracker": "https://github.com/automl/#TODO/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",  # TODO
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # package_dir={"": "src"},
    python_requires=">=3.9",
    setup_requires=[""],
    install_requires=[
        "gym>=0.18.3",
        "scipy>=1.7.0",
        "ConfigArgParse>=1.5.1",
        "numpy>=1.19.5",
        "pandas>=1.3.0",
        "matplotlib>=3.4.2",
        "optuna>=2.9.1",
        "dataclasses>=0.6",
        "numpyencoder>=0.3.0",
        "pyglet>=1.5.15",
        "pytablewriter>=0.62.0",
        "PyYAML>=5.4.1",
        "tabulate>=0.8.9",
        "box2d-py>=2.3.5",
        "sb3_contrib>=1.4.0",
        "stable_baselines3>=1.1.0",
    ],
    extras_require={
        "baselines": [
            "ray[tune]>=1.1.0",
        ],
        "experiments": ["seaborn>=0.11.1", "tensorflow>=2.5.0"],
    },
)
