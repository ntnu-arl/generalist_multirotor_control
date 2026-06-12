from setuptools import setup, find_packages

setup(
    name="generalist_multirotor_control",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "torch", "rl-games"],
    author="Orestis Konstantaropoulos",
    author_email="orestiskonsta@gmail.com",
    description="A package for generalist multirotor control and RL training",
    long_description_content_type="text/markdown",
    url="https://github.com/ntnu-arl/generalist_multirotor_control",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires=">=3.7",

)
