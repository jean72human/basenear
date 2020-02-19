import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="AutoML",
    version="0.1",
    author="Gbetondji Dovonon",
    author_email="",
    description="Bayesian Search of Neural Architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jean72human/basenear",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)