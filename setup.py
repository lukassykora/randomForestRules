import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randomForestRules-lukassykora",
    version="1.1.2",
    author="Lukas Sykora",
    author_email="lukassykora@seznam.cz",
    description="Random forest classification rules mining package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukassykora/randomForestRules",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'numpy',
        'sklearn',
        'typing',
    ],
)