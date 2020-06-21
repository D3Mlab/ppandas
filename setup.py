import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PPandas",
    version="0.0.1.7.1",
    author="Amy Sui, Alex Kwan",
    author_email="suiyiamy@gmail.com, alex.kwan@mail.utoronto.ca",
    description="A python tool for merging different datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/echoyi/ppandas",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],
)
