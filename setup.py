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
    install_requires=[
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "networkx>=3.2",
    "pgmpy>=0.1.26",
    "shapely>=2.0",
    "geopandas>=0.14",
    "Rtree>=1.2",
    "geovoronoi>=0.4",
    "intervals>=0.9",
    "matplotlib>=3.8",
],
)
