import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soraxas_toolbox",
    version="0.1.3",
    author="soraxas",
    author_email="oscar@tinyiu.com",
    description="A simple packaged toolbox for various scenario",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soraxas/pytoolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
