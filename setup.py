import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Img",
    version="0.1.0",
    author="Ushakov Alexey",
    author_email="kafka.pochta@gmail.com",
    description="Image utils and helpers",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dearkafka/FaceEditor",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    zip_safe=False,
    platforms='any'
)
