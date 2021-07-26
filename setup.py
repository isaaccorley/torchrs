from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().strip().splitlines()

setup_requires = ["pytest-runner"]
tests_require = ["pytest", "pytest-cov", "mock", "mypy", "black", "pylint"]

url = "https://github.com/isaaccorley/torchrs"
__version__ = "0.0.1"

setup(
    name="torch-rs",
    version="0.0.1",
    license='MIT License',
    author="Isaac Corley",
    author_email="isaac.corley@my.utsa.edu",
    description="PyTorch Library for Remote Sensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=["pytorch", "remote-sensing", "computer-vision"],
    project_urls={
        "Bug Tracker": "https://github.com/isaaccorley/torchrs/issues",
    },
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
)
