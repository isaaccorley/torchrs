from setuptools import setup, find_packages
from setuptools.config import read_configuration

extras = {
    "train": ["pytorch-lightning>=1.4.0", "torchmetrics>=0.4.1"],
}
install_requires = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "torchaudio>=0.9.0",
    "einops>=0.3.0",
    "numpy>=1.21.0",
    "pillow>=8.3.1",
    "tifffile>=2021.7.2",
    "h5py>=3.3.0",
    "imagecodecs>=2021.7.30"
]
setup_requires = ["pytest-runner"]
tests_require = ["pytest", "pytest-cov", "mock", "mypy", "black", "pylint"]

cfg = read_configuration("setup.cfg")

setup(
    download_url='{}/archive/{}.tar.gz'.format(cfg["metadata"]["url"], cfg["metadata"]["version"]),
    project_urls={"Bug Tracker": cfg["metadata"]["url"] + "/issues"},
    install_requires=install_requires,
    extras_require=extras,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
    python_requires=">=3.7",
)
