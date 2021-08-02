from setuptools import setup, find_packages
from setuptools.config import read_configuration

extras = {
    "train": ["pytorch-lightning", "torchmetrics"],
}
install_requires = [
    "torch",
    "torchvision",
    "einops",
    "numpy",
    "pillow",
    "tifffile",
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
