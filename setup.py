from setuptools import find_packages, setup

install_requires = ["networkx",
                    "tqdm",
                    "python-louvain",
                    "pandas",
                    "numpy",
                    "six",
                    "scipy"]

setup_requires = ['pytest-runner']

tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
  name = "torch_geometric_temporal",
  packages = find_packages(),
  version = "0.0.1",
  license = "MIT",
  description = "A general purpose library for subsampling large graphs.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/pytorch_geometric_temporal",
  download_url = "https://github.com/benedekrozemberczki/pytorch_geometric_temporal/archive/v_00001.tar.gz",
  keywords = ["machine-learning", "deep-learning", "deeplearning"],
  install_requires = install_requires,
  setup_requires = setup_requires,
  tests_require = tests_require,
  classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
  ],
)
