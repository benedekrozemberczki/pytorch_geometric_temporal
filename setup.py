from setuptools import find_packages, setup

install_requires = ["decorator==4.4.2",
                    "torch",
                    "torch_sparse",
                    "torch_scatter",
                    "torch_cluster",
                    "torch_spline_conv",
                    "torch_geometric",
                    "numpy",
                    "scipy==1.5.4",
                    "tqdm",
                    "six"]

setup_requires = ['pytest-runner']

tests_require = ['pytest',
                 'pytest-cov',
                 'mock']

keywords = ["machine-learning",
            "deep-learning",
            "deeplearning",
            "deep learning",
            "machine learning",
            "signal processing",
            "temporal signal",
            "graph",
            "dynamic graph",
            "embedding",
            "dynamic embedding",
            "graph convolution",
            "gcn",
            "graph neural network",
            "graph attention",
            "lstm",
            "temporal network",
            "representation learning",
            "learning"]

setup(
  name = "torch_geometric_temporal",
  packages = find_packages(),
  version = "0.33",
  license = "MIT",
  description = "A Temporal Extension Library for PyTorch Geometric.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/pytorch_geometric_temporal",
  download_url = "https://github.com/benedekrozemberczki/pytorch_geometric_temporal/archive/v_00033.tar.gz",
  keywords = keywords,
  install_requires = install_requires,
  setup_requires = setup_requires,
  tests_require = tests_require,
  python_requires = '>=3.6',
  classifiers = ["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "Topic :: Software Development :: Build Tools",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.6"],
)
