from setuptools import find_packages, setup

install_requires = [
    "decorator==4.4.2",
    "torch",
    "cython",
    "torch_geometric",
    "numpy",
    "networkx",
]
tests_require = ["pytest", "pytest-cov", "mock", "networkx", "tqdm",'dask', "pandas", "tables", "scipy"]
index_require = ['dask', "pandas", "tables"]
ddp_require = ["dask[distributed]", "dask_pytorch_ddp", "pandas", "tables"]

keywords = [
    "machine-learning",
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
    "learning",
]

setup(
    name="torch_geometric_temporal",
    packages=find_packages(),
    version="0.56.2",
    license="MIT",
    description="A Temporal Extension Library for PyTorch Geometric.",
    author="Benedek Rozemberczki",
    author_email="benedek.rozemberczki@gmail.com",
    url="https://github.com/benedekrozemberczki/pytorch_geometric_temporal",
    download_url="https://github.com/benedekrozemberczki/pytorch_geometric_temporal/archive/v0.54.0.tar.gz",
    keywords=keywords,
    install_requires=install_requires,
    extras_require={
        "test": tests_require,
        "index": index_require,
        "ddp": ddp_require
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
