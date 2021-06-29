Installation
============

The installation of PyTorch Geometric Temporal requires the presence of certain prerequisites. These are described in great detail in the installation description of PyTorch Geometric. Please follow the instructions laid out `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. You might also take a look at the `readme file <https://github.com/benedekrozemberczki/pytorch_geometric_temporal>`_ of the PyTorch Geometric Temporal repository.
Binaries are provided for Python version <= 3.9.

**PyTorch 1.9.0**

To install the binaries for PyTorch 1.9.0, simply run

    .. code-block:: none

        $ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
        $ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
        $ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
        $ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
        $ pip install torch-geometric
        $ pip install torch-geometric-temporal


where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu111` depending on your PyTorch installation.


**PyTorch 1.8.0**

To install the binaries for PyTorch 1.8.0, simply run

    .. code-block:: none

        $ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
        $ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
        $ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
        $ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
        $ pip install torch-geometric
        $ pip install torch-geometric-temporal


where `${CUDA}` should be replaced by either `cpu`, `cu101`, `cu102`, or `cu111` depending on your PyTorch installation.

**PyTorch 1.7.0**

To install the binaries for PyTorch 1.7.0, simply run

    .. code-block:: none

        $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
        $ pip install torch-geometric
        $ pip install torch-geometric-temporal


where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102` or `cu110` depending on your PyTorch installation.


**PyTorch 1.6.0**

To install the binaries for PyTorch 1.6.0, simply run

    .. code-block:: none

        $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
        $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
        $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
        $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
        $ pip install torch-geometric
        $ pip install torch-geometric-temporal

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.

**Updating the Library**

The package itself can be installed via pip:

    .. code-block:: none

        $ pip install torch-geometric-temporal

Upgrade your outdated PyTorch Geometric Temporal version by using:

    .. code-block:: none

        $ pip install torch-geometric-temporal --upgrade


To check your current package version just simply run:

    .. code-block:: none

        $ pip freeze | grep torch-geometric-temporal

