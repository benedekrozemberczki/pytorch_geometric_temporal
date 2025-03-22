Installation
============

The installation of PyTorch Geometric Temporal requires the presence of certain prerequisites. These are described in great detail in the installation description of PyTorch Geometric. Please follow the instructions laid out `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_. You might also take a look at the `readme file <https://github.com/benedekrozemberczki/pytorch_geometric_temporal>`_ of the PyTorch Geometric Temporal repository.

Once the required versions of PyTorch and PyTorch Geometric are installed, simply run:

    .. code-block:: none

        $ pip install torch-geometric-temporal

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

**Index-Batching**

The package was recently updated to include index-batching, a new method of batching that improves 
memory efficiency without any impact on accuracy. To install the needed packages for index-batching,
run the following command:

    .. code-block:: none

        $ pip install torch-geometric-temporal[index]

**Distributed Data Parallel**

Alongside index-batching, PGT was recently updated with features to support distributed data parallel training.
To install the needed packages, run the following: 

    .. code-block:: none

        $ pip install torch-geometric-temporal[ddp]

