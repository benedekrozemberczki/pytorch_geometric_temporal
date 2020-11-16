Introduction
=======================

`*PyTorch Geometric Temporal* <https://github.com/benedekrozemberczki/pytorch_geometric_temporal>`_ is an temporal graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_. It builds on open-source deep-learning and graph processing libraries. *PyTorch Geometric Temporal* consists of state-of-the-art deep learning and parametric learning methods to process spatio-temporal signals. It is the first open-source library for temporal deep learning on geometric structures. First, it provides discrete time graph neural networks on dynamic and static graphs. Second, it allows for spatio-temporal learning when the time is represented continuously without the use of discrete snapshots. Implemented methods cover a wide range of data mining (`WWW <https://www2021.thewebconf.org/>`_, `KDD <https://www.kdd.org/kdd2020/>`_), artificial intelligence and machine learning (`AAAI <http://www.aaai.org/Conferences/conferences.php>`_, `ICONIP <https://www.apnns.org/ICONIP2020/>`_, `ICLR <https://iclr.cc/>`_) conferences, workshops, and pieces from prominent journals. 
 

Citing
=======================
If you find *PyTorch Geometric Temporal* useful in your research, please consider adding the following citation:

.. code-block:: latex

    >@misc{pytorch_geometric_temporal,
           author = {Benedek, Rozemberczki and Paul, Scherer},
           title = {{PyTorch Geometric Temporal}},
           year = {2020},
           publisher = {GitHub},
           journal = {GitHub repository},
           howpublished = {\url{https://github.com/benedekrozemberczki/pytorch_geometric_temporal}},
    }

We briefly overview the fundamental concepts and features of PyTorch Geometric Temporal through simple examples.

Data Structures
=============================

Discrete Dataset Iterators
--------------------------

PyTorch Geometric Tenporal offers data iterators for discrete time datasets which contain the temporal snapshots. There are two types of discrete time data iterators:

- ``StaticGraphDiscreteSignal`` - Is designed for discrete spatio-temporal signals defined on a **static** graph.
- ``DynamicGraphDiscreteSignal`` - Is designed for discrete spatio-temporal signals defined on a **dynamic** graph.


Static Graphs with Discrete Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor of a ``StaticGraphDiscreteSignal`` object requires the following parameters:

- ``edge_index`` - A **single** ``NumPy`` array to hold the edge indices.
- ``edge_weight`` - A **single** ``NumPy`` array to hold the edge weights.
- ``features`` - A **list** of ``NumPy`` arrays to hold the vertex features for each time period.
- ``targets`` - A **list** of ``NumPy`` arrays to hold the vertex level targets for each time period.
 
Static Graphs with Dynamic Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The constructor of a ``DynamicGraphDiscreteSignal`` object requires the following parameters:

- ``edge_indices`` - A **list** of ``NumPy`` arrays to hold the edge indices.
- ``edge_weights`` - A **list** of ``NumPy`` arrays to hold the edge weights.
- ``features`` - A **list** of ``NumPy`` arrays to hold the vertex features for each time period.
- ``targets`` - A **list** of ``NumPy`` arrays to hold the vertex level targets for each time period.

Temporal Snapshots
^^^^^^^^^^^^^^^^^^ 

A discrete temporal snapshot is a PyTorch Geometric ``Data`` object. Please take a look at this `readme <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_ for the details. The returned temporal snapshot has the following attributes:

- ``edge_index`` - A PyTorch ``LongTensor`` of edge indices used for node feature aggregation (optional).
- ``edge_attr`` - A PyTorch ``FloatTensor`` of edge features used for weighting the node feature aggregation (optional).
- ``x`` - A PyTorch ``FloatTensor`` of vertex features (optional).
- ``y`` - A PyTorch ``FloatTensor`` or ``LongTensor`` of vertex targets (optional).

Benchmark Datasets
-------------------

We released and included a number of datasets which can be used for comparing the performance of temporal graph neural networks algorithms. The related machine learning tasks are node and graph level supervised learning.

Discrete Time Datasets
^^^^^^^^^^^^^^^^^^^^^^
In case of discrete time graph neural networks these datasets are as follows:

- `Hungarian Chickenpox Dataset. <https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/dataset.html#torch_geometric_temporal.data.dataset.chickenpox.ChickenpoxDatasetLoader>`_


The Hungarian Chickenpox Dataset can be loaded by the following code snippet. The ``dataset`` returned by the public ``get_dataset`` method is a ``StaticGraphDiscreteSignal`` object. 

.. code-block:: python

    from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader

    loader = ChickenpoxDatasetLoader()

    dataset = loader.get_dataset()

Train-Test Splitter
-------------------


Discrete Train-Test Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide functions to create temporal splits of the discrete time iterators. These functions return train and test data iterators which split the original iterator using a fix ratio. Snapshots from the earlier time periods from the training dataset and snapshots from the later periods form the test dataset. This way temporal forecasts can be evaluated in a real life like scenario. The function ``discrete_train_tes_split`` takes either a ``StaticGraphDiscreteSignal`` or a ``DynamicGraphDiscreteSignal`` and returns two iterattors according to the split ratio specified by ``train_ratio``.

.. code-block:: python

    from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
    from torch_geometric_temporal.data.splitter import discrete_train_test_split

    loader = ChickenpoxDatasetLoader()

    dataset = loader.get_dataset()

    train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.8)



Applications
=============

In the following we will overview two case studies where PyTorch Geometric Temporal can be used to solve real world relevant machine learning problems. One of them is on discrete time spatial data and the other one uses continuous time graphs.   

Learning from a Discrete Temporal Signal
-------------------------------------------

We are using the Hungarian Chickenpox Cases dataset in this case study. We will train a regressor to predict the weekly cases reported by the counties using a recurrent graph convolutional network. First, we will load the dataset and create an appropriate spatio-temporal split.

.. code-block:: python

    from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader
    from torch_geometric_temporal.data.splitter import discrete_train_test_split

    loader = ChickenpoxDatasetLoader()

    dataset = loader.get_dataset()

    train_dataset, test_dataset = discrete_train_test_split(dataset, train_ratio=0.2)

In the next step we will define the recurrent graph neural network. 


Learning from a Continuous Temporal Signal
-------------------------------------------

 
