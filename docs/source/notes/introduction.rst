Introduction
=======================

`*PyTorch Geometric Temporal* <https://github.com/benedekrozemberczki/pytorch_geometric_temporal>`_ is an temporal graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_. It builds on open-source deep-learning and graph processing libraries. *PyTorch Geometric Temporal* consists of state-of-the-art deep learning and parametric learning methods to process spatio-temporal signals. It is the first open-source library for temporal deep learning on geometric structures. First, it provides discrete time graph neural networks on dynamic and static graphs. Second, it allows for spatio-temporal learning when the time is represented continuously without the use of discrete snapshots. Implemented methods cover a wide range of data mining (`WWW <https://www2021.thewebconf.org/>`_, `KDD <https://www.kdd.org/kdd2020/>`_), artificial intelligence and machine learning (`AAAI <http://www.aaai.org/Conferences/conferences.php>`_, `ICONIP <https://www.apnns.org/ICONIP2020/>`_, `ICLR <https://iclr.cc/>`_) conferences, workshops, and pieces from prominent journals. 
 

--------------------------------------------------------------------------------

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

Overview
=========
--------------------------------------------------------------------------------

We shortly overview the fundamental concepts and features of PyTorch Geometric Temporal through simple examples. These are the following:

.. contents::
    :local:

Data Structures and Splitting
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

A discrete temporal snapshot is a PyTorch Geometric ``Data`` object. The returned temporal snapshot has the following attributes:

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

- `Hungarian Chickenpox Dataset. <https://arxiv.org/abs/2005.07959>`_


The Hungarian Chickenpox Dataset (which is represented as a ``StaticGraphDiscreteSignal`` object) can be loaded by the following code snippet. 

.. code-block:: python

    from torch_geometric_temporal.data.dataset import ChickenpoxDatasetLoader

    loader = ChickenpoxDatasetLoader()

    dataset = loader.get_dataset()

Train-Test Splitter
-----------------

Discrete Train-Test Splitter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Applications
=============

Learning from a Discrete Temporal Signal
----------------------------------------

The third machine learning task that we look at is the classification of threads from the online forum Reddit. The threads
can be of of two types - discussion and non-discussion based ones. Our goal is to predict the type of the thread based on
the topological (structural) properties of the graphs. The specific dataset that we look a 10 thousand graph subsample of
the Reddit 204K dataset which contains a large number of threads from the spring of 2018. The graphs in the dataset do not
have a specific feature. Because of this we use the degree centrality as a string feature.
For details about the dataset `see this paper <https://arxiv.org/abs/2003.04819>`_.

We first need to load the Reddit 10K dataset. We will use the use the graphs and the discussion/non-discussion target vector.
These are returned as a list of ``NetworkX`` graphs and ``numpy`` array respectively.

.. code-block:: python

    from karateclub.dataset import GraphSetReader

    reader = GraphSetReader("reddit10k")

    graphs = reader.get_graphs()
    y = reader.get_target()

We fit a FEATHER graph level embedding, with the standard hyperparameter settings. These are pretty widely used settings.
First, we use the model constructor without custom parameters. Second, we fit the model to the graphs. Third, we get the graph embedding
which is a ``numpy`` array.

.. code-block:: python

    from karateclub import FeatherGraph

    model = FeatherGraph()
    model.fit(graphs)
    X = model.get_embedding()

We use the graph embedding features as predictors of the thread type. So let us create a train-test split of the explanatory variables
and the target variable with Scikit-Learn. We will use a test data ratio of 20%. Here it is.

.. code-block:: python

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Using the training data (``X_train`` and ``y_train``) we learn a logistic regression model to predict the probability of a thread being discussion based. We perform inference on the test 
set for this target. Finally, we evaluate the model performance by printing an area under the ROC curve value.

.. code-block:: python

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    
    downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_hat)
    print('AUC: {:.4f}'.format(auc))
    >>> AUC: 0.7127

Learning from a Continuous Time Signal
--------------------------------------

 
