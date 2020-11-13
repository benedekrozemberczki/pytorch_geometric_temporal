Introduction by example
=======================

`*PyTorch Geometric Temporal* <https://github.com/benedekrozemberczki/pytorch_geometric_temporal>`_ is an temporal graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_. It builds on open-source deep-learning and graph processing libraries. *PyTorch Geometric Temporal* consists of state-of-the-art deep learning and parametric learning methods to process spatio-temporal signals. It is the first open-source library for temporal deep learning on geometric structures. First, it provides discrete time graph neural networks on dynamic and static graphs. Second, it allows for spatio-temporal learning when the time is represented continuously without the use of discrete snapshots. Implemented methods cover a wide range of data mining (`WWW <https://www2021.thewebconf.org/>`_, `KDD <https://www.kdd.org/kdd2020/>`_), artificial intelligence and machine learning (`AAAI <http://www.aaai.org/Conferences/conferences.php>`_, `ICONIP <https://www.apnns.org/ICONIP2020/>`_, `ICLR <https://iclr.cc/>`_) conferences, workshops, and pieces from prominent journals. 
 

--------------------------------------------------------------------------------

**Citing**

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
=======================
--------------------------------------------------------------------------------

We shortly overview the fundamental concepts and features of PyTorch Geometric Temporal through simple examples. These are the following:

.. contents::
    :local:

Dataset Iterators
------------------------------

Karate Club assumes that the NetworkX graph provided by the user for node embedding and community detection has the following important properties:

- The graph is undirected.
- Nodes are indexed with integers.
- There are no orphaned nodes in the graph.
- The node indexing starts with zero and the indices are consecutive.

Node attribute matrices can be provided as ``scipy sparse`` and ``numpy`` arrays. 

The returned community membership dictionaries and embedding matrices use the same numeric, consecutive indexing.

API driven design
-----------------

Karate Club uses the design principles of Scikit-Learn which means that the algorithms in the package share the same API. Each machine learning model
is implemented as a class which inherits from ``Estimator``. The constructors of the models are used to set the hyperparameters. The models have
default hyperparameters that work well out of the box. This means that non expert users do not have to make decisions about these in advance and only a little fine tuning is required. For each class the ``fit`` method provided learns the embedding or clustering of nodes/graphs in the ``NetworkX`` graph. This method takes the data used for learn the embedding or clustering. Models provide the additional public methods ``get_embedding``, ``get_memberships``, ``get_cluster_centers``. This API driven design means that one can create a ``DeepWalk`` embedding of a Watts-Strogatz graph just like this.

.. code-block:: python

    import networkx as nx
    from karateclub import DeepWalk
    
    g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

    model = DeepWalk()
    model.fit(g)
    embedding = model.get_embedding()

This can be modified to create a ``Walklets`` embedding with minimal effort like this.

.. code-block:: python

    import networkx as nx
    from karateclub.node_embedding.neighbourhood import Walklets
    
    g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

    model = Walklets()
    model.fit(g)
    embedding = model.get_embedding()

Looking at these two snippets the advantage of the API driven design is evident. First, one had to change the import of the model. Second, we needed to change the model construction and the default hyperparameters
were already set. The public methods provided by ``DeepWalk`` and ``Walklets`` are the same. An embedding is learned with ``fit`` and it is returned by
``get_embedding``. This allows for quick and minimal changes to the code when a model performs poorly.


Community detection
-------------------

The first machine learning task that we will do is the clustering of pages on Facebook. In this network
nodes represent official verified Facebook pages and the links between them are mutual likes. The pages
have categories and we will look how well the cluster and group memberships are aligned. For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the Facebook page-page network dataset. We will use the page-page graph and the 
page category vector. These are returned as a ``NetworkX`` graph and ``numpy`` array respectively.

.. code-block:: python

    from karateclub import GraphReader

    reader = GraphReader("facebook")

    graph = reader.get_graph()
    target = reader.get_target()

The constructor defines the graph reader object while the methods ``get_graph`` and ``get_target`` read the data.

Now let's use the ``Label Propagation`` community detection method from `Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks <https://arxiv.org/abs/0709.2938>`_. 

.. code-block:: python

    from karateclub import LabelPropagation
    
    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()

The constructor defines a model, we fit the model on the Facebook graph with the ``fit`` method and return the cluster memberships
with the ``get_memberships`` method as a dictionary.


Finally we can evaluate the clustering using normalized mutual information. First we need to create an ordered list of the node memberships.
We use the ground truth about the cluster memberships for calculating the NMI.


.. code-block:: python

    from sklearn.metrics.cluster import normalized_mutual_info_score

    cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]

    nmi = normalized_mutual_info_score(target, cluster_membership)
    print('NMI: {:.4f}'.format(nmi))
    >>> NMI: 0.34374

It is worth noting that the clustering methods in Karate Club work on arbitrary ``NetworkX`` graphs that follow the 
dataset formatting requirements. One could simply cluster a randomly generated Watts-Strogatz graph just like this.

.. code-block:: python

    import networkx as nx
    from karateclub import LabelPropagation
    
    graph = nx.newman_watts_strogatz_graph(100, 20, 0.05)

    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()  


Node embedding
--------------

The second machine learning task that we look at is the identification of users from the UK who abuse the platform on Twitch. 
In the social network of interest nodes represent users and the links are mutual friendships between the users. Our goal is
to perform binary classification of the users (platform abusers and general good guy users).  For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the Twitch UK dataset. We will use the user friendship graph and the 
abusive user target vector. These are returned as a ``NetworkX`` graph and ``numpy`` array respectively.

.. code-block:: python

    from karateclub.dataset import GraphReader

    reader = GraphReader("twitch")

    graph = reader.get_graph()
    y = reader.get_target()

We fit a `Diff2vec node embedding <https://arxiv.org/abs/2001.07463>`_, with a low number of dimensions, diffusions per source node, and short Euler walks.
First, we use the model constructor with custom parameters. Second, we fit the model to the graph. Third, we get the node embedding
which is a ``numpy`` array.

.. code-block:: python

    from karateclub import Diff2Vec

    model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
    model.fit(graph)
    X = model.get_embedding()

We use the node embedding features as predictors of the abusive behaviour. So let us create a train-test split of the explanatory variables
and the target variable with Scikit-Learn. We will use a test data ratio of 20%. Here it is.

.. code-block:: python

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Using the training data (``X_train`` and ``y_train``) we learn a logistic regression model to predict the probability of someone being an abusive user. We perform inference on the test 
set for this target. Finally, we evaluate the model performance by printing an area under the ROC curve value.

.. code-block:: python

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    
    downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_hat = downstream_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_hat)
    print('AUC: {:.4f}'.format(auc))
    >>> AUC: 0.6069

Graph embedding
--------------

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


Benchmark datasets
------------------

We included a number of datasets which can be used for comparing the performance of temporal graph neural networks algorithms. In case of discrete graph neural networks these are as follows:

- `European Deezer user network. <https://arxiv.org/abs/2005.07959>`_
 
