Introduction by example
=======================

*Little Ball of Fur* is a graph sampling extension library for `NetworkX <https://networkx.github.io/>`_.

*Little Ball of Fur* consists of methods which sample from graphs. To put it simply it is a Swiss Army knife for graph sampling tasks. First, it includes a large variety of vertex, edge, and exploration sampling techniques. Second, it provides a unified application public interface which makes the application of sampling algorithms trivial for end-users. Implemented methods cover a wide range of networking (`Networking <https://link.springer.com/conference/networking>`_, `INFOCOM <https://infocom2020.ieee-infocom.org/>`_, `SIGCOMM  <http://www.sigcomm.org/>`_) and data mining (`KDD <https://www.kdd.org/kdd2020/>`_, `TKDD <https://dl.acm.org/journal/tkdd>`_, `ICDE <http://www.wikicfp.com/cfp/program?id=1331&s=ICDE&f=International%20Conference%20on%20Data%20Engineering>`_) conferences, workshops, and pieces from prominent journals.

--------------------------------------------------------------------------------

**Citing**

If you find *Little Ball of Fur* useful in your research, please consider citing the following paper:

.. code-block:: latex

    >@misc{rozemberczki2020little,
        title={Little Ball of Fur: A Python Library for Graph Sampling},
        author={Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
        year={2020},
        eprint={2006.04311},
        archivePrefix={arXiv},
        primaryClass={cs.SI}
    }

Overview
=======================
--------------------------------------------------------------------------------

We shortly overview the fundamental concepts and features of Little Ball of Fur through simple examples. These are the following:

.. contents::
    :local:

Standardized dataset ingestion
------------------------------

Little Ball of Fur assumes that the ``NetworkX`` graph provided by the user has the following important properties:

- The graph is undirected.
- The graph is connected (it consists of a single strongly connected component).
- Nodes are indexed with integers.
- There are no orphaned nodes in the graph.
- The node indexing starts with zero and the indices are consecutive.

The returned ``NetworkX`` graph uses the same indexing.

API driven design
-----------------

Little Ball of Fur uses the design principles of Scikit-Learn which means that the algorithms in the package share the same API. Each graph sampling procedure is implemented as a class which inherits from ``Sampler``. The constructors of the sampling algorithms are used to set the hyperparameters. The sampling procedures have default hyperparameters that work well out of the box. This means that non expert users do not have to make decisions about these in advance and only a little fine tuning is required. For each class the ``sample`` public method provides sampling from the graph. This API driven design in practice means that one can sample a subgraph from a Watts-Strogatz graph with a ``RandomWalkSampler`` just like this.

.. code-block:: python

    import networkx as nx
    from littleballoffur import RandomWalkSampler
    
    graph = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

    model = RandomWalkSampler()
    new_graph = model.sample(graph)

This snippet can be modified to use a ``ForestFireSampler`` with minimal effort like this.

.. code-block:: python

    import networkx as nx
    from littleballoffur import ForestFireSampler
    
    graph = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

    model = ForestFireSampler()
    new_graph = model.sample(graph)

Looking at these two snippets the advantage of the API driven design is evident. First, one had to change the import of the sampler. Second, we needed to change the sampler construction and the default hyperparameters
were already set. The public methods provided by ``RandomWalkSampler`` and ``ForestFireSampler`` are the same. A subsample is returned by
``sample``. This allows for quick and minimal changes to the code when a sampling procedure performs poorly and has to be replaced.


Node sampling
-------------------

The first task that we will look at is sampling a subgraph by drawing a representative set of nodes from a Facebook graph. In this network
nodes represent official verified Facebook pages and the links between them are mutual likes. For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the Facebook page-page network dataset which is returned as a ``NetworkX`` graph.

.. code-block:: python

    from littleballoffur import GraphReader

    reader = GraphReader("facebook")

    graph = reader.get_graph()

The constructor defines the parametrized graph reader object while the ``get_graph`` method reads the data.

Now let's use the ``PageRank Proportional Node Sampling`` method from `Sampling From Large Graphs <https://cs.stanford.edu/people/jure/pubs/sampling-kdd06.pdf>`_. We will sample approximately 50% of the original nodes from the network.

.. code-block:: python

    from littleballoffur import PageRankBasedSampler
    
    number_of_nodes = int(0.5*graph.number_of_nodes())
    sampler = PageRankBasedSampler(number_of_nodes = number_of_nodes)
    new_graph = sampler.sample(graph)

The constructor defines a graph sampler, we sample nodes from the Facebook graph with the ``sample`` method and return the induced subgraph. Finally, we can evaluate the sample quality by comparing clustering coefficient values calculated for the original and subsampled graphs. We somewhat overestimated the transitivity.

.. code-block:: python

    import networkx as nx

    transitivity = nx.transitivity(graph)
    transitivity_sampled = nx.transitivity(new_graph)

    print('Transitivity Original: {:.4f}'.format(transitivity))
    print('Transitivity Sampled: {:.4f}'.format(transitivity_sampled))

    >>> Transitivity Original: 0.2323
    >>> Transitivity Sampled: 0.2673

Edge sampling
--------------

The second task that we will look at is sampling a subgraph by drawing a representative set of edges from a Wikipedia graph. In this network
nodes represent Wikipedia pages about Crocodiles and the edges between them are mutual links. For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the Wikipedia dataset which is returned as a ``NetworkX`` graph.

.. code-block:: python

    from littleballoffur import GraphReader

    reader = GraphReader("wikipedia")

    graph = reader.get_graph()

The constructor defines the parametrized graph reader object while the ``get_graph`` method reads the dataset.

Now let's use the ``Hybrid Node-Edge Sampling`` method from `Reducing Large Internet Topologies for Faster Simulations <http://www.cs.ucr.edu/~michalis/PAPERS/sampling-networking-05.pdf>`_. We will sample approximately 50% of the original edges from the network.

.. code-block:: python

    from littleballoffur import HybridNodeEdgeSampler
    
    number_of_edges = int(0.5*graph.number_of_edges())
    sampler = HybridNodeEdgeSampler(number_of_edges = number_of_edges)
    new_graph = sampler.sample(graph)

The constructor defines a graph sampler, we sample edges from the Wikipedia graph with the ``sample`` method and return the induced subgraph. Finally, we can evaluate the sample quality by comparing clustering coefficient values calculated for the original and subsampled graphs. We massively underestimated the transitivity.

.. code-block:: python

    import networkx as nx

    transitivity = nx.transitivity(graph)
    transitivity_sampled = nx.transitivity(new_graph)

    print('Transitivity Original: {:.4f}'.format(transitivity))
    print('Transitivity Sampled: {:.4f}'.format(transitivity_sampled))

    >>> Transitivity Original: 0.0261
    >>> Transitivity Sampled: 0.0070

Exploration sampling
--------------------

The third task that we will look at is extracting a subgraph with exploration sampling from a GitHub social network. In this graph
nodes represent GitHub developers and the edges between them are mutual follower relationships. For details
about the dataset `see this paper <https://arxiv.org/abs/1909.13021>`_.

We first need to load the GitHub dataset which is returned as a ``NetworkX`` graph.

.. code-block:: python

    from littleballoffur import GraphReader

    reader = GraphReader("github")

    graph = reader.get_graph()

The constructor defines the parametrized graph reader object again, while the ``get_graph`` method reads the dataset.

Now let's use the ``Metropolis-Hastings Random Walk Sampler`` method from `Metropolis Algorithms for Representative Subgraph Sampling <http://mlcb.is.tuebingen.mpg.de/Veroeffentlichungen/papers/HueBorKriGha08.pdf>`_. We will sample approximately 50% of the original nodes from the network.

.. code-block:: python

    from littleballoffur import MetropolisHastingsRandomWalkSampler
    
    number_of_nodes = int(0.5*graph.number_of_nodes())
    sampler = MetropolisHastingsRandomWalkSampler(number_of_nodes = number_of_nodes)
    new_graph = sampler.sample(graph)

The constructor defines a graph sampler, we sample from the Github graph with the ``sample`` method and return the new graph. Finally, we can evaluate the sampling by comparing clustering coefficient values calculated from the original and subsampled graphs.

.. code-block:: python

    import networkx as nx

    transitivity = nx.transitivity(graph)
    transitivity_sampled = nx.transitivity(new_graph)

    print('Transitivity Original: {:.4f}'.format(transitivity))
    print('Transitivity Sampled: {:.4f}'.format(transitivity_sampled))

    >>> Transitivity Original: 0.0124
    >>> Transitivity Sampled: 0.0228

Benchmark datasets
------------------

We included a number of datasets which can be used for comparing the performance of sampling algorithms. These are the following:

- `Twitch user network from the UK. <https://arxiv.org/abs/1909.13021>`_
- `Wikipedia page-page network with articles about Crocodiles. <https://arxiv.org/abs/1909.13021>`_
- `GitHub machine learning and web developers social network. <https://arxiv.org/abs/1909.13021>`_
- `Facebook verified page-page network. <https://arxiv.org/abs/1909.13021>`_
- `Deezer Hungarian user network. <https://arxiv.org/abs/1802.03997>`_
- `LastFM Asian user network. <https://arxiv.org/abs/2005.07959>`_
