:github_url: https://github.com/benedekrozemberczki/pytorch_geometric_temporal

PyTorch Geometric Temporal Documentation
==================================

*PyTorch Geometric Temporal* is an temporal graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_.  It builds on open-source deep-learning and graph processing libraries. *PyTorch Geometric Temporal* consists of state-of-the-art deep learning and parametric learning methods to process temporal signals on static and dynamic graphs. It is the first open-source library for temporal deep learning on geometric structures. First, it provides network embedding techniques at the node and graph level. Second, it includes a variety of overlapping and non-overlapping commmunity detection methods. Implemented methods cover a wide range of network science (`NetSci <https://netscisociety.net/home>`_, `Complenet <https://complenet.weebly.com/>`_), data mining (`ICDM <http://icdm2019.bigke.org/>`_, `CIKM <http://www.cikm2019.net/>`_, `KDD <https://www.kdd.org/kdd2020/>`_), artificial intelligence (`AAAI <http://www.aaai.org/Conferences/conferences.php>`_, `IJCAI <https://www.ijcai.org/>`_) and machine learning (`NeurIPS <https://nips.cc/>`_, `ICML <https://icml.cc/>`_, `ICLR <https://iclr.cc/>`_) conferences, workshops, and pieces from prominent journals. 

.. code-block:: latex

    >@misc{rozemberczki_temporal,
           author = {Benedek, Rozemberczki},
           title = {{PyTorch Geometric Temporal}},
           year = {2020},
           publisher = {GitHub},
           journal = {GitHub repository},
           howpublished = {\url{https://github.com/benedekrozemberczki/pytorch_geometric_temporal}},
    }

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/installation
   notes/introduction
   notes/create_dataset
   notes/examples
   notes/resources

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/root
   modules/dataset
