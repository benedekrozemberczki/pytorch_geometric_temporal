:github_url: https://github.com/benedekrozemberczki/pytorch_geometric_temporal

PyTorch Geometric Temporal Documentation
========================================

PyTorch Geometric Temporal is a temporal graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_.  It builds on open-source deep-learning and graph processing libraries. *PyTorch Geometric Temporal* consists of state-of-the-art deep learning and parametric learning methods to process spatio-temporal signals. It is the first open-source library for temporal deep learning on geometric structures and provides constant time difference graph neural networks on dynamic and static graphs. We make this happen with the use of discrete time graph snapshots. Implemented methods cover a wide range of data mining (`WWW <https://www2021.thewebconf.org/>`_, `KDD <https://www.kdd.org/kdd2020/>`_), artificial intelligence and machine learning (`AAAI <http://www.aaai.org/Conferences/conferences.php>`_, `ICONIP <https://www.apnns.org/ICONIP2020/>`_, `ICLR <https://iclr.cc/>`_) conferences, workshops, and pieces from prominent journals. 



PyTorch Geometric Temporal includes support for index-batching - a new batching technique that improves spatiotemporal memory efficiency without any impact on accuracy. Additionally, PyTorch Geometric Temporal supports memory-efficient distributed data parallel training using Dask-DDP in combination with index-batching.

.. The package interfaces well with `Pytorch Lightning <https://pytorch-lightning.readthedocs.io>`_ which allows training on CPUs, single and multiple GPUs out-of-the-box. Take a look at this introductory example of using PyTorch Geometric Temporal with Pytorch Lighning.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/installation
   notes/introduction
   notes/resources

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/root
   modules/signal
   modules/dataset
