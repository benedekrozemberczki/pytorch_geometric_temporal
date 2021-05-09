[pypi-image]: https://badge.fury.io/py/torch-geometric-temporal.svg
[pypi-url]: https://pypi.python.org/pypi/torch-geometric-temporal
[size-image]: https://img.shields.io/github/repo-size/benedekrozemberczki/pytorch_geometric_temporal.svg
[size-url]: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/archive/master.zip
[build-image]: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/workflows/CI/badge.svg
[build-url]: https://github.com/benedekrozemberczki/pytorch_geometric_temporal/actions?query=workflow%3ACI
[docs-image]: https://readthedocs.org/projects/pytorch-geometric-temporal/badge/?version=latest
[docs-url]: https://pytorch-geometric-temporal.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/benedekrozemberczki/pytorch_geometric_temporal/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/benedekrozemberczki/pytorch_geometric_temporal?branch=master

<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]
[![Repo size][size-image]][size-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Build Status][build-image]][build-url]
[![Arxiv](https://img.shields.io/badge/ArXiv-2104.07788-orange.svg)](https://arxiv.org/abs/2104.07788)

**[Documentation](https://pytorch-geometric-temporal.readthedocs.io)** | **[External Resources](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/resources.html)** | **[Datasets](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#discrete-time-datasets)**

*PyTorch Geometric Temporal* is a temporal (dynamic) extension library for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

<p align="justify">The library consists of various dynamic and temporal geometric deep learning, embedding, and spatio-temporal regression methods from a variety of published research papers. Moreover, it comes with an easy-to-use dataset loader, train-test splitter and temporal snaphot iterator for dynamic and temporal graphs. The framework naturally provides GPU support. It also comes with a number of benchmark datasets from the epidemological forecasting, sharing economy, energy production and web traffic management domains. Finally, you can also create your own datasets.</p>

The package interfaces well with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io) which allows training on CPUs, single and multiple GPUs out-of-the-box. Take a look at this [introductory example](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/examples/lightning_example.py) of using PyTorch Geometric Temporal with Pytorch Lighning.

--------------------------------------------------------------------------------

**Citing**


If you find *PyTorch Geometric Temporal* and the new datasets useful in your research, please consider adding the following citation:

```bibtex
@misc{rozemberczki2021pytorch,
      author = {Benedek Rozemberczki and Paul Scherer and Yixuan He and George Panagopoulos and Maria Astefanoaei and Oliver Kiss and Ferenc Beres and Nicolas Collignon and Rik Sarkar},
      title = {{PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine Learning Models}},
      year = {2021},
      eprint = {arXiv:2104.07788},
}
```

--------------------------------------------------------------------------------

**A simple example**

PyTorch Geometric Temporal makes implementing Dynamic and Temporal Graph Neural Networks quite easy - see the accompanying [tutorial](https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#applications). For example, this is all it takes to implement a recurrent graph convolutional network with two consecutive [graph convolutional GRU](https://arxiv.org/abs/1612.07659) cells and a linear layer:

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):

    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
```
--------------------------------------------------------------------------------

**Methods Included**

In detail, the following temporal graph neural networks were implemented.


**Recurrent Graph Convolutions**

* **[DCRNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dcrnn.DCRNN)** from Li *et al.*: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) (ICLR 2018)

* **[GConvGRU](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_gru.GConvGRU)** from Seo *et al.*: [Structured Sequence Modeling with Graph  Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659) (ICONIP 2018)

* **[GConvLSTM](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gconv_lstm.GConvLSTM)** from Seo *et al.*: [Structured Sequence Modeling with Graph  Convolutional Recurrent Networks](https://arxiv.org/abs/1612.07659) (ICONIP 2018)

* **[GC-LSTM](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.gc_lstm.GCLSTM)** from Chen *et al.*: [GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction](https://arxiv.org/abs/1812.04206) (CoRR 2018)

* **[LRGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.lrgcn.LRGCN)** from Li *et al.*: [Predicting Path Failure In Time-Evolving Graphs](https://arxiv.org/abs/1905.03994) (KDD 2019)

* **[DyGrEncoder](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dygrae.DyGrEncoder)** from Taheri *et al.*: [Learning to Represent the Evolution of Dynamic Graphs with Recurrent Models](https://dl.acm.org/doi/10.1145/3308560.3316581) (WWW 2019)

* **[EvolveGCNH](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.evolvegcnh.EvolveGCNH)** from Pareja *et al.*: [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191) (AAAI 2020)

* **[EvolveGCNO](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.evolvegcno.EvolveGCNO)** from Pareja *et al.*: [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191) (AAAI 2020)

* **[T-GCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.temporalgcn.TGCN)** from Zhao *et al.*: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320) (IEEE TITS 2019)

* **[A3T-GCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.attentiontemporalgcn.A3TGCN)** from Zhu *et al.*: [A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting](https://arxiv.org/abs/2006.11583) (CoRR 2020)

* **[AGCRN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.agcrn.AGCRN)** from Bai *et al.*: [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842) (NeurIPS 2020)

* **[MPNN LSTM](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.mpnn_lstm.MPNNLSTM)** from Panagopoulos *et al.*: [Transfer Graph Neural Networks for Pandemic Forecasting](https://arxiv.org/abs/2009.08388) (AAAI 2021)

**Temporal Graph Convolutions**

* **[STGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.stgcn.STConv)** from Yu *et al.*: [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) (IJCAI 2018)

* **[ASTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.astgcn.ASTGCN)** from Guo *et al.*: [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (AAAI 2019)

* **[MSTGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mstgcn.MSTGCN)** from Guo *et al.*: [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (AAAI 2019)

* **[GMAN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.gman.GMAN)** from Zheng *et al.*: [GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/pdf/1911.08415.pdf) (AAAI 2020)

* **[MTGNN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.mtgnn.MTGNN)** from Wu *et al.*: [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650) (KDD 2020)

**Auxiliary Graph Convolutions**

* **[TemporalConv](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.stgcn.TemporalConv)** from Yu *et al.*: [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) (IJCAI 2018)

* **[DConv](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.dcrnn.DConv)** from Li *et al.*: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) (ICLR 2018)

* **[ChebConvAttention](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.attention.astgcn.ChebConvAttention)** from Guo *et al.*: [Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/3881) (AAAI 2019)

* **[AVWGCN](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#torch_geometric_temporal.nn.recurrent.agcrn.AVWGCN)** from Bai *et al.*: [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/abs/2007.02842) (NeurIPS 2020)

--------------------------------------------------------------------------------


Head over to our [documentation](https://pytorch-geometric-temporal.readthedocs.io) to find out more about installation, creation of datasets and a full list of implemented methods and available datasets.
For a quick start, check out the [examples](https://pytorch-geometric-temporal.readthedocs.io) in the `examples/` directory.

If you notice anything unexpected, please open an [issue](https://benedekrozemberczki/pytorch_geometric_temporal/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues).


--------------------------------------------------------------------------------

**Installation**


**PyTorch 1.8.0**

To install the binaries for PyTorch 1.8.0, simply run

```sh
$ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu101`, `cu102`, or `cu111` depending on your PyTorch installation.

|             | `cpu` | `cu101` | `cu102` | `cu111` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      | ✅      |
| **macOS**   | ✅    |         |         |         |

**PyTorch 1.7.0**

To install the binaries for PyTorch 1.7.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102` or `cu110` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` | `cu110` |
|-------------|-------|--------|---------|---------|---------|
| **Linux**   | ✅    | ✅     | ✅     | ✅      | ✅     |
| **Windows** | ✅    | ❌     | ✅     | ✅      | ✅     |
| **macOS**   | ✅    |        |         |         |         |

--------------------------------------------------------------------------------

**PyTorch 1.6.0**

To install the binaries for PyTorch 1.6.0, simply run

```sh
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
$ pip install torch-geometric
$ pip install torch-geometric-temporal
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.

|             | `cpu` | `cu92` | `cu101` | `cu102` |
|-------------|-------|--------|---------|---------|
| **Linux**   | ✅    | ✅    | ✅     | ✅      |
| **Windows** | ✅    | ❌    | ✅     | ✅      |
| **macOS**   | ✅    |        |         |         |

--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [MIT License](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/LICENSE)
