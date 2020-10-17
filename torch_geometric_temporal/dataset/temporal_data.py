import re
import logging

from torch_sparse import coalesce, SparseTensor
import torch_geometric
from torch_geometric.utils import (contains_isolated_nodes,
                                   contains_self_loops, is_undirected)

from torch_geometric.utils.num_nodes import maybe_num_nodes

# __num_nodes_warn_msg__ = (
#     'The number of nodes in your data object can only be inferred by its {} '
#     'indices, and hence may result in unexpected batch-wise behavior, e.g., '
#     'in case there exists isolated nodes. Please consider explicitly setting '
#     'the number of nodes for this data object by assigning it to '
#     'data.num_nodes.')

from torch.utils.data import Dataset
from torch_geometric.data import Data


class TemporalData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, temporal_dim=0, **kwargs):
        """ A TemporalData instance """
        # TODO documentation

        # TODO temporal_dim should be a property...
        assert temporal_dim in [0, 1], "Specified temporal dimension must be in [0, 1], but got {}".format(temporal_dim)
        self.temporal_dim = temporal_dim

        super().__init__(x, edge_index, edge_attr, y,
                         pos, norm, face, **kwargs)

    @property
    def temporal_node_attrs(self):
        return self.x is not None and self.x.dim() == 3

    @property
    def temporal_edge_index(self):
        return self.edge_index is not None and self.edge_index.dim() == 3

    @property
    def temporal_edge_attr(self):
        # handle the case where you have a single and/or multiple edge attributes/weights
        return self.edge_attr is not None and self.edge_attr.dim() in [2, 3]

    @property
    def temporal_length(self):
        """ returns the length of the temporal dimension"""
        return self.x.size(self.temporal_dim)

    def to_data_list(self):
        data_dict = {}
        for key, value in self.__dict__.items():
            if key is not None and key != "temporal_dim" and value is not None:
                data_dict[key] = self[key].unbind(self.temporal_dim)

        data_list = []
        for t in range(self.temporal_length):
            data_dict_list = {}
            for key in data_dict:
                if key != "temporal_dim" and key is not None:
                    data_dict_list.update({key: data_dict[key][t]})
            data_list.append(Data.from_dict(data_dict_list))

        return data_list

    @classmethod
    def from_data_list(cls, data_list):
        """ data_list is a list of pytorch geometric Data objects.
        Restrict to all Data objects in data_list contain the same attributes!"""
        temporal_data = cls()

        from collections import defaultdict

        tmp = defaultdict(list)
        for data in data_list:
            for key in data.__dict__:
                if data[key] is not None:
                    tmp[key].append(data[key])

        for key in tmp:
            temporal_data.__setattr__(key, torch.stack(tmp[key]))

        if torch_geometric.is_debug_enabled():
            temporal_data.debug()

        return temporal_data

    @property
    def __temporal_dim__(self):
        return self.temporal_dim

    # TODO check that this fn works for creating batches
    def __cat_dim__(self, key, value):
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Concatenate `*index*` and `*face*` attributes in the last dimension.
        if bool(re.search('(index|face)', key)):
            return -1
        # By default, concatenate sparse matrices diagonally.
        elif isinstance(value, SparseTensor):
            return (0, 1)
        elif self.__temporal_dim__ == 0:
            return 1
        return 0

    # TODO check that this fn works for creating batches
    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        return self.num_nodes if bool(re.search('(index|face)', key)) else 0

    @property
    def num_nodes(self):
        r"""Returns or sets the number of nodes in the graph.

        .. note::
            The number of nodes in your data object is typically automatically
            inferred, *e.g.*, when node features :obj:`x` are present.
            In some cases however, a graph may only be given by its edge
            indices :obj:`edge_index`.
            PyTorch Geometric then *guesses* the number of nodes
            according to :obj:`edge_index.max().item() + 1`, but in case there
            exists isolated nodes, this number has not to be correct and can
            therefore result in unexpected batch-wise behavior.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'pos', 'norm', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        if hasattr(self, 'adj'):
            return self.adj.size(0)
        if hasattr(self, 'adj_t'):
            return self.adj_t.size(1)
        if self.face is not None:
            logging.warning(__num_nodes_warn_msg__.format('face'))
            return maybe_num_nodes(self.face)
        if self.edge_index is not None:
            logging.warning(__num_nodes_warn_msg__.format('edge'))
            return maybe_num_nodes(self.edge_index)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_edges(self):
        r"""Returns the number of edges in the graph."""
        for key, item in self('edge_index', 'edge_attr'):
            return item.size(self.__cat_dim__(key, item))
        return None

    @property
    def num_faces(self):
        r"""Returns the number of faces in the mesh."""
        if self.face is not None:
            return self.face.size(self.__cat_dim__('face', self.face))
        return None

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    def is_coalesced(self):
        r"""Returns :obj:`True`, if edge indices are ordered and do not contain
        duplicate entries."""
        edge_index, _ = coalesce(self.edge_index, None, self.num_nodes,
                                 self.num_nodes)
        return self.edge_index.numel() == edge_index.numel() and (
                self.edge_index != edge_index).sum().item() == 0

    def coalesce(self):
        r""""Orders and removes duplicated entries from edge indices."""
        self.edge_index, self.edge_attr = coalesce(self.edge_index,
                                                   self.edge_attr,
                                                   self.num_nodes,
                                                   self.num_nodes)
        return self

    def temporal_coalesce(self):
        r""""Orders and removes duplicated entries from edge indices.
        Supports temporality."""
        assert self.edge_index is not None, "edge_index found to be None, cannot coalesce"
        # TODO handle the case where edge_attr is None but edge_index isn't
        coalesced_ei, coalesced_eattr = [], []
        for i in zip(self.edge_index, self.edge_attr):
            ei, eattr = coalesce(self.edge_index[i],
                                 self.edge_attr[i],
                                 self.num_nodes,
                                 self.num_nodes)
            coalesced_ei.append(ei)
            coalesced_eattr.append(eattr)
        self.edge_index = torch.stack(coalesced_ei)
        self.edge_attr = torch.stack(coalesced_eattr)
        return self

    def contains_isolated_nodes(self):
        r"""Returns :obj:`True`, if the graph contains isolated nodes."""
        return contains_isolated_nodes(self.edge_index, self.num_nodes)

    def contains_self_loops(self):
        """Returns :obj:`True`, if the graph contains self-loops."""
        return contains_self_loops(self.edge_index)

    def is_undirected(self):
        r"""Returns :obj:`True`, if graph edges are undirected."""
        return is_undirected(self.edge_index, self.edge_attr, self.num_nodes)

    def is_directed(self):
        r"""Returns :obj:`True`, if graph edges are directed."""
        return not self.is_undirected()

    def debug(self):
        if self.edge_index is not None:
            if self.edge_index.dtype != torch.long:
                raise RuntimeError(
                    ('Expected edge indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.edge_index.dtype))

        if self.face is not None:
            if self.face.dtype != torch.long:
                raise RuntimeError(
                    ('Expected face indices of dtype {}, but found dtype '
                     ' {}').format(torch.long, self.face.dtype))

        if self.edge_index is not None:
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise RuntimeError(
                    ('Edge indices should have shape [2, num_edges] but found'
                     ' shape {}').format(self.edge_index.size()))

        if self.edge_index is not None and self.num_nodes is not None:
            if self.edge_index.numel() > 0:
                min_index = self.edge_index.min()
                max_index = self.edge_index.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Edge indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                        self.num_nodes - 1, min_index, max_index))

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    ('Face indices should have shape [3, num_faces] but found'
                     ' shape {}').format(self.face.size()))

        if self.face is not None and self.num_nodes is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    ('Face indices must lay in the interval [0, {}]'
                     ' but found them in the interval [{}, {}]').format(
                        self.num_nodes - 1, min_index, max_index))

        if self.edge_index is not None and self.edge_attr is not None:
            if self.edge_index.size(1) != self.edge_attr.size(0):
                raise RuntimeError(
                    ('Edge indices and edge attributes hold a differing '
                     'number of edges, found {} and {}').format(
                        self.edge_index.size(), self.edge_attr.size()))

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node features should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.x.size(0)))

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node positions should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.pos.size(0)))

        if self.norm is not None and self.num_nodes is not None:
            if self.norm.size(0) != self.num_nodes:
                raise RuntimeError(
                    ('Node normals should hold {} elements in the first '
                     'dimension but found {}').format(self.num_nodes,
                                                      self.norm.size(0)))


class TemporalDataset(Dataset):
    def __init__(self, data_list, horizon, laplacian_normalization):
        super().__init__()

        if isinstance(data_list, dict):
            self.data = list(data_list.values())
        elif isinstance(data_list, (list, tuple)):
            self.data = data_list
        else:
            self.data = data_list

        self.horizon = horizon
        self.normalization = laplacian_normalization

    def __len__(self):
        return len(self.data) - self.horizon

    def __getitem__(self, idx):
        data_list = self.data[idx:idx + self.horizon]
        # if self.normalization is not None:
        #     data_list = [LaplacianLambdaMax(normalization=self.normalization, is_undirected=d.is_undirected())(d) for d in
        #                  data_list]
        tgt = self.data[idx + self.horizon].y
        return data_list, tgt


if __name__ == "__main__":
    import numpy as np
    import torch

    temporal_data = TemporalData(x=torch.empty(size=(10, 100, 1000)).random_(),
                                 y=torch.empty(size=(100,)).random_()
                                 )

    data_list_test = [Data(x=torch.empty(size=(10, 100))) for x in range(10)]
    temporal_data_from_data_list = TemporalData.from_data_list(data_list_test)
    temporal_data_from_data_list.to_data_list()
    print("done")
