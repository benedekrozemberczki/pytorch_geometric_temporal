"""Tests for batch behaviour."""

import numpy as np
import networkx as nx

from torch_geometric_temporal.signal import temporal_signal_split

from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicGraphStaticSignalBatch


def get_edge_array(node_count, node_start):
    edges = []
    for edge in nx.gnp_random_graph(node_count, 0.1).edges():
        edges.append([edge[0] + node_start, edge[1] + node_start])
    return np.array(edges)


def generate_signal(snapshot_count, node_count, feature_count, graph_count):
    edge_indices = []
    edge_weights = []
    features = []
    targets = []
    batches = []
    for _ in range(snapshot_count):
        node_start = 0
        edge_indices_s = []
        edge_weights_s = []
        features_s = []
        targets_s = []
        batches_s = []
        for i in range(graph_count):
            edge_indices_s.append(get_edge_array(node_count, node_start))
            edge_weights_s.append((np.ones(edge_indices_s[-1].shape[0])))
            features_s.append(np.random.uniform(0, 1, (node_count, feature_count)))
            targets_s.append(
                np.array([np.random.choice([0, 1]) for _ in range(node_count)])
            )
            batches_s.append(np.array([i for _ in range(node_count)]))
            node_start = node_start + node_count
        edge_indices.append(np.concatenate(edge_indices_s).T)
        edge_weights.append(np.concatenate(edge_weights_s))
        features.append(np.concatenate(features_s))
        targets.append(np.concatenate(targets_s))
        batches.append(np.concatenate(batches_s))
    return edge_indices, edge_weights, features, targets, batches


def test_dynamic_graph_temporal_signal_real_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_indices, edge_weights, features, targets, batches = generate_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = DynamicGraphTemporalSignalBatch(
        edge_indices, edge_weights, features, targets, batches
    )

    for _ in range(15):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (graph_count * node_count, feature_count)
            assert snapshot.y.shape == (graph_count * node_count,)
            assert snapshot.batch.shape == (graph_count * node_count,)


def test_static_graph_temporal_signal_batch():
    dataset = StaticGraphTemporalSignalBatch(
        None, None, [None, None], [None, None], None
    )
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None
        assert snapshot.batch is None


def test_dynamic_graph_temporal_signal_batch():
    dataset = DynamicGraphTemporalSignalBatch(
        [None, None], [None, None], [None, None], [None, None], [None, None]
    )
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None
        assert snapshot.batch is None


def test_static_graph_temporal_signal_typing_batch():
    dataset = StaticGraphTemporalSignalBatch(
        None, None, [np.array([1])], [np.array([2])], None
    )
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x.shape == (1,)
        assert snapshot.y.shape == (1,)
        assert snapshot.batch is None


def test_dynamic_graph_static_signal_typing_batch():
    dataset = DynamicGraphStaticSignalBatch([None], [None], None, [None], [None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None
        assert snapshot.batch is None


def test_dynamic_graph_temporal_signal_batch_additional_attrs():
    dataset = DynamicGraphTemporalSignalBatch([None], [None], [None], [None], [None],
                                             optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_static_graph_temporal_signal_batch_additional_attrs():
    dataset = StaticGraphTemporalSignalBatch(None, None, [None], [None], None,
                                             optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_dynamic_graph_static_signal_batch_additional_attrs():
    dataset = DynamicGraphStaticSignalBatch([None], [None], None, [None], [None],
                                             optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_discrete_train_test_split_dynamic_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_indices, edge_weights, features, targets, batches = generate_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = DynamicGraphTemporalSignalBatch(
        edge_indices, edge_weights, features, targets, batches
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)

    for _ in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)


def test_train_test_split_static_graph_temporal_signal_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_indices, edge_weights, features, targets, batches = generate_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = StaticGraphTemporalSignalBatch(
        edge_indices[0], edge_weights[0], features, targets, batches[0]
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)

    for _ in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)


def test_train_test_split_dynamic_graph_static_signal_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_indices, edge_weights, features, targets, batches = generate_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    feature = features[0]

    dataset = DynamicGraphStaticSignalBatch(
        edge_indices, edge_weights, feature, targets, batches
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)

    for _ in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (node_count * graph_count, feature_count)
            assert snapshot.y.shape == (node_count * graph_count,)
