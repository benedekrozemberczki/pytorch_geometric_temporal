"""Tests for batch behaviour."""

import numpy as np
import networkx as nx

from torch_geometric_temporal.signal import temporal_signal_split

from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicGraphStaticSignalBatch

from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignalBatch
from torch_geometric_temporal.signal import DynamicHeteroGraphStaticSignalBatch


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


def generate_heterogeneous_signal(snapshot_count, node_count, feature_count, graph_count):
    edge_index_dicts = []
    edge_weight_dicts = []
    feature_dicts = []
    target_dicts = []
    batch_dicts = []
    for _ in range(snapshot_count):
        node_start = 0
        edge_index_dict_s = {('author', 'writes', 'paper'): []}
        edge_weight_dict_s = {('author', 'writes', 'paper'): []}
        feature_dict_s = {'author': [],
                          'paper': []}
        target_dict_s = {'author': [],
                         'paper': []}
        batch_dict_s = {'author': [],
                        'paper': []}
        for i in range(graph_count):
            edge_index_dict_s[('author', 'writes', 'paper')].append(get_edge_array(node_count, node_start))
            edge_weight_dict_s[('author', 'writes', 'paper')].append(
                (np.ones(edge_index_dict_s[('author', 'writes', 'paper')][-1].shape[0]))
            )
            feature_dict_s['paper'].append(np.random.uniform(0, 1, (node_count, feature_count)))
            feature_dict_s['author'].append(np.random.uniform(0, 1, (node_count, feature_count)))
            target_dict_s['paper'].append(
                np.array([np.random.choice([0, 1]) for _ in range(node_count)])
            )
            target_dict_s['author'].append(
                np.array([np.random.choice([0, 1]) for _ in range(node_count)])
            )
            batch_dict_s['paper'].append(np.array([i for _ in range(node_count)]))
            batch_dict_s['author'].append(np.array([i for _ in range(node_count)]))
            node_start = node_start + node_count
        edge_index_dicts.append(
            {node_type: np.concatenate(edge_indices_s).T for node_type, edge_indices_s in edge_index_dict_s.items()}
        )
        edge_weight_dicts.append(
            {node_type: np.concatenate(edge_weights_s) for node_type, edge_weights_s in edge_weight_dict_s.items()}
        )
        feature_dicts.append(
            {node_type: np.concatenate(features_s) for node_type, features_s in feature_dict_s.items()}
        )
        target_dicts.append({node_type: np.concatenate(targets_s) for node_type, targets_s in target_dict_s.items()})
        batch_dicts.append({node_type: np.concatenate(batches_s) for node_type, batches_s in batch_dict_s.items()})
    return edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, batch_dicts


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


def test_static_hetero_graph_temporal_signal_batch():
    dataset = StaticHeteroGraphTemporalSignalBatch(
        None, None, [None, None], [None, None], None
    )
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


def test_dynamic_hetero_graph_static_signal_batch():
    dataset = DynamicHeteroGraphStaticSignalBatch(
        [None], [None], None, [None], [None]
    )
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


def test_dynamic_hetero_graph_temporal_signal_batch():
    dataset = DynamicHeteroGraphTemporalSignalBatch(
        [None, None], [None, None], [None, None], [None, None], [None, None]
    )
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


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


def test_static_hetero_graph_temporal_signal_typing_batch():
    dataset = StaticHeteroGraphTemporalSignalBatch(
        None, None, [{'author': np.array([1])}], [{'author': np.array([2])}], None
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert 'batch' not in list(dict(snapshot.node_stores[0]).keys())
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_static_signal_typing_batch():
    dataset = DynamicHeteroGraphStaticSignalBatch(
        [None], [None], {'author': np.array([1])}, [{'author': np.array([2])}], [None]
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert 'batch' not in list(dict(snapshot.node_stores[0]).keys())
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_temporal_signal_typing_batch():
    dataset = DynamicHeteroGraphTemporalSignalBatch(
        [None], [None], [{'author': np.array([1])}], [{'author': np.array([2])}], [None]
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert 'batch' not in list(dict(snapshot.node_stores[0]).keys())
        assert len(snapshot.edge_types) == 0


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


def test_static_hetero_graph_temporal_signal_batch_additional_attrs():
    dataset = StaticHeteroGraphTemporalSignalBatch(None, None, [None], [None], None,
                                                   optional1=[{'author': np.array([1])}],
                                                   optional2=[{'author': np.array([2])}],
                                                   optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_dynamic_hetero_graph_static_signal_batch_additional_attrs():
    dataset = DynamicHeteroGraphStaticSignalBatch([None], [None], None, [None], [None],
                                                  optional1=[{'author': np.array([1])}],
                                                  optional2=[{'author': np.array([2])}],
                                                  optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_dynamic_hetero_graph_temporal_signal_batch_additional_attrs():
    dataset = DynamicHeteroGraphTemporalSignalBatch([None], [None], [None], [None], [None],
                                                    optional1=[{'author': np.array([1])}],
                                                    optional2=[{'author': np.array([2])}],
                                                    optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_dynamic_graph_static_signal_batch_additional_attrs():
    dataset = DynamicGraphStaticSignalBatch([None], [None], None, [None], [None],
                                             optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_static_hetero_graph_temporal_signal_batch_edges():
    dataset = StaticHeteroGraphTemporalSignalBatch({("author", "writes", "paper"): np.array([[0, 1], [1, 0]])},
                                                   {("author", "writes", "paper"): np.array([[0.1], [0.1]])},
                                                   [{"author": np.array([[0], [0]]),
                                                     "paper": np.array([[0], [0], [0]])},
                                                    {"author": np.array([[0.1], [0.1]]),
                                                     "paper": np.array([[0.1], [0.1], [0.1]])}],
                                                   [None, None],
                                                   None)
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]


def test_dynamic_hetero_graph_static_signal_batch_edges():
    dataset = DynamicHeteroGraphStaticSignalBatch([{("author", "writes", "paper"): np.array([[0, 1], [1, 0]])}],
                                                  [{("author", "writes", "paper"): np.array([[0.1], [0.1]])}],
                                                  {"author": np.array([[0], [0]]),
                                                   "paper": np.array([[0], [0], [0]])},
                                                  [None],
                                                  [None])
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]


def test_dynamic_hetero_graph_temporal_signal_batch_edges():
    dataset = DynamicHeteroGraphTemporalSignalBatch([{("author", "writes", "paper"): np.array([[0, 1], [1, 0]])}],
                                                    [{("author", "writes", "paper"): np.array([[0.1], [0.1]])}],
                                                    [{"author": np.array([[0], [0]]),
                                                      "paper": np.array([[0], [0], [0]])}],
                                                    [None],
                                                    [None])
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]


def test_static_hetero_graph_temporal_signal_batch_assigned():
    dataset = StaticHeteroGraphTemporalSignalBatch(
        None, None, [{'author': np.array([1])}], [{'author': np.array([2])}], {'author': np.array([1])}
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert snapshot.node_stores[0]['batch'].shape == (1,)
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_static_signal_batch_assigned():
    dataset = DynamicHeteroGraphStaticSignalBatch(
        [None], [None], {'author': np.array([1])}, [{'author': np.array([2])}], [{'author': np.array([1])}]
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert snapshot.node_stores[0]['batch'].shape == (1,)
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_temporal_signal_batch_assigned():
    dataset = DynamicHeteroGraphTemporalSignalBatch(
        [None], [None], [{'author': np.array([1])}], [{'author': np.array([2])}], [{'author': np.array([1])}]
    )
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert snapshot.node_stores[0]['batch'].shape == (1,)
        assert len(snapshot.edge_types) == 0


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


def test_train_test_split_dynamic_hetero_graph_temporal_signal_batch():
    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, batch_dicts = generate_heterogeneous_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = DynamicHeteroGraphTemporalSignalBatch(
        edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, batch_dicts
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]

    for _ in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]


def test_train_test_split_static_hetero_graph_temporal_signal_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, batch_dicts = generate_heterogeneous_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = StaticHeteroGraphTemporalSignalBatch(
        edge_index_dicts[0], edge_weight_dicts[0], feature_dicts, target_dicts, batch_dicts[0]
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]

    for _ in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]


def test_train_test_split_dynamic_hetero_graph_static_signal_batch():

    snapshot_count = 250
    node_count = 100
    feature_count = 32
    graph_count = 10

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, batch_dicts = generate_heterogeneous_signal(
        snapshot_count, node_count, feature_count, graph_count
    )

    dataset = DynamicHeteroGraphStaticSignalBatch(
        edge_index_dicts, edge_weight_dicts, feature_dicts[0], target_dicts, batch_dicts
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for _ in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]

    for _ in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (node_count * graph_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (node_count * graph_count,)
            assert snapshot.node_stores[1]['y'].shape == (node_count * graph_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
