import numpy as np
import networkx as nx

from torch_geometric_temporal.signal import temporal_signal_split

from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicGraphStaticSignal

from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicHeteroGraphStaticSignal

from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.dataset import (
    ChickenpoxDatasetLoader,
    PedalMeDatasetLoader,
    WikiMathsDatasetLoader,
    EnglandCovidDatasetLoader,
)
from torch_geometric_temporal.dataset import (
    TwitterTennisDatasetLoader,
    MontevideoBusDatasetLoader,
    MTMDatasetLoader,
)

from torch_geometric_temporal.dataset import (
    WindmillOutputLargeDatasetLoader,
    WindmillOutputMediumDatasetLoader,
    WindmillOutputSmallDatasetLoader,
)


def get_edge_array(n_count):
    return np.array([edge for edge in nx.gnp_random_graph(n_count, 0.1).edges()]).T


def generate_signal(snapshot_count, n_count, feature_count, additional_features_keys=[]):
    edge_indices = [get_edge_array(n_count) for _ in range(snapshot_count)]
    edge_weights = [np.ones(edge_indices[t].shape[1]) for t in range(snapshot_count)]
    features = [
        np.random.uniform(0, 1, (n_count, feature_count)) for _ in range(snapshot_count)
    ]

    if additional_features_keys:
        additional_features = {
            key: [np.random.uniform(0, 1, (n_count, feature_count)) for _ in range(snapshot_count)
                  ] for key in additional_features_keys}
        return edge_indices, edge_weights, features, additional_features

    return edge_indices, edge_weights, features


def generate_heterogeneous_signal(snapshot_count, n_count, feature_count, *additional_features_keys):
    edge_index_dicts = [{('author', 'writes', 'paper'): get_edge_array(n_count)} for _ in range(snapshot_count)]
    edge_weight_dicts = [{('author', 'writes', 'paper'): np.ones(edge_index_dicts[t][('author', 'writes', 'paper')].shape[1])}
                         for t in range(snapshot_count)]
    feature_dicts = [{'author': np.random.uniform(0, 1, (n_count, feature_count)),
                      'paper': np.random.uniform(0, 1, (n_count, feature_count))} for _ in range(snapshot_count)]

    target_dicts = [{'author': np.random.uniform(0, 10, (n_count,)),
                     'paper': np.random.uniform(0, 10, (n_count,))} for _ in range(snapshot_count)]

    if additional_features_keys:
        additional_features = {
            key: [{'author': np.random.uniform(0, 1, (n_count, feature_count)),
                   'paper': np.random.uniform(0, 1, (n_count, feature_count))} for _ in range(snapshot_count)]
            for key in additional_features_keys}
        return edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, additional_features

    return edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts


def test_dynamic_graph_temporal_signal_real():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features = generate_signal(250, 100, 32)

    targets = [np.random.uniform(0, 10, (n_count,)) for _ in range(snapshot_count)]

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)

    targets = [
        np.floor(np.random.uniform(0, 10, (n_count,))).astype(int)
        for _ in range(snapshot_count)
    ]

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)


def test_static_graph_temporal_signal():
    dataset = StaticGraphTemporalSignal(None, None, [None, None], [None, None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None


def test_dynamic_graph_temporal_signal():
    dataset = DynamicGraphTemporalSignal(
        [None, None], [None, None], [None, None], [None, None]
    )
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None


def test_static_graph_temporal_signal_typing():
    dataset = StaticGraphTemporalSignal(None, None, [np.array([1])], [np.array([2])])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x.shape == (1,)
        assert snapshot.y.shape == (1,)


def test_dynamic_graph_static_signal_typing():
    dataset = DynamicGraphStaticSignal([None], [None], None, [None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None


def test_static_graph_temporal_signal_additional_attrs():
    dataset = StaticGraphTemporalSignal(None, None, [None], [None],
                                        optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_dynamic_graph_static_signal_additional_attrs():
    dataset = DynamicGraphStaticSignal([None], [None], None, [None],
                                       optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_dynamic_graph_temporal_signal_additional_attrs():
    dataset = DynamicGraphTemporalSignal([None], [None], [None], [None],
                                         optional1=[np.array([1])], optional2=[np.array([2])])
    assert dataset.additional_feature_keys == ["optional1", "optional2"]
    for snapshot in dataset:
        assert snapshot.optional1.shape == (1,)
        assert snapshot.optional2.shape == (1,)


def test_static_hetero_graph_temporal_signal():
    dataset = StaticHeteroGraphTemporalSignal(None, None, [None], [None])
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


def test_static_hetero_graph_temporal_signal_typing():
    dataset = StaticHeteroGraphTemporalSignal(None, None, [{'author': np.array([1])}], [{'author': np.array([2])}])
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert len(snapshot.edge_types) == 0


def test_static_hetero_graph_temporal_signal_additional_attrs():
    dataset = StaticHeteroGraphTemporalSignal(None, None, [None], [None],
                                              optional1=[{'author': np.array([1])}],
                                              optional2=[{'author': np.array([2])}],
                                              optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_static_hetero_graph_temporal_signal_edges():
    dataset = StaticHeteroGraphTemporalSignal({("author", "writes", "paper"): np.array([[0, 1], [1, 0]])},
                                              {("author", "writes", "paper"): np.array([[0.1], [0.1]])},
                                              [{"author": np.array([[0], [0]]),
                                                "paper": np.array([[0], [0], [0]])},
                                               {"author": np.array([[0.1], [0.1]]),
                                                "paper": np.array([[0.1], [0.1], [0.1]])}],
                                              [None, None])
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]


def test_dynamic_hetero_graph_static_signal():
    dataset = DynamicHeteroGraphStaticSignal([None], [None], None, [None])
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


def test_dynamic_hetero_graph_static_signal_typing():
    dataset = DynamicHeteroGraphStaticSignal([None], [None], {'author': np.array([1])}, [{'author': np.array([2])}])
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_static_signal_additional_attrs():
    dataset = DynamicHeteroGraphStaticSignal([None], [None], None, [None],
                                             optional1=[{'author': np.array([1])}],
                                             optional2=[{'author': np.array([2])}],
                                             optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_dynamic_hetero_graph_static_signal_edges():
    dataset = DynamicHeteroGraphStaticSignal([{("author", "writes", "paper"): np.array([[0, 1], [1, 0]])}],
                                             [{("author", "writes", "paper"): np.array([[0.1], [0.1]])}],
                                             {"author": np.array([[0], [0]]),
                                              "paper": np.array([[0], [0], [0]])},
                                             [None])
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]


def test_dynamic_hetero_graph_temporal_signal():
    dataset = DynamicHeteroGraphTemporalSignal(
        [None, None], [None, None], [None, None], [None, None]
    )
    for snapshot in dataset:
        assert len(snapshot.node_types) == 0
        assert len(snapshot.node_stores) == 0
        assert len(snapshot.edge_types) == 0
        assert len(snapshot.edge_stores) == 0


def test_dynamic_hetero_graph_temporal_signal_typing():
    dataset = DynamicHeteroGraphTemporalSignal([None], [None], [{'author': np.array([1])}], [{'author': np.array([2])}])
    for snapshot in dataset:
        assert snapshot.node_types[0] == 'author'
        assert snapshot.node_stores[0]['x'].shape == (1,)
        assert snapshot.node_stores[0]['y'].shape == (1,)
        assert len(snapshot.edge_types) == 0


def test_dynamic_hetero_graph_temporal_signal_additional_attrs():
    dataset = DynamicHeteroGraphTemporalSignal([None], [None], [None], [None],
                                               optional1=[{'author': np.array([1])}],
                                               optional2=[{'author': np.array([2])}],
                                               optional3=[None])
    assert dataset.additional_feature_keys == ["optional1", "optional2", "optional3"]
    for snapshot in dataset:
        assert snapshot.node_stores[0]['optional1'].shape == (1,)
        assert snapshot.node_stores[0]['optional2'].shape == (1,)
        assert "optional3" not in list(dict(snapshot.node_stores[0]).keys())


def test_dynamic_hetero_graph_temporal_signal_edges():
    dataset = DynamicHeteroGraphTemporalSignal([{("author", "writes", "paper"): np.array([[0, 1], [1, 0]])}],
                                               [{("author", "writes", "paper"): np.array([[0.1], [0.1]])}],
                                               [{"author": np.array([[0], [0]]),
                                                 "paper": np.array([[0], [0], [0]])}],
                                               [None])
    for snapshot in dataset:
        assert snapshot.edge_stores[0]['edge_index'].shape == (2, 2)
        assert snapshot.edge_stores[0]['edge_attr'].shape == (2, 1)
        assert snapshot.edge_stores[0]['edge_index'].shape[0] == snapshot.edge_stores[0]['edge_attr'].shape[0]



def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102,)
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20,)


def test_pedalme():
    loader = PedalMeDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 225)
            assert snapshot.edge_attr.shape == (225,)
            assert snapshot.x.shape == (15, 4)
            assert snapshot.y.shape == (15,)


def test_wiki():
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(1):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 27079)
            assert snapshot.edge_attr.shape == (27079,)
            assert snapshot.x.shape == (1068, 8)
            assert snapshot.y.shape == (1068,)


def test_windmilllarge():
    loader = WindmillOutputLargeDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 101761)
            assert snapshot.edge_attr.shape == (101761,)
            assert snapshot.x.shape == (319, 8)
            assert snapshot.y.shape == (319,)


def test_windmillsmall():
    loader = WindmillOutputSmallDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 121)
            assert snapshot.edge_attr.shape == (121,)
            assert snapshot.x.shape == (11, 8)
            assert snapshot.y.shape == (11,)


def test_windmillmedium():
    loader = WindmillOutputMediumDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 676)
            assert snapshot.edge_attr.shape == (676,)
            assert snapshot.x.shape == (26, 8)
            assert snapshot.y.shape == (26,)


def test_covid():
    loader = EnglandCovidDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_attr.shape[0] == snapshot.edge_index.shape[1]
            assert snapshot.x.shape == (129, 8)
            assert snapshot.y.shape == (129,)


def test_montevideobus():
    loader = MontevideoBusDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(1):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 690)
            assert snapshot.edge_attr.shape == (690,)
            assert snapshot.x.shape == (675, 4)
            assert snapshot.y.shape == (675,)


def test_metrla():
    loader = METRLADatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 1722)
            assert snapshot.edge_attr.shape == (1722,)
            assert snapshot.x.shape == (207, 2, 12)
            assert snapshot.y.shape == (207, 12)


def test_metrla_task_generator():
    loader = METRLADatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset(num_timesteps_in=6, num_timesteps_out=5)
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 1722)
            assert snapshot.edge_attr.shape == (1722,)
            assert snapshot.x.shape == (207, 2, 6)
            assert snapshot.y.shape == (207, 5)


def test_pemsbay():
    loader = PemsBayDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 2694)
            assert snapshot.edge_attr.shape == (2694,)
            assert snapshot.x.shape == (325, 2, 12)
            assert snapshot.y.shape == (325, 2, 12)


def test_pemsbay_task_generator():
    loader = PemsBayDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset(num_timesteps_in=6, num_timesteps_out=5)
    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 2694)
            assert snapshot.edge_attr.shape == (2694,)
            assert snapshot.x.shape == (325, 2, 6)
            assert snapshot.y.shape == (325, 2, 5)


def check_tennis_data(event_id, node_count, mode, edge_cnt):
    loader = TwitterTennisDatasetLoader(event_id, N=node_count, feature_mode=mode)
    dataset = loader.get_dataset()
    for epoch in range(3):
        i = 0
        for snapshot in dataset:
            if node_count == 1000:
                assert snapshot.edge_index.shape == (2, edge_cnt[i])
                assert snapshot.edge_attr.shape == (edge_cnt[i],)
            else:
                assert snapshot.edge_index.shape[1] <= edge_cnt[i]
                assert snapshot.edge_attr.shape[0] <= edge_cnt[i]
            if mode == "encoded":
                assert snapshot.x.shape == (node_count, 16)
            elif mode == "diagonal":
                assert snapshot.x.shape == (node_count, node_count)
            else:
                assert snapshot.x.shape == (node_count, 2)
            assert snapshot.y.shape == (node_count,)
            i += 1


def test_twitter_tennis_rg17():
    edges_in_snapshots = [
        89,
        61,
        67,
        283,
        569,
        515,
        527,
        262,
        115,
        85,
        127,
        315,
        639,
        841,
        662,
        341,
        136,
        108,
        127,
        257,
        564,
        664,
        646,
        424,
        179,
        82,
        111,
        250,
        689,
        897,
        597,
        352,
        225,
        109,
        81,
        305,
        483,
        816,
        665,
        310,
        141,
        145,
        86,
        285,
        748,
        703,
        682,
        341,
        199,
        102,
        84,
        327,
        786,
        776,
        419,
        208,
        91,
        78,
        83,
        263,
        670,
        880,
        731,
        361,
        122,
        68,
        101,
        269,
        547,
        673,
        612,
        221,
        156,
        99,
        137,
        262,
        373,
        368,
        648,
        288,
        127,
        62,
        84,
        319,
        936,
        889,
        699,
        291,
        186,
        83,
        99,
        191,
        343,
        502,
        561,
        283,
        96,
        92,
        74,
        178,
        461,
        720,
        712,
        279,
        88,
        41,
        74,
        137,
        266,
        664,
        364,
        167,
        68,
        59,
        48,
        178,
        391,
        815,
        315,
        189,
    ]

    check_tennis_data("rg17", 1000, None, edges_in_snapshots)
    check_tennis_data("rg17", 50, "diagonal", edges_in_snapshots)


def test_twitter_tennis_uo17():
    edges_in_snapshots = [
        88,
        113,
        273,
        423,
        718,
        625,
        640,
        758,
        434,
        137,
        289,
        450,
        625,
        489,
        336,
        462,
        284,
        130,
        188,
        335,
        523,
        652,
        584,
        619,
        452,
        198,
        206,
        387,
        464,
        698,
        601,
        434,
        279,
        180,
        162,
        350,
        613,
        793,
        474,
        368,
        231,
        195,
        152,
        404,
        591,
        709,
        642,
        476,
        413,
        248,
        160,
        296,
        521,
        727,
        725,
        542,
        200,
        157,
        268,
        382,
        638,
        612,
        640,
        588,
        250,
        142,
        142,
        197,
        341,
        458,
        395,
        535,
        256,
        128,
        180,
        274,
        732,
        610,
        632,
        732,
        481,
        194,
        206,
        241,
        287,
        304,
        376,
        742,
        196,
        172,
        117,
        220,
        311,
        389,
        610,
        596,
        165,
        183,
        183,
        163,
        406,
        738,
        464,
        209,
        103,
        143,
        115,
        227,
        203,
        455,
        638,
        195,
    ]

    check_tennis_data("uo17", 1000, None, edges_in_snapshots)
    check_tennis_data("uo17", 200, "encoded", edges_in_snapshots)


def test_mtm():
    loader = MTMDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 19)
            assert snapshot.edge_attr.shape == (19,)
            assert snapshot.x.shape == (3, 21, 16)
            assert snapshot.y.shape == (16, 6)


def test_discrete_train_test_split_static():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102,)
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20,)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102,)
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20,)


def test_discrete_train_test_split_dynamic():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features, additional_features = generate_signal(
        250, 100, 32, ["optional1", "optional2"]
    )

    targets = [np.random.uniform(0, 10, (n_count,)) for _ in range(snapshot_count)]

    dataset = DynamicGraphTemporalSignal(
        edge_indices, edge_weights, features, targets, **additional_features
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)


def test_train_test_split_dynamic_graph_static_signal():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features, additional_features = generate_signal(
        250, 100, 32, ["optional1", "optional2"]
    )

    targets = [np.random.uniform(0, 10, (n_count,)) for _ in range(snapshot_count)]
    dataset = StaticGraphTemporalSignal(
        edge_indices[0], edge_weights[0], features, targets, **additional_features
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)


def test_discrete_train_test_split_dynamic_graph_static_signal():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features, additional_features = generate_signal(
        250, 100, 32, ["optional1", "optional2"]
    )

    feature = features[0]

    targets = [np.random.uniform(0, 10, (n_count,)) for _ in range(snapshot_count)]

    dataset = DynamicGraphStaticSignal(
        edge_indices, edge_weights, feature, targets,  **additional_features
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100,)
            assert getattr(snapshot, "optional1").shape == (100, 32)
            assert getattr(snapshot, "optional2").shape == (100, 32)


def test_train_test_split_dynamic_hetero_graph_temporal_signal():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, additional_feature_dicts = generate_heterogeneous_signal(
        snapshot_count, n_count, feature_count, "optional1", "optional2"
    )

    dataset = DynamicHeteroGraphTemporalSignal(
        edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, **additional_feature_dicts
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)


def test_train_test_split_static_hetero_graph_temporal_signal():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, additional_feature_dicts = generate_heterogeneous_signal(
        snapshot_count, n_count, feature_count, "optional1", "optional2"
    )

    dataset = StaticHeteroGraphTemporalSignal(
        edge_index_dicts[0], edge_weight_dicts[0], feature_dicts, target_dicts, **additional_feature_dicts
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)


def test_train_test_split_dynamic_hetero_graph_static_signal():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_index_dicts, edge_weight_dicts, feature_dicts, target_dicts, additional_feature_dicts = generate_heterogeneous_signal(
        snapshot_count, n_count, feature_count, "optional1", "optional2"
    )

    dataset = DynamicHeteroGraphStaticSignal(
        edge_index_dicts, edge_weight_dicts, feature_dicts[0], target_dicts,  **additional_feature_dicts
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in test_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert len(snapshot.node_types) == 2
            assert snapshot.node_types[0] == 'author'
            assert snapshot.node_types[1] == 'paper'
            assert snapshot.node_stores[0]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['x'].shape == (n_count, feature_count)
            assert snapshot.node_stores[0]['y'].shape == (n_count,)
            assert snapshot.node_stores[1]['y'].shape == (n_count,)
            assert len(snapshot.edge_types) == 1
            assert snapshot.edge_types[0] == ('author', 'writes', 'paper')
            assert snapshot.edge_stores[0].edge_index.shape[0] == 2
            assert snapshot.edge_stores[0].edge_index.shape[1] == snapshot.edge_stores[0].edge_attr.shape[0]
            assert snapshot.node_stores[1]['optional1'].shape == (n_count, feature_count)
            assert snapshot.node_stores[1]['optional2'].shape == (n_count, feature_count)
