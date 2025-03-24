import os
import ast
from collections import defaultdict

def find_defined_classes_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read(), filename=filepath)
        except SyntaxError:
            return []

    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

def map_defined_classes(directories):
    file_to_classes = defaultdict(list)
    all_classes = set()
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    class_names = find_defined_classes_in_file(full_path)
                    if class_names:
                        file_to_classes[full_path[3:].replace("/",".")].extend(class_names)
                        all_classes.update(class_names)

    return all_classes, file_to_classes

if __name__ == "__main__":
    # Replace these with the paths you want to analyze
    directories_to_scan = [
        "../torch_geometric_temporal/nn/recurrent",
        "../torch_geometric_temporal/nn/attention",
        "../torch_geometric_temporal/nn/hetero"
    ]

    all_classes, mapping = map_defined_classes(directories_to_scan)


    order = [

        "torch_geometric_temporal.nn.recurrent.gconv_gru",
        "torch_geometric_temporal.nn.recurrent.gconv_lstm",
        "torch_geometric_temporal.nn.recurrent.gc_lstm",
        "torch_geometric_temporal.nn.recurrent.lrgcn",
        "torch_geometric_temporal.nn.recurrent.dygrae",
        "torch_geometric_temporal.nn.recurrent.evolvegcnh",
        "torch_geometric_temporal.nn.recurrent.evolvegcno",
        "torch_geometric_temporal.nn.recurrent.temporalgcn",
        "torch_geometric_temporal.nn.recurrent.attentiontemporalgcn",
        "torch_geometric_temporal.nn.recurrent.mpnn_lstm",
        "torch_geometric_temporal.nn.recurrent.dcrnn",
        "torch_geometric_temporal.nn.recurrent.agcrn",

        "torch_geometric_temporal.nn.attention.stgcn", 
        "torch_geometric_temporal.nn.attention.astgcn", 
        "torch_geometric_temporal.nn.attention.mstgcn", 
        "torch_geometric_temporal.nn.attention.gman", 
        "torch_geometric_temporal.nn.attention.mtgnn", 
        "torch_geometric_temporal.nn.attention.tsagcn", 
        "torch_geometric_temporal.nn.attention.dnntsp", 

        "torch_geometric_temporal.nn.hetero.heterogclstm"

        ]
    model = [
        "GConvGRU",
        "GConvLSTM",
        "GCLSTM",
        "LRGCN",
        "DyGrEncoder",
        "EvolveGCNH",
        "EvolveGCNO",
        "GCNConv_Fixed_W",
        "TGCN",
        "TGCN2",
        "A3TGCN",
        "A3TGCN2",
        "MPNNLSTM",
        "DCRNN",
        "BatchedDCRNN",
        "AGCRN",
        "STConv",
        "ASTGCN",
        "MSTGCN",
        "GMAN",
        "SpatioTemporalAttention",
        "GraphConstructor",
        "MTGNN",
        "AAGCN",
        "DNNTSP"
    ]
    aux = [
        "TemporalConv",
        "DConv",
        "BatchedDConv",
        "ChebConvAttention",
        "AVWGCN",
        "UnitGCN",
        "UnitTCN"
    ]

    het = [
        "HeteroGCLSTM"
    ]
    # print(mapping.keys())
    target = {}
    for file, classes in mapping.items():


        line = ""
            
        for c in all_classes:
            if c not in classes:
                line += f"{c}, "
        
        line = line[:-2]
        target[file[:-3]] = line
        

    for key in order:
        print(f".. autoapimodule:: {key}")
        print("\t:members:")
        print(f"\t:exclude-members: {target[key]}, LayerNormalization, AggregateTemporalNodeFeatures, GlobalGatedUpdater, MaskedSelfAttention, WeightedGCNBlock, LayerNormalization, K, bias, in_channels, out_channels, normalization, num_bases, num_relations, conv_aggr, conv_num_layers, conv_out_channels, lstm_num_layers, lstm_out_channels, add_self_loops, cached, improved, initial_weight, normalize, num_of_nodes, reinitialize_weight, reset_parameters, weight, batch_size, periods, dropout, hidden_size, num_nodes, window, number_of_nodes, bias_pool, weights_pool, hidden_channels, A, attention, edge_index, gcn1, graph, relu, tcn1, kernel_size, conv_1, conv_2, conv_3, nb_time_filter, adaptive, bn, conv_d, in_c, inter_c, num_jpts, num_subset, out_c, sigmoid, soft, tan, conv, embedding_dimensions, Wq, global_gated_updater, item_embedding, item_embedding_dim, items_total, masked_self_attention, stacked_gcn, in_channels_dict, meta")
        print()