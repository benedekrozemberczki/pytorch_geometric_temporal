#!/usr/bin/env python3
"""
示例：如何为DynamicGraphLightning准备不同大小的图数据
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/root/pytorch_geometric_temporal')

from torch_geometric_temporal.nn.recurrent.adaptive_adj import DynamicGraphLightning

class VariableSizeGraphDataset(Dataset):
    """处理不同大小图的数据集"""
    
    def __init__(self, num_samples=100, min_nodes=10, max_nodes=100, 
                 seq_len=30, node_feat_dim=64):
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.seq_len = seq_len
        self.node_feat_dim = node_feat_dim
        
        # 预生成数据
        self.data = []
        self.targets = []
        
        for i in range(num_samples):
            # 随机选择图的大小
            num_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # 生成图的时序特征: [num_nodes, seq_len, node_feat_dim]
            graph_features = torch.randn(num_nodes, seq_len, node_feat_dim)
            
            # 生成目标值: [num_nodes] (每个节点一个预测值)
            target = torch.randn(num_nodes)
            
            self.data.append(graph_features)
            self.targets.append(target)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def custom_collate_fn(batch):
    """自定义的collate函数，用于处理不同大小的图"""
    # batch是一个列表，每个元素是(graph_features, target)
    graphs = []
    targets = []
    
    for graph_features, target in batch:
        graphs.append(graph_features)
        targets.append(target)
    
    return graphs, targets

class FixedSizeGraphDataset(Dataset):
    """处理固定大小图的数据集（传统方式）"""
    
    def __init__(self, num_samples=100, num_nodes=50, seq_len=30, node_feat_dim=64):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.node_feat_dim = node_feat_dim
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成固定大小的图: [num_nodes, seq_len, node_feat_dim]
        graph_features = torch.randn(self.num_nodes, self.seq_len, self.node_feat_dim)
        target = torch.randn(self.num_nodes)
        return graph_features, target

def demo_variable_size_graphs():
    """演示如何使用不同大小的图"""
    print("=== 演示变长图数据 ===")
    
    # 创建数据集
    dataset = VariableSizeGraphDataset(
        num_samples=20, 
        min_nodes=10, 
        max_nodes=50,
        seq_len=30, 
        node_feat_dim=64
    )
    
    # 创建DataLoader，使用自定义collate函数
    dataloader = DataLoader(
        dataset, 
        batch_size=4,  # 每个batch包含4个不同大小的图
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # 创建模型
    model = DynamicGraphLightning(
        node_feat_dim=64,
        gru_hidden_dim=32,
        gnn_hidden_dim=32,
        k_nn=5
    )
    
    # 测试前向传播
    for batch_idx, (graphs, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  图数量: {len(graphs)}")
        for i, graph in enumerate(graphs):
            print(f"  图{i}: 节点数={graph.shape[0]}, 序列长度={graph.shape[1]}, 特征维度={graph.shape[2]}")
        
        # 前向传播
        try:
            outputs = model(graphs)
            print(f"  输出: {len(outputs)}个图的预测结果")
            for i, output in enumerate(outputs):
                print(f"    图{i}输出形状: {output.shape}")
            
            print("  ✓ 前向传播成功")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        
        if batch_idx >= 2:  # 只测试前几个batch
            break

def demo_fixed_size_graphs():
    """演示如何使用固定大小的图（传统方式）"""
    print("\n=== 演示固定大小图数据 ===")
    
    # 创建数据集
    dataset = FixedSizeGraphDataset(
        num_samples=20,
        num_nodes=30,
        seq_len=20,
        node_feat_dim=64
    )
    
    # 创建DataLoader（使用默认collate函数）
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建模型
    model = DynamicGraphLightning(
        node_feat_dim=64,
        gru_hidden_dim=32,
        gnn_hidden_dim=32,
        k_nn=5
    )
    
    # 测试前向传播
    for batch_idx, (graphs, targets) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  输入形状: {graphs.shape}")  # [batch_size, num_nodes, seq_len, feat_dim]
        print(f"  目标形状: {targets.shape}")  # [batch_size, num_nodes]
        
        try:
            outputs = model(graphs)
            print(f"  输出形状: {outputs.shape}")  # [batch_size, num_nodes, 1]
            print("  ✓ 前向传播成功")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        
        if batch_idx >= 2:  # 只测试前几个batch
            break

def demo_mixed_usage():
    """演示混合使用：单个图和图列表"""
    print("\n=== 演示混合使用 ===")
    
    model = DynamicGraphLightning(
        node_feat_dim=32,
        gru_hidden_dim=16,
        gnn_hidden_dim=16,
        k_nn=3
    )
    
    # 测试1: 单个图
    print("测试单个图:")
    single_graph = torch.randn(20, 10, 32)  # [num_nodes, seq_len, feat_dim]
    try:
        output = model([single_graph])  # 注意：需要放在列表中
        print(f"  输出: {len(output)}个图，形状: {output[0].shape}")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 测试2: 多个不同大小的图
    print("测试多个不同大小的图:")
    graph_list = [
        torch.randn(15, 10, 32),  # 15个节点
        torch.randn(25, 10, 32),  # 25个节点
        torch.randn(10, 10, 32),  # 10个节点
    ]
    try:
        outputs = model(graph_list)
        print(f"  输出: {len(outputs)}个图")
        for i, output in enumerate(outputs):
            print(f"    图{i}: {output.shape}")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    # 测试3: 固定大小批量
    print("测试固定大小批量:")
    batch_tensor = torch.randn(4, 20, 10, 32)  # [batch, nodes, seq, feat]
    try:
        output = model(batch_tensor)
        print(f"  输出形状: {output.shape}")
        print("  ✓ 成功")
    except Exception as e:
        print(f"  ✗ 错误: {e}")

if __name__ == "__main__":
    print("DynamicGraphLightning 数据使用示例")
    print("=" * 50)
    
    # 演示不同的使用方式
    demo_variable_size_graphs()
    demo_fixed_size_graphs() 
    demo_mixed_usage()
    
    print("\n" + "=" * 50)
    print("总结:")
    print("1. 变长图: 使用list[torch.Tensor]，每个tensor形状为[num_nodes_i, seq_len, feat_dim]")
    print("2. 固定图: 使用torch.Tensor，形状为[batch_size, num_nodes, seq_len, feat_dim]")
    print("3. 需要自定义collate_fn来处理变长图的DataLoader")
    print("4. 模型会自动检测输入类型并选择合适的处理方式")
