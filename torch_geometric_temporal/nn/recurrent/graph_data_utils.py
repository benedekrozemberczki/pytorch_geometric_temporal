#!/usr/bin/env python3
"""
图数据处理工具类
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union, Optional

class GraphDataProcessor:
    """图数据处理工具类"""
    
    @staticmethod
    def create_variable_size_batch(graphs: List[torch.Tensor], 
                                 targets: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        创建变长图的batch
        
        Args:
            graphs: 图特征列表，每个元素形状为[num_nodes_i, seq_len, feat_dim]
            targets: 目标值列表，每个元素形状为[num_nodes_i]
            
        Returns:
            处理后的图和目标值列表
        """
        return graphs, targets
    
    @staticmethod
    def pad_graphs_to_same_size(graphs: List[torch.Tensor], 
                              pad_value: float = 0.0) -> torch.Tensor:
        """
        将不同大小的图padding到相同大小
        
        Args:
            graphs: 图特征列表
            pad_value: padding值
            
        Returns:
            padded后的tensor: [batch_size, max_nodes, seq_len, feat_dim]
        """
        if not graphs:
            raise ValueError("Empty graph list")
        
        # 获取最大节点数和其他维度
        max_nodes = max(g.shape[0] for g in graphs)
        seq_len = graphs[0].shape[1]
        feat_dim = graphs[0].shape[2]
        batch_size = len(graphs)
        
        # 创建padded tensor
        padded = torch.full((batch_size, max_nodes, seq_len, feat_dim), 
                          pad_value, dtype=graphs[0].dtype, device=graphs[0].device)
        
        # 填充数据
        for i, graph in enumerate(graphs):
            num_nodes = graph.shape[0]
            padded[i, :num_nodes] = graph
        
        return padded
    
    @staticmethod
    def create_graph_from_adjacency(node_features: torch.Tensor,
                                  adj_matrix: torch.Tensor,
                                  seq_len: Optional[int] = None) -> torch.Tensor:
        """
        从邻接矩阵和节点特征创建图数据
        
        Args:
            node_features: [num_nodes, feat_dim] 或 [num_nodes, seq_len, feat_dim]
            adj_matrix: [num_nodes, num_nodes] 邻接矩阵
            seq_len: 如果node_features是2D，需要指定序列长度
            
        Returns:
            图数据: [num_nodes, seq_len, feat_dim]
        """
        if node_features.dim() == 2:
            if seq_len is None:
                seq_len = 1
            node_features = node_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        return node_features
    
    @staticmethod
    def create_temporal_graph_from_timeseries(timeseries_data: torch.Tensor,
                                            window_size: int,
                                            stride: int = 1) -> List[torch.Tensor]:
        """
        从时序数据创建时序图
        
        Args:
            timeseries_data: [time_steps, num_entities, feat_dim]
            window_size: 时间窗口大小
            stride: 滑动步长
            
        Returns:
            图数据列表，每个元素: [num_entities, window_size, feat_dim]
        """
        time_steps, num_entities, feat_dim = timeseries_data.shape
        graphs = []
        
        for t in range(0, time_steps - window_size + 1, stride):
            window_data = timeseries_data[t:t+window_size].permute(1, 0, 2)  # [num_entities, window_size, feat_dim]
            graphs.append(window_data)
        
        return graphs

class FinancialGraphDataset(Dataset):
    """金融图数据集示例"""
    
    def __init__(self, 
                 stock_data: torch.Tensor,  # [time_steps, num_stocks, features]
                 returns: torch.Tensor,     # [time_steps, num_stocks]
                 window_size: int = 30,
                 prediction_horizon: int = 1,
                 min_stocks: Optional[int] = None,
                 max_stocks: Optional[int] = None):
        """
        Args:
            stock_data: 股票特征数据
            returns: 股票收益率
            window_size: 用于预测的历史窗口大小
            prediction_horizon: 预测时间范围
            min_stocks: 最小股票数（用于创建变长图）
            max_stocks: 最大股票数（用于创建变长图）
        """
        self.stock_data = stock_data
        self.returns = returns
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.variable_size = (min_stocks is not None and max_stocks is not None)
        self.min_stocks = min_stocks or stock_data.shape[1]
        self.max_stocks = max_stocks or stock_data.shape[1]
        
        # 计算可用的样本数
        self.num_samples = stock_data.shape[0] - window_size - prediction_horizon + 1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 获取特征窗口
        feature_window = self.stock_data[idx:idx+self.window_size]  # [window_size, num_stocks, feat_dim]
        
        # 获取目标收益率
        target_returns = self.returns[idx+self.window_size:idx+self.window_size+self.prediction_horizon]
        target = target_returns.mean(dim=0)  # [num_stocks]
        
        if self.variable_size:
            # 随机选择股票子集以创建变长图
            num_stocks_to_select = np.random.randint(self.min_stocks, self.max_stocks + 1)
            selected_indices = np.random.choice(self.stock_data.shape[1], 
                                              num_stocks_to_select, replace=False)
            selected_indices = sorted(selected_indices)
            
            # 选择子集
            feature_subset = feature_window[:, selected_indices].permute(1, 0, 2)  # [selected_stocks, window_size, feat_dim]
            target_subset = target[selected_indices]  # [selected_stocks]
            
            return feature_subset, target_subset
        else:
            # 固定大小图
            feature_window = feature_window.permute(1, 0, 2)  # [num_stocks, window_size, feat_dim]
            return feature_window, target

def create_sample_financial_data(num_days: int = 1000, 
                                num_stocks: int = 100, 
                                num_features: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建示例金融数据
    
    Returns:
        stock_data: [num_days, num_stocks, num_features]
        returns: [num_days, num_stocks]
    """
    # 生成股票特征数据（价格、成交量、技术指标等）
    np.random.seed(42)
    stock_data = torch.randn(num_days, num_stocks, num_features)
    
    # 添加一些趋势和相关性
    for i in range(1, num_days):
        stock_data[i] = 0.95 * stock_data[i-1] + 0.05 * torch.randn(num_stocks, num_features)
    
    # 生成收益率数据
    returns = torch.randn(num_days, num_stocks) * 0.02  # 2%的波动率
    
    return stock_data, returns

# 使用示例
if __name__ == "__main__":
    print("图数据处理工具演示")
    print("=" * 50)
    
    # 1. 创建示例金融数据
    print("1. 创建示例金融数据")
    stock_data, returns = create_sample_financial_data(num_days=500, num_stocks=50, num_features=10)
    print(f"  股票数据形状: {stock_data.shape}")
    print(f"  收益率形状: {returns.shape}")
    
    # 2. 创建固定大小的数据集
    print("\n2. 固定大小图数据集")
    fixed_dataset = FinancialGraphDataset(
        stock_data=stock_data,
        returns=returns,
        window_size=30,
        prediction_horizon=1
    )
    print(f"  数据集大小: {len(fixed_dataset)}")
    
    sample_x, sample_y = fixed_dataset[0]
    print(f"  样本特征形状: {sample_x.shape}")  # [num_stocks, window_size, feat_dim]
    print(f"  样本目标形状: {sample_y.shape}")  # [num_stocks]
    
    # 3. 创建变长图数据集
    print("\n3. 变长图数据集")
    variable_dataset = FinancialGraphDataset(
        stock_data=stock_data,
        returns=returns,
        window_size=30,
        prediction_horizon=1,
        min_stocks=20,
        max_stocks=40
    )
    
    # 显示几个样本
    for i in range(3):
        sample_x, sample_y = variable_dataset[i]
        print(f"  样本{i}: {sample_x.shape[0]}只股票, 特征形状: {sample_x.shape}")
    
    # 4. 使用DataLoader
    print("\n4. DataLoader使用")
    
    def variable_collate_fn(batch):
        graphs, targets = zip(*batch)
        return list(graphs), list(targets)
    
    # 固定大小
    fixed_loader = DataLoader(fixed_dataset, batch_size=4, shuffle=True)
    # 变长大小
    variable_loader = DataLoader(variable_dataset, batch_size=4, shuffle=True, 
                                collate_fn=variable_collate_fn)
    
    print("  固定大小批次:")
    batch_x, batch_y = next(iter(fixed_loader))
    print(f"    特征形状: {batch_x.shape}")  # [batch_size, num_stocks, window_size, feat_dim]
    print(f"    目标形状: {batch_y.shape}")  # [batch_size, num_stocks]
    
    print("  变长批次:")
    batch_graphs, batch_targets = next(iter(variable_loader))
    print(f"    图数量: {len(batch_graphs)}")
    for i, graph in enumerate(batch_graphs):
        print(f"      图{i}: {graph.shape}")
    
    print("\n" + "=" * 50)
    print("数据准备完成！可以用于训练DynamicGraphLightning模型")
