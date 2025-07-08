import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import pickle
import os
import gzip
from datetime import datetime

class StockDataset(Dataset):
    """股票数据集"""
    def __init__(self, 
                 features: torch.Tensor,  # [T, F, N] 或 [T, N, F]
                 targets: torch.Tensor,   # [T, N, future_periods]
                 sequence_length: int = 20,
                 prediction_horizon: int = 7):
        """
        Args:
            features: 特征数据 [时间, 特征数, 股票数]
            targets: 目标数据 [时间, 股票数, 预测期数]
            sequence_length: 序列长度
            prediction_horizon: 预测期数
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 计算有效样本数量
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self):
        """获取有效的样本索引"""
        max_time = self.features.shape[0]
        # 确保有足够的历史数据和未来数据
        valid_indices = []
        for i in range(self.sequence_length, max_time - self.prediction_horizon + 1):
            valid_indices.append(i)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: [sequence_length, feature_dim, n_stocks]
            targets: [prediction_horizon, n_stocks]
        """
        current_idx = self.valid_indices[idx]
        
        # 获取历史特征序列
        start_idx = current_idx - self.sequence_length
        end_idx = current_idx
        features = self.features[start_idx:end_idx]  # [L, F, N]
        
        # 获取未来 T 期的收益率
        end_target_idx = current_idx + self.prediction_horizon
        targets = self.targets[current_idx:end_target_idx]  # [T, N]
        
        return features, targets

class StockDataModule(pl.LightningDataModule):
    """股票数据模块"""
    
    def __init__(self,
                 data_dir: str = '/home/xu/clean_data',
                 use_factors: bool = True,
                 sequence_length: int = 20,
                 prediction_horizons: List[int] = [1, 5, 10, 20],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 normalize_features: bool = True,
                 normalize_targets: bool = True,
                 normalization_method: str = 'zscore'):
        """
        Args:
            data_dir: 数据目录
            use_factors: 是否使用因子数据
            sequence_length: 序列长度
            prediction_horizons: 预测期数列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            batch_size: 批次大小
            num_workers: 数据加载进程数
            normalize_features: 是否对特征进行标准化
            normalize_targets: 是否对目标进行标准化
            normalization_method: 标准化方法 ('zscore', 'minmax')
        """
        super().__init__()
        self.data_dir = data_dir
        self.use_factors = use_factors
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        self.normalization_method = normalization_method
        
        # 数据文件定义
        price_files = [
            'open.pkl', 'high.pkl', 'low.pkl', 'close.pkl', 'vwap.pkl',
            'open_adj.pkl', 'high_adj.pkl', 'low_adj.pkl', 'close_adj.pkl', 'vwap_adj.pkl',
            'volume.pkl', 'volume_adj.pkl', 'total_mv.pkl', 'total_share.pkl', 'turnover.pkl'
        ]
        self.price_files = [f"filtered_{filename}" for filename in price_files]
        
        factor_files = [
            'momentum.pkl', 'resvol.pkl', 'beta.pkl', 'srisk.pkl', 'ltrevrsl.pkl',
            'btop.pkl', 'earnyild.pkl', 'earnqlty.pkl', 'earnvar.pkl', 'divyild.pkl',
            'liquidty.pkl', 'invsqlty.pkl', 'size.pkl', 'midcap.pkl',
            'growth.pkl', 'profit.pkl', 'leverage.pkl'
        ]
        self.factor_files = [f"filtered_{filename}" for filename in factor_files]
        # 收益率文件
        self.return_files = [f'returns_{h}d.pkl' for h in prediction_horizons]
        
        # 数据容器
        self.features = None
        self.targets = None
        self.feature_names = None
        self.stock_names = None
        self.date_index = None
        
        # 标准化统计信息
        self.feature_stats = None  # 特征的均值和标准差
        self.target_stats = None   # 目标的均值和标准差
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_pkl_safe(self, filename: str) -> Optional[pd.DataFrame]:
        """安全加载pkl文件"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"加载 {filename} 失败: {e}")
            return None
    
    def prepare_data(self):
        """准备数据（下载、预处理等）"""
        print("=== 准备股票数据 ===")
        
        # 检查必要文件是否存在
        required_files = self.price_files + self.return_files
        if self.use_factors:
            required_files += self.factor_files
        
        missing_files = []
        for file in required_files:
            filepath = os.path.join(self.data_dir, file)
            if not os.path.exists(filepath):
                missing_files.append(file)
        
        if missing_files:
            print(f"缺少文件: {missing_files}")
            raise FileNotFoundError(f"缺少必要的数据文件: {missing_files}")
        
        print("✓ 所有必要文件都存在")
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == 'fit' or stage is None:
            self._load_and_process_data()
            self._split_data()
        
        if stage == 'test' or stage is None:
            if self.features is None:
                self._load_and_process_data()
                self._split_data()
    
    def _load_and_process_data(self):
        """加载并处理数据"""
        print("=== 加载数据 ===")
        
        # 1. 加载价格数据
        print("1. 加载价格数据...")
        price_data = {}
        for file in self.price_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                price_data[file.replace('.pkl', '')] = data
                print(f"  ✓ {file}: {data.shape}")
            else:
                print(f"  ✗ {file}: 加载失败")
        
        # 2. 加载因子数据（如果使用）
        factor_data = {}
        if self.use_factors:
            print("2. 加载因子数据...")
            for file in self.factor_files:
                data = self.load_pkl_safe(file)
                if data is not None:
                    factor_data[file.replace('.pkl', '')] = data
                    print(f"  ✓ {file}: {data.shape}")
                else:
                    print(f"  ✗ {file}: 加载失败")
        
        # 3. 加载收益率数据
        print("3. 加载收益率数据...")
        return_data = {}
        for file in self.return_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                return_data[file.replace('.pkl', '')] = data
                print(f"  ✓ {file}: {data.shape}")
            else:
                print(f"  ✗ {file}: 加载失败")
        
        # 4. 数据对齐和处理
        print("4. 数据对齐...")
        self._align_and_process_data(price_data, factor_data, return_data)
    
    def _align_and_process_data(self, price_data: Dict, factor_data: Dict, return_data: Dict):
        """对齐并处理数据"""
        # 找到公共的股票和时间
        all_data = {**price_data, **factor_data, **return_data}
        
        # 获取公共股票
        common_stocks = None
        for name, data in all_data.items():
            if data is not None and not data.empty:
                if common_stocks is None:
                    common_stocks = set(data.columns)
                else:
                    common_stocks = common_stocks.intersection(set(data.columns))
        
        if common_stocks is None or len(common_stocks) == 0:
            raise ValueError("没有找到公共股票")
        
        common_stocks = sorted(list(common_stocks))
        print(f"  公共股票数量: {len(common_stocks)}")
        
        # 获取公共时间范围 - 修改版本：按可用数据保留
        print("4. 确定时间范围...")
        if self.use_factors and factor_data:
            # 如果使用因子数据，以因子数据的时间范围为准
            common_dates = None
            for name, data in factor_data.items():
                if data is not None and not data.empty:
                    if common_dates is None:
                        common_dates = set(data.index)
                    else:
                        common_dates = common_dates.intersection(set(data.index))
            
            if common_dates is None:
                raise ValueError("因子数据没有公共时间")
            
            common_dates = sorted(list(common_dates))
            print(f"  因子数据时间范围: {min(common_dates)} 到 {max(common_dates)}")
            
            # 检查价格数据和收益率数据的可用性，但不强制要求完整覆盖
            available_dates_by_source = {}
            for name, data in {**price_data, **return_data}.items():
                if data is not None:
                    available_dates = set(data.index)
                    overlap = set(common_dates).intersection(available_dates)
                    available_dates_by_source[name] = overlap
                    coverage = len(overlap) / len(common_dates) * 100
                    print(f"  {name} 覆盖率: {coverage:.1f}% ({len(overlap)}/{len(common_dates)})")
            
            # 更新common_dates为所有数据源都有的日期
            all_available_dates = set(common_dates)
            for name, dates in available_dates_by_source.items():
                all_available_dates = all_available_dates.intersection(dates)
            
            common_dates = sorted(list(all_available_dates))
            print(f"  最终时间范围: {min(common_dates)} 到 {max(common_dates)} (共{len(common_dates)}天)")
            
        else:
            # 如果不使用因子数据，以价格数据的时间范围为准
            common_dates = None
            for name, data in price_data.items():
                if data is not None and not data.empty:
                    if common_dates is None:
                        common_dates = set(data.index)
                    else:
                        common_dates = common_dates.intersection(set(data.index))
            
            if common_dates is None:
                raise ValueError("价格数据没有公共时间")
            
            # 检查收益率数据的可用性
            available_dates_by_source = {}
            for name, data in return_data.items():
                if data is not None:
                    available_dates = set(data.index)
                    overlap = set(common_dates).intersection(available_dates)
                    available_dates_by_source[name] = overlap
                    coverage = len(overlap) / len(common_dates) * 100
                    print(f"  {name} 覆盖率: {coverage:.1f}% ({len(overlap)}/{len(common_dates)})")
            
            # 更新common_dates为所有数据源都有的日期
            all_available_dates = set(common_dates)
            for name, dates in available_dates_by_source.items():
                all_available_dates = all_available_dates.intersection(dates)
            
            common_dates = sorted(list(all_available_dates))
            print(f"  最终时间范围: {min(common_dates)} 到 {max(common_dates)} (共{len(common_dates)}天)")
        
        # 构建特征矩阵
        print("5. 构建特征矩阵...")
        feature_list = []
        feature_names = []
        
        # 添加价格特征
        for name, data in price_data.items():
            if data is not None:
                # 只保留common_dates和common_stocks的交集
                aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                # 对于缺失值，先用前向填充，然后用0填充
                aligned_data = aligned_data.fillna(method='ffill').fillna(0)
                feature_list.append(aligned_data.values)  # [T, N]
                feature_names.append(name)
                print(f"  添加特征: {name}, 形状: {aligned_data.shape}")
        
        # 添加因子特征（如果使用）
        if self.use_factors:
            for name, data in factor_data.items():
                if data is not None:
                    # 只保留common_dates和common_stocks的交集
                    aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                    # 对于缺失值，先用前向填充，然后用0填充
                    aligned_data = aligned_data.fillna(method='ffill').fillna(0)
                    feature_list.append(aligned_data.values)  # [T, N]
                    feature_names.append(name)
                    print(f"  添加特征: {name}, 形状: {aligned_data.shape}")
        
        # 构建目标矩阵
        print("6. 构建目标矩阵...")
        target_list = []
        for horizon in self.prediction_horizons:
            return_key = f'returns_{horizon}d'
            if return_key in return_data and return_data[return_key] is not None:
                # 只保留common_dates和common_stocks的交集
                aligned_data = return_data[return_key].reindex(index=common_dates, columns=common_stocks)
                # 目标值的缺失用0填充（表示无收益）
                aligned_data = aligned_data.fillna(0)
                target_list.append(aligned_data.values)  # [T, N]
                print(f"  添加目标: {return_key}, 形状: {aligned_data.shape}")
        
        # 检查是否有有效数据
        if len(feature_list) == 0:
            raise ValueError("没有有效的特征数据")
        if len(target_list) == 0:
            raise ValueError("没有有效的目标数据")
        
        # 转换为张量前先进行标准化处理
        print("7. 数据标准化...")
        
        # 7.1 特征标准化
        if self.normalize_features:
            print("  对特征进行标准化 (按特征在所有股票上标准化)...")
            # Stack features: [T, F, N]
            features_array = np.stack(feature_list, axis=1)
            
            # 标准化处理
            normalized_features, self.feature_stats = self._normalize_data(
                features_array, fit=True
            )
            
            print(f"  特征标准化完成: {self.normalization_method}")
            print(f"  原始特征范围: [{features_array.min():.4f}, {features_array.max():.4f}]")
            print(f"  标准化后范围: [{normalized_features.min():.4f}, {normalized_features.max():.4f}]")
            
            # 验证标准化效果：每个特征在所有股票上的均值和标准差
            for f in range(features_array.shape[1]):
                feature_data = features_array[:, f, :].flatten()  # 展平时间和股票维度
                norm_feature_data = normalized_features[:, f, :].flatten()
                print(f"    特征 {f}: 原始均值={feature_data.mean():.4f}, 标准差={feature_data.std():.4f}")
                print(f"    特征 {f}: 标准化后均值={norm_feature_data.mean():.4f}, 标准差={norm_feature_data.std():.4f}")
            
            self.features = torch.tensor(normalized_features, dtype=torch.float32)
        else:
            # features: [T, F, N]
            self.features = torch.tensor(np.stack(feature_list, axis=1), dtype=torch.float32)
        
        # 7.2 目标标准化
        if self.normalize_targets:
            print("  对目标进行标准化 (按预测期在所有股票上标准化)...")
            # Stack targets: [T, N, H]
            targets_array = np.stack(target_list, axis=2)
            
            # 标准化处理
            normalized_targets, self.target_stats = self._normalize_data(
                targets_array, fit=True
            )
            
            print(f"  目标标准化完成: {self.normalization_method}")
            print(f"  原始目标范围: [{targets_array.min():.4f}, {targets_array.max():.4f}]")
            print(f"  标准化后范围: [{normalized_targets.min():.4f}, {normalized_targets.max():.4f}]")
            
            # 验证标准化效果：每个预测期在所有股票上的均值和标准差
            for h in range(targets_array.shape[2]):
                target_data = targets_array[:, :, h].flatten()  # 展平时间和股票维度
                norm_target_data = normalized_targets[:, :, h].flatten()
                print(f"    预测期 {h}: 原始均值={target_data.mean():.4f}, 标准差={target_data.std():.4f}")
                print(f"    预测期 {h}: 标准化后均值={norm_target_data.mean():.4f}, 标准差={norm_target_data.std():.4f}")
            
            self.targets = torch.tensor(normalized_targets, dtype=torch.float32)
        else:
            # targets: [T, N, H] (H是预测期数)
            self.targets = torch.tensor(np.stack(target_list, axis=2), dtype=torch.float32)
        
        self.feature_names = feature_names
        self.stock_names = common_stocks
        self.date_index = common_dates
        
        print(f"  特征矩阵形状: {self.features.shape}")
        print(f"  目标矩阵形状: {self.targets.shape}")
        print(f"  特征名称: {self.feature_names}")
        print(f"  预测期数: {self.prediction_horizons}")
        
        # 数据质量检查
        print("8. 数据质量检查...")
        feature_nan_count = torch.isnan(self.features).sum().item()
        target_nan_count = torch.isnan(self.targets).sum().item()
        print(f"  特征数据NaN数量: {feature_nan_count}")
        print(f"  目标数据NaN数量: {target_nan_count}")
        
        if feature_nan_count > 0 or target_nan_count > 0:
            print("  警告: 数据中仍有NaN值，可能影响训练")
    
    def _split_data(self):
        """按时间划分数据集"""
        print("=== 划分数据集 ===")
        
        total_time = self.features.shape[0]
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        print(f"总时间步数: {total_time}")
        print(f"训练集: 0 - {train_end} ({train_end} 步)")
        print(f"验证集: {train_end} - {val_end} ({val_end - train_end} 步)")
        print(f"测试集: {val_end} - {total_time} ({total_time - val_end} 步)")
        
        # 创建数据集
        self.train_dataset = StockDataset(
            features=self.features[:train_end],
            targets=self.targets[:train_end],
            sequence_length=self.sequence_length
        )
        
        self.val_dataset = StockDataset(
            features=self.features[train_end:val_end],
            targets=self.targets[train_end:val_end],
            sequence_length=self.sequence_length
        )
        
        self.test_dataset = StockDataset(
            features=self.features[val_end:],
            targets=self.targets[val_end:],
            sequence_length=self.sequence_length
        )
        
        print(f"训练样本数: {len(self.train_dataset)}")
        print(f"验证样本数: {len(self.val_dataset)}")
        print(f"测试样本数: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self):
        """获取特征维度"""
        return len(self.feature_names) if self.feature_names else 0
    
    def get_stock_num(self):
        """获取股票数量"""
        return len(self.stock_names) if self.stock_names else 0
    
    def get_prediction_horizons(self):
        """获取预测期数"""
        return self.prediction_horizons
    
    def _normalize_data(self, data: np.ndarray, stats: Optional[Dict] = None, fit: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        标准化数据 - 按特征在所有股票上进行标准化
        
        Args:
            data: 输入数据 [T, F, N] 或 [T, N, H]
            stats: 已有的统计信息 (用于验证/测试集)
            fit: 是否计算统计信息 (True for training, False for val/test)
            
        Returns:
            normalized_data: 标准化后的数据
            stats: 统计信息字典
        """
        if self.normalization_method == 'zscore':
            if fit or stats is None:
                if data.ndim == 3 and data.shape[1] > data.shape[2]:  # Features: [T, F, N]
                    # 对每个特征F，在时间T和股票N维度上计算统计量
                    # 重塑为 [T*N, F] 来计算每个特征的全局统计量
                    reshaped_data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # [F, T*N]
                    mean = np.mean(reshaped_data, axis=1, keepdims=True)  # [F, 1]
                    std = np.std(reshaped_data, axis=1, keepdims=True)    # [F, 1]
                    std = np.where(std == 0, 1.0, std)  # 避免除零
                    
                    # 重塑回原始形状用于广播
                    mean = mean.reshape(1, -1, 1)  # [1, F, 1]
                    std = std.reshape(1, -1, 1)    # [1, F, 1]
                    
                else:  # Targets: [T, N, H]
                    # 对每个预测期H，在时间T和股票N维度上计算统计量
                    # 重塑为 [T*N, H] 来计算每个预测期的全局统计量
                    reshaped_data = data.reshape(-1, data.shape[2])  # [T*N, H]
                    mean = np.mean(reshaped_data, axis=0, keepdims=True)  # [1, H]
                    std = np.std(reshaped_data, axis=0, keepdims=True)    # [1, H]
                    std = np.where(std == 0, 1.0, std)  # 避免除零
                    
                    # 重塑回原始形状用于广播
                    mean = mean.reshape(1, 1, -1)  # [1, 1, H]
                    std = std.reshape(1, 1, -1)    # [1, 1, H]
                
                stats = {'mean': mean, 'std': std}
            else:
                mean = stats['mean']
                std = stats['std']
            
            normalized_data = (data - mean) / std
            
        elif self.normalization_method == 'minmax':
            if fit or stats is None:
                if data.ndim == 3 and data.shape[1] > data.shape[2]:  # Features: [T, F, N]
                    # 对每个特征F，在时间T和股票N维度上计算min/max
                    reshaped_data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # [F, T*N]
                    data_min = np.min(reshaped_data, axis=1, keepdims=True)  # [F, 1]
                    data_max = np.max(reshaped_data, axis=1, keepdims=True)  # [F, 1]
                    data_range = data_max - data_min
                    data_range = np.where(data_range == 0, 1.0, data_range)  # 避免除零
                    
                    # 重塑回原始形状用于广播
                    data_min = data_min.reshape(1, -1, 1)    # [1, F, 1]
                    data_range = data_range.reshape(1, -1, 1)  # [1, F, 1]
                    
                else:  # Targets: [T, N, H]
                    # 对每个预测期H，在时间T和股票N维度上计算min/max
                    reshaped_data = data.reshape(-1, data.shape[2])  # [T*N, H]
                    data_min = np.min(reshaped_data, axis=0, keepdims=True)  # [1, H]
                    data_max = np.max(reshaped_data, axis=0, keepdims=True)  # [1, H]
                    data_range = data_max - data_min
                    data_range = np.where(data_range == 0, 1.0, data_range)  # 避免除零
                    
                    # 重塑回原始形状用于广播
                    data_min = data_min.reshape(1, 1, -1)    # [1, 1, H]
                    data_range = data_range.reshape(1, 1, -1)  # [1, 1, H]
                
                stats = {'min': data_min, 'max': data_max, 'range': data_range}
            else:
                data_min = stats['min']
                data_range = stats['range']
            
            normalized_data = (data - data_min) / data_range
            
        else:
            raise ValueError(f"不支持的标准化方法: {self.normalization_method}")
        
        return normalized_data, stats

    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """
        反标准化目标数据
        
        Args:
            normalized_targets: 标准化的目标数据 [..., H]
            
        Returns:
            原始尺度的目标数据
        """
        if not self.normalize_targets or self.target_stats is None:
            return normalized_targets
        
        if self.normalization_method == 'zscore':
            mean = torch.tensor(self.target_stats['mean'], dtype=torch.float32)
            std = torch.tensor(self.target_stats['std'], dtype=torch.float32)
            
            # 确保维度匹配用于广播
            if normalized_targets.dim() == 3:  # [B, N, H]
                mean = mean.view(1, 1, -1)
                std = std.view(1, 1, -1)
            elif normalized_targets.dim() == 2:  # [B*N, H] 或 [N, H]
                mean = mean.view(1, -1)
                std = std.view(1, -1)
            
            return normalized_targets * std + mean
            
        elif self.normalization_method == 'minmax':
            data_min = torch.tensor(self.target_stats['min'], dtype=torch.float32)
            data_range = torch.tensor(self.target_stats['range'], dtype=torch.float32)
            
            # 确保维度匹配用于广播
            if normalized_targets.dim() == 3:  # [B, N, H]
                data_min = data_min.view(1, 1, -1)
                data_range = data_range.view(1, 1, -1)
            elif normalized_targets.dim() == 2:  # [B*N, H] 或 [N, H]
                data_min = data_min.view(1, -1)
                data_range = data_range.view(1, -1)
            
            return normalized_targets * data_range + data_min
        
        return normalized_targets

    def get_normalization_stats(self):
        """获取标准化统计信息"""
        return {
            'feature_stats': self.feature_stats,
            'target_stats': self.target_stats,
            'normalization_method': self.normalization_method
        }

# 使用示例
def main():
    """使用示例"""
    
    # 示例1：只使用价格数据
    print("=== 示例1：只使用价格数据 ===")
    price_datamodule = StockDataModule(
        use_factors=False,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=32,
        normalize_features=True,
        normalize_targets=True,
        normalization_method='zscore'
    )
    
    price_datamodule.prepare_data()
    price_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = price_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    features, targets = batch
    
    print(f"特征形状: {features.shape}")  # [B, L, F, N]
    print(f"目标形状: {targets.shape}")   # [B, N, H]
    print(f"特征维度: {price_datamodule.get_feature_dim()}")
    print(f"股票数量: {price_datamodule.get_stock_num()}")
    
    # 示例2：使用因子数据和价格数据
    print("\n=== 示例2：使用因子数据和价格数据 ===")
    factor_datamodule = StockDataModule(
        use_factors=True,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=32,
        normalize_features=True,
        normalize_targets=True,
        normalization_method='zscore'
    )
    
    factor_datamodule.prepare_data()
    factor_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = factor_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    features, targets = batch
    
    print(f"特征形状: {features.shape}")  # [B, L, F, N]
    print(f"目标形状: {targets.shape}")   # [B, N, H]
    print(f"特征维度: {factor_datamodule.get_feature_dim()}")
    print(f"股票数量: {factor_datamodule.get_stock_num()}")

if __name__ == "__main__":
    main()
