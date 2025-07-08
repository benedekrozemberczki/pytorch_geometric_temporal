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
                 prediction_horizon: int = 1):
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
            targets: [n_stocks, prediction_horizon]
        """
        current_idx = self.valid_indices[idx]
        
        # 获取历史特征序列
        start_idx = current_idx - self.sequence_length
        end_idx = current_idx
        features = self.features[start_idx:end_idx]  # [L, F, N]
        
        # 获取未来目标
        targets = self.targets[current_idx]  # [N, future_periods]
        
        return features, targets

class StockDataModule(pl.LightningDataModule):
    """股票数据模块"""
    
    def __init__(self,
                 data_dir: str = './clean_data',
                 use_factors: bool = True,
                 sequence_length: int = 20,
                 prediction_horizons: List[int] = [1, 5, 10, 20],
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 32,
                 num_workers: int = 4):
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
        
        # 转换为张量
        # features: [T, F, N]
        self.features = torch.tensor(np.stack(feature_list, axis=1), dtype=torch.float32)
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
        print("7. 数据质量检查...")
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
            sequence_length=self.sequence_length,
            prediction_horizon=len(self.prediction_horizons)
        )
        
        self.val_dataset = StockDataset(
            features=self.features[train_end:val_end],
            targets=self.targets[train_end:val_end],
            sequence_length=self.sequence_length,
            prediction_horizon=len(self.prediction_horizons)
        )
        
        self.test_dataset = StockDataset(
            features=self.features[val_end:],
            targets=self.targets[val_end:],
            sequence_length=self.sequence_length,
            prediction_horizon=len(self.prediction_horizons)
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

# 使用示例
def main():
    """使用示例"""
    
    # 示例1：只使用价格数据
    print("=== 示例1：只使用价格数据 ===")
    price_datamodule = StockDataModule(
        use_factors=False,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=32
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
        batch_size=32
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