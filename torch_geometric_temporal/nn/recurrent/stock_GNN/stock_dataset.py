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
    """股票数据集 - 支持序列级别的价格标准化"""
    def __init__(self, 
                 features: torch.Tensor,  # [T, F, N] 
                 targets: torch.Tensor,   # [T, N, H]
                 feature_names: List[str],
                 sequence_length: int = 20,
                 prediction_horizon: int = 7,
                 normalize_features: bool = True):
        """
        Args:
            features: 特征数据 [时间, 特征数, 股票数]
            targets: 目标数据 [时间, 股票数, 预测期数]
            feature_names: 特征名称列表
            sequence_length: 序列长度
            prediction_horizon: 预测期数
            normalize_features: 是否对特征进行序列级标准化
        """
        self.features = features
        self.targets = targets
        self.feature_names = feature_names
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize_features = normalize_features
        
        # 定义价格特征和成交量特征的索引
        self.price_feature_indices = []
        self.volume_feature_indices = []
        self.close_feature_index = None
        
        for i, name in enumerate(feature_names):
            if any(price_name in name for price_name in ['open_adj', 'high_adj', 'low_adj', 'close_adj', 'vwap_adj']):
                self.price_feature_indices.append(i)
                if 'close_adj' in name:
                    self.close_feature_index = i
            elif any(vol_name in name for vol_name in ['volume_adj', 'turnover']):
                self.volume_feature_indices.append(i)
        
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
        
        # 如果需要标准化特征，应用序列级别的标准化
        if self.normalize_features:
            features = self._normalize_sequence_features(features)
        
        # 获取未来 T 期的收益率
        end_target_idx = current_idx + self.prediction_horizon
        targets = self.targets[current_idx:end_target_idx]  # [T, N]
        
        return features, targets
    
    def _normalize_sequence_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        对序列特征进行标准化
        - 价格特征：用最后一天的close价格标准化该股票的OHLCV
        - 成交量特征：用最后一天的成交量标准化该股票的成交量序列
        - 因子特征：不标准化（已预处理）
        
        Args:
            features: [L, F, N] 序列特征
            
        Returns:
            normalized_features: [L, F, N] 标准化后的序列特征
        """
        L, F, N = features.shape
        normalized_features = features.clone()
        
        # 对每只股票分别进行标准化
        for stock_idx in range(N):
            stock_features = features[:, :, stock_idx]  # [L, F]
            
            # 价格特征标准化：用最后一天的close价格
            if self.close_feature_index is not None and len(self.price_feature_indices) > 0:
                last_close = stock_features[-1, self.close_feature_index]  # 最后一天的收盘价
                
                # 避免除零
                if last_close != 0 and not torch.isnan(last_close) and not torch.isinf(last_close):
                    for price_idx in self.price_feature_indices:
                        normalized_features[:, price_idx, stock_idx] = stock_features[:, price_idx] / last_close
                else:
                    # 如果close价格无效，保持原值
                    pass
            
            # 成交量特征标准化：用最后一天的成交量
            for vol_idx in self.volume_feature_indices:
                last_volume = stock_features[-1, vol_idx]  # 最后一天的成交量
                
                # 避免除零
                if last_volume != 0 and not torch.isnan(last_volume) and not torch.isinf(last_volume):
                    normalized_features[:, vol_idx, stock_idx] = stock_features[:, vol_idx] / last_volume
                else:
                    # 如果成交量无效，保持原值
                    pass
        
        return normalized_features

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
                 normalization_method: str = 'sequence_price_cross_section_return',
                 debug: bool = False):
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
            normalization_method: 标准化方法 ('sequence_price_cross_section_return')
            debug: 是否开启调试模式
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
        self.debug = debug  # 添加调试选项
        
        # 数据文件定义 - 只使用复权调整的价格数据
        price_files = [
            'open_adj.pkl', 'high_adj.pkl', 'low_adj.pkl', 'close_adj.pkl', 'vwap_adj.pkl',
            'volume_adj.pkl', 'turnover.pkl'
        ]
        self.price_files = [f"filtered_{filename}" for filename in price_files]
        
        # 定义价格特征和成交量特征
        self.price_features = ['filtered_open_adj', 'filtered_high_adj', 'filtered_low_adj', 
                              'filtered_close_adj', 'filtered_vwap_adj']
        self.volume_features = ['filtered_volume_adj', 'filtered_turnover']
        
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
    
    def _debug_print(self, *args, **kwargs):
        """调试打印方法 - 只在debug模式下打印"""
        if self.debug:
            print(*args, **kwargs)
    
    def _print(self, *args, **kwargs):
        """普通打印方法 - 总是打印"""
        print(*args, **kwargs)
    
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
        self._print("=== 准备股票数据 ===")
        
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
            self._print(f"缺少文件: {missing_files}")
            raise FileNotFoundError(f"缺少必要的数据文件: {missing_files}")
        
        self._print("✓ 所有必要文件都存在")
    
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
        """加载并处理数据 - 新的标准化策略"""
        self._print("=== 加载数据 ===")
        
        # 1. 加载价格数据
        self._print("1. 加载价格数据...")
        price_data = {}
        for file in self.price_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                price_data[file.replace('.pkl', '')] = data
                self._print(f"  ✓ {file}: {data.shape}")
            else:
                self._print(f"  ✗ {file}: 加载失败")
        
        # 2. 加载因子数据（如果使用）
        factor_data = {}
        if self.use_factors:
            self._print("2. 加载因子数据...")
            for file in self.factor_files:
                data = self.load_pkl_safe(file)
                if data is not None:
                    factor_data[file.replace('.pkl', '')] = data
                    self._print(f"  ✓ {file}: {data.shape}")
                else:
                    self._print(f"  ✗ {file}: 加载失败")
        
        # 3. 加载收益率数据
        self._print("3. 加载收益率数据...")
        return_data = {}
        for file in self.return_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                return_data[file.replace('.pkl', '')] = data
                self._print(f"  ✓ {file}: {data.shape}")
            else:
                self._print(f"  ✗ {file}: 加载失败")
        
        # 4. 数据对齐和组织（不进行预标准化）
        self._print("4. 数据对齐和组织...")
        self._align_and_organize_data(price_data, factor_data, return_data)
    
    def _align_and_organize_data(self, price_data: Dict, factor_data: Dict, return_data: Dict):
        """对齐数据并组织，不进行预标准化"""
        
        # === 1. 数据对齐 ===
        self._debug_print("=== 1. 数据对齐 ===")
        
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
        self._print(f"  公共股票数量: {len(common_stocks)}")
        
        # 获取公共时间范围
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
            
            # 检查其他数据源的覆盖率
            all_available_dates = set(common_dates)
            for name, data in {**price_data, **return_data}.items():
                if data is not None:
                    available_dates = set(data.index)
                    all_available_dates = all_available_dates.intersection(available_dates)
            
            common_dates = sorted(list(all_available_dates))
        else:
            # 如果不使用因子数据，以价格数据的时间范围为准
            common_dates = None
            for name, data in {**price_data, **return_data}.items():
                if data is not None and not data.empty:
                    if common_dates is None:
                        common_dates = set(data.index)
                    else:
                        common_dates = common_dates.intersection(set(data.index))
            
            common_dates = sorted(list(common_dates))
        
        self._debug_print(f"  最终时间范围: {min(common_dates)} 到 {max(common_dates)} (共{len(common_dates)}天)")
        
        # === 2. 时间划分 ===
        self._debug_print("=== 2. 时间划分 ===")
        
        total_time = len(common_dates)
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        train_dates = common_dates[:train_end]
        val_dates = common_dates[train_end:val_end]
        test_dates = common_dates[val_end:]
        
        self._print(f"总时间步数: {total_time}")
        self._debug_print(f"训练集: {train_dates[0]} 到 {train_dates[-1]} ({len(train_dates)} 天)")
        self._debug_print(f"验证集: {val_dates[0]} 到 {val_dates[-1]} ({len(val_dates)} 天)")
        self._debug_print(f"测试集: {test_dates[0]} 到 {test_dates[-1]} ({len(test_dates)} 天)")
        
        # === 3. 组织特征数据（不标准化）===
        self._debug_print("=== 3. 组织特征数据 ===")
        
        feature_list = []
        feature_names = []
        
        # 处理价格特征（只填充，不标准化）
        for name, data in price_data.items():
            if data is not None:
                self._debug_print(f"  处理价格特征: {name}")
                # 对齐数据
                aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                # 只填充，不标准化
                filled_data = aligned_data.ffill().bfill()
                
                feature_list.append(filled_data.values)  # [T, N]
                feature_names.append(name)
        
        # 处理因子特征（如果使用，不标准化）
        if self.use_factors:
            for name, data in factor_data.items():
                if data is not None:
                    self._debug_print(f"  处理因子特征: {name}")
                    # 对齐数据
                    aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                    # 因子数据已经标准化，只填充
                    filled_data = aligned_data.ffill().bfill()
                    
                    feature_list.append(filled_data.values)  # [T, N]
                    feature_names.append(name)
        
        # === 4. 处理目标数据（截面标准化）===
        self._debug_print("=== 4. 处理目标数据 ===")
        
        target_list = []
        for horizon in self.prediction_horizons:
            return_key = f'returns_{horizon}d'
            if return_key in return_data and return_data[return_key] is not None:
                self._debug_print(f"  处理目标: {return_key}")
                # 对齐数据
                aligned_data = return_data[return_key].reindex(index=common_dates, columns=common_stocks)
                
                # 截面标准化：每天对所有股票做标准化
                if self.normalize_targets:
                    normalized_data = self._cross_section_normalize(aligned_data, train_dates)
                else:
                    # 如果不标准化，填充缺失值（目标值通常用0填充表示无收益）
                    normalized_data = aligned_data.fillna(0)
                
                target_list.append(normalized_data.values)  # [T, N]
        
        # === 5. 构建最终数据 ===
        self._debug_print("=== 5. 构建最终数据 ===")
        
        # 检查是否有有效数据
        if len(feature_list) == 0:
            raise ValueError("没有有效的特征数据")
        if len(target_list) == 0:
            raise ValueError("没有有效的目标数据")
        
        # Stack features: [T, F, N]
        self.features = torch.tensor(np.stack(feature_list, axis=1), dtype=torch.float32)
        # Stack targets: [T, N, H]
        self.targets = torch.tensor(np.stack(target_list, axis=2), dtype=torch.float32)
        
        self.feature_names = feature_names
        self.stock_names = common_stocks
        self.date_index = common_dates
        
        self._print(f"  特征矩阵形状: {self.features.shape}")
        self._print(f"  目标矩阵形状: {self.targets.shape}")
        self._debug_print(f"  特征名称: {self.feature_names}")
        self._debug_print(f"  预测期数: {self.prediction_horizons}")
        
        # 数据质量检查
        self._debug_print("=== 6. 数据质量检查 ===")
        feature_nan_count = torch.isnan(self.features).sum().item()
        target_nan_count = torch.isnan(self.targets).sum().item()
        self._debug_print(f"  特征数据NaN数量: {feature_nan_count}")
        self._debug_print(f"  目标数据NaN数量: {target_nan_count}")
        
        if feature_nan_count > 0 or target_nan_count > 0:
            self._print("  警告: 数据中仍有NaN值，可能影响训练")

    def _cross_section_normalize(self, df: pd.DataFrame, train_dates: List) -> pd.DataFrame:
        """
        截面标准化：每天对所有股票做标准化
        只使用训练集的统计量来计算标准化参数
        
        Args:
            df: 要标准化的DataFrame [时间, 股票]
            train_dates: 训练集日期列表
            
        Returns:
            标准化后的DataFrame
        """
        self._debug_print(f"    截面标准化收益率数据")
        
        # 1. 首先填充缺失值（收益率用0填充）
        filled_df = df.fillna(0)
        
        # 2. 计算训练集的截面统计量（每天的均值和标准差）
        train_data = filled_df.loc[train_dates]
        
        # 每天计算所有股票的均值和标准差
        daily_mean = train_data.mean(axis=1)  # 每天的均值
        daily_std = train_data.std(axis=1)    # 每天的标准差
        
        # 避免除零
        daily_std = daily_std.replace(0, 1.0)
        daily_std = daily_std.fillna(1.0)
        
        self._debug_print(f"      训练集日均收益率统计: 均值范围=[{daily_mean.min():.6f}, {daily_mean.max():.6f}]")
        self._debug_print(f"      训练集日收益率波动统计: 标准差范围=[{daily_std.min():.6f}, {daily_std.max():.6f}]")
        
        # 3. 对整个数据集应用截面标准化
        # 扩展训练集统计量到所有日期
        all_dates_mean = daily_mean.reindex(filled_df.index, method='ffill').fillna(daily_mean.mean())
        all_dates_std = daily_std.reindex(filled_df.index, method='ffill').fillna(daily_std.mean())
        
        # 应用标准化：(return - daily_mean) / daily_std
        normalized_df = filled_df.sub(all_dates_mean, axis=0).div(all_dates_std, axis=0)
        
        # 4. 验证标准化效果
        train_normalized = normalized_df.loc[train_dates]
        train_daily_mean = train_normalized.mean(axis=1).mean()  # 整个训练期的日均值的均值
        train_daily_std = train_normalized.std(axis=1).mean()   # 整个训练期的日标准差的均值
        
        self._debug_print(f"      标准化后训练集统计: 日均值={train_daily_mean:.6f}, 日标准差={train_daily_std:.6f}")
        
        # 检查NaN值
        nan_count = normalized_df.isna().sum().sum()
        if nan_count > 0:
            self._debug_print(f"      警告: 截面标准化后仍有 {nan_count} 个NaN值")
        
        return normalized_df

    def _split_data(self):
        """按时间划分数据集"""
        self._print("=== 划分数据集 ===")
        
        total_time = self.features.shape[0]
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        self._print(f"总时间步数: {total_time}")
        self._print(f"训练集: 0 - {train_end} ({train_end} 步)")
        self._print(f"验证集: {train_end} - {val_end} ({val_end - train_end} 步)")
        self._print(f"测试集: {val_end} - {total_time} ({total_time - val_end} 步)")
        
        # 创建数据集，传递feature_names和normalize_features参数
        self.train_dataset = StockDataset(
            features=self.features[:train_end],
            targets=self.targets[:train_end],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features
        )
        
        self.val_dataset = StockDataset(
            features=self.features[train_end:val_end],
            targets=self.targets[train_end:val_end],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features
        )
        
        self.test_dataset = StockDataset(
            features=self.features[val_end:],
            targets=self.targets[val_end:],
            feature_names=self.feature_names,
            sequence_length=self.sequence_length,
            normalize_features=self.normalize_features
        )
        
        self._print(f"训练样本数: {len(self.train_dataset)}")
        self._print(f"验证样本数: {len(self.val_dataset)}")
        self._print(f"测试样本数: {len(self.test_dataset)}")
    
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
    
    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """
        反标准化目标数据
        注意：由于采用截面标准化，反标准化需要对应的日期信息
        这里提供简化版本，实际使用时可能需要更复杂的逻辑
        
        Args:
            normalized_targets: 标准化的目标数据 [..., H]
            
        Returns:
            原始尺度的目标数据（截面标准化情况下较复杂）
        """
        # 截面标准化的反标准化较为复杂，需要知道具体的日期
        # 这里返回原值，实际应用中可能需要更精细的处理
        return normalized_targets

    def get_normalization_stats(self):
        """获取标准化统计信息"""
        return {
            'normalization_method': self.normalization_method,
            'normalize_features': self.normalize_features,
            'normalize_targets': self.normalize_targets,
            'price_features': self.price_features,
            'volume_features': self.volume_features
        }

# 使用示例
def main():
    """使用示例 - 新的标准化策略"""
    
    # 示例1：只使用价格数据，序列标准化
    print("=== 示例1：只使用价格数据，序列标准化 ===")
    price_datamodule = StockDataModule(
        use_factors=False,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=32,
        normalize_features=True,  # 启用序列级价格标准化
        normalize_targets=True,   # 启用截面标准化
        normalization_method='sequence_price_cross_section_return',
        debug=True  # 开启调试输出
    )
    
    price_datamodule.prepare_data()
    price_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = price_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    features, targets = batch
    
    print(f"特征形状: {features.shape}")  # [B, L, F, N]
    print(f"目标形状: {targets.shape}")   # [B, H, N]
    print(f"特征维度: {price_datamodule.get_feature_dim()}")
    print(f"股票数量: {price_datamodule.get_stock_num()}")
    
    # 检查标准化效果
    print(f"\n价格特征标准化检查:")
    if price_datamodule.train_dataset.close_feature_index is not None:
        close_idx = price_datamodule.train_dataset.close_feature_index
        print(f"Close价格特征索引: {close_idx}")
        # 检查最后一天的close价格是否接近1.0（相对标准化）
        last_day_close = features[15, -1, close_idx, :]  # 第一个样本，最后一天，close特征，所有股票
        print(f"标准化后最后一天close价格统计: 均值={last_day_close.mean():.4f}, 标准差={last_day_close.std():.4f}")
    
    # 示例2：使用因子数据和价格数据
    print("\n=== 示例2：使用因子+价格数据，新标准化策略 ===")
    factor_datamodule = StockDataModule(
        use_factors=True,
        sequence_length=20,
        prediction_horizons=[1, 5, 10],
        batch_size=32,
        normalize_features=True,  # 序列级价格标准化，因子不标准化
        normalize_targets=True,   # 截面标准化
        normalization_method='sequence_price_cross_section_return',
        debug=True  # 关闭调试输出
    )
    
    factor_datamodule.prepare_data()
    factor_datamodule.setup()
    
    # 获取一个批次的数据
    train_loader = factor_datamodule.train_dataloader()
    batch = next(iter(train_loader))
    features, targets = batch
    
    print(f"特征形状: {features.shape}")  # [B, L, F, N]
    print(f"目标形状: {targets.shape}")   # [B, H, N]
    print(f"特征维度: {factor_datamodule.get_feature_dim()}")
    print(f"股票数量: {factor_datamodule.get_stock_num()}")
    
    # 检查目标值的截面标准化效果
    print(f"\n收益率截面标准化检查:")
    # targets的形状是 [B, H, N]，检查每个horizon每天的均值和标准差
    for h in range(targets.shape[1]):
        horizon_targets = targets[:, h, :]  # [B, N]
        daily_means = horizon_targets.mean(dim=1)  # [B] - 每个样本（日期）的股票均值
        daily_stds = horizon_targets.std(dim=1)    # [B] - 每个样本（日期）的股票标准差
        print(f"  Horizon {h+1}: 日均值范围=[{daily_means.min():.4f}, {daily_means.max():.4f}], "
              f"日标准差范围=[{daily_stds.min():.4f}, {daily_stds.max():.4f}]")

if __name__ == "__main__":
    main()
