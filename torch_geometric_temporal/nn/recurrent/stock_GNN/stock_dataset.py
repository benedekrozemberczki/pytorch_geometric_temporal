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
                 normalization_method: str = 'zscore',
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
            normalization_method: 标准化方法 ('zscore', 'minmax')
            debug: 是否开启调试模式（打印详细信息）
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
        """加载并处理数据"""
        self._print("=== 加载数据 ===")
        
        # 1. 加载价格数据
        self._debug_print("1. 加载价格数据...")
        price_data = {}
        for file in self.price_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                price_data[file.replace('.pkl', '')] = data
                self._debug_print(f"  ✓ {file}: {data.shape}")
            else:
                self._debug_print(f"  ✗ {file}: 加载失败")
        
        # 2. 加载因子数据（如果使用）
        factor_data = {}
        if self.use_factors:
            self._debug_print("2. 加载因子数据...")
            for file in self.factor_files:
                data = self.load_pkl_safe(file)
                if data is not None:
                    factor_data[file.replace('.pkl', '')] = data
                    self._debug_print(f"  ✓ {file}: {data.shape}")
                else:
                    self._debug_print(f"  ✗ {file}: 加载失败")
        
        # 3. 加载收益率数据
        self._debug_print("3. 加载收益率数据...")
        return_data = {}
        for file in self.return_files:
            data = self.load_pkl_safe(file)
            if data is not None:
                return_data[file.replace('.pkl', '')] = data
                self._debug_print(f"  ✓ {file}: {data.shape}")
            else:
                self._debug_print(f"  ✗ {file}: 加载失败")
        
        # 4. 数据对齐、划分和标准化
        self._debug_print("4. 数据对齐、时间划分和标准化...")
        self._align_split_and_normalize_data(price_data, factor_data, return_data)
    
    def _align_split_and_normalize_data(self, price_data: Dict, factor_data: Dict, return_data: Dict):
        """对齐数据，划分时间序列，并进行标准化"""
        
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
        
        # === 3. 处理特征数据 ===
        self._debug_print("=== 3. 处理特征数据 ===")
        
        feature_list = []
        feature_names = []
        
        # 处理价格特征
        for name, data in price_data.items():
            if data is not None:
                self._debug_print(f"  处理价格特征: {name}")
                # 对齐数据
                aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                
                # 按您的需求进行标准化处理（内部包含填充操作）
                if self.normalize_features:
                    normalized_data = self._normalize_single_feature(
                        aligned_data, train_dates, val_dates, test_dates, name
                    )
                else:
                    # 如果不标准化，仍需填充缺失值
                    normalized_data = aligned_data.ffill().bfill()
                
                feature_list.append(normalized_data.values)  # [T, N]
                feature_names.append(name)
        
        # 处理因子特征（如果使用）
        if self.use_factors:
            for name, data in factor_data.items():
                if data is not None:
                    self._debug_print(f"  处理因子特征: {name}")
                    # 对齐数据
                    aligned_data = data.reindex(index=common_dates, columns=common_stocks)
                    
                    # 按您的需求进行标准化处理（内部包含填充操作）
                    if self.normalize_features:
                        normalized_data = self._normalize_single_feature(
                            aligned_data, train_dates, val_dates, test_dates, name
                        )
                    else:
                        # 如果不标准化，仍需填充缺失值
                        normalized_data = aligned_data.ffill().bfill()
                    
                    feature_list.append(normalized_data.values)  # [T, N]
                    feature_names.append(name)
        
        # === 4. 处理目标数据 ===
        self._debug_print("=== 4. 处理目标数据 ===")
        
        target_list = []
        for horizon in self.prediction_horizons:
            return_key = f'returns_{horizon}d'
            if return_key in return_data and return_data[return_key] is not None:
                self._debug_print(f"  处理目标: {return_key}")
                # 对齐数据
                aligned_data = return_data[return_key].reindex(index=common_dates, columns=common_stocks)
                
                # 按您的需求进行标准化处理
                if self.normalize_targets:
                    # 对目标值也应用相同的标准化逻辑
                    normalized_data = self._normalize_single_feature(
                        aligned_data, train_dates, val_dates, test_dates, return_key, is_target=True
                    )
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

    def _normalize_single_feature(self, df: pd.DataFrame, train_dates: List, 
                                 val_dates: List, test_dates: List, feature_name: str, 
                                 is_target: bool = False) -> pd.DataFrame:
        """
        按照您的需求对单个特征进行标准化
        df_zscore = ((df.ffill().bfill()) - df.values.mean()) / df.values.std()
        只使用训练集的mean和std
        
        Args:
            df: 要标准化的DataFrame
            train_dates: 训练集日期
            val_dates: 验证集日期  
            test_dates: 测试集日期
            feature_name: 特征名称
            is_target: 是否为目标变量（影响填充策略）
        """
        self._debug_print(f"    标准化{'目标' if is_target else '特征'}: {feature_name}")
        
        # 1. 对DataFrame进行填充
        if is_target:
            # 目标值用0填充（表示无收益）
            filled_df = df.fillna(0)
        else:
            # 特征用前向后向填充
            filled_df = df.ffill().bfill()
        
        self._debug_print(f"      填充前形状: {df.shape}, 填充后形状: {filled_df.shape}")
        
        # 2. 分割训练集数据以计算统计量
        train_data = filled_df.loc[train_dates]
        
        # 3. 只用训练集计算统计量
        train_mean = train_data.values.mean()
        train_std = train_data.values.std()
        
        self._debug_print(f"      训练集原始统计: 均值={train_mean:.6f}, 标准差={train_std:.6f}")
        
        # 4. 避免除零
        if train_std == 0 or np.isnan(train_std):
            self._debug_print(f"      警告: {feature_name} 的训练集标准差为0或NaN，使用1.0代替")
            train_std = 1.0
        
        # 5. 对填充后的整个数据集应用训练集的统计量进行标准化
        # df_zscore = ((df.ffill().bfill()) - train_mean) / train_std
        normalized_df = (filled_df - train_mean) / train_std
        
        # 6. 验证标准化效果
        train_normalized = normalized_df.loc[train_dates]
        val_normalized = normalized_df.loc[val_dates] if len(val_dates) > 0 else None
        test_normalized = normalized_df.loc[test_dates] if len(test_dates) > 0 else None
        
        # 训练集应该接近标准正态分布
        train_norm_mean = train_normalized.values.mean()
        train_norm_std = train_normalized.values.std()
        self._debug_print(f"      标准化后训练集: 均值={train_norm_mean:.6f}, 标准差={train_norm_std:.6f}")
        
        # 验证集和测试集的统计量
        if val_normalized is not None:
            val_norm_mean = val_normalized.values.mean()
            val_norm_std = val_normalized.values.std()
            self._debug_print(f"      标准化后验证集: 均值={val_norm_mean:.6f}, 标准差={val_norm_std:.6f}")
        
        if test_normalized is not None:
            test_norm_mean = test_normalized.values.mean()
            test_norm_std = test_normalized.values.std()
            self._debug_print(f"      标准化后测试集: 均值={test_norm_mean:.6f}, 标准差={test_norm_std:.6f}")
        
        # 7. 检查是否还有NaN值
        nan_count = normalized_df.isna().sum().sum()
        if nan_count > 0:
            self._debug_print(f"      警告: 标准化后仍有 {nan_count} 个NaN值")
        
        return normalized_df
    
    def _split_data(self):
        """按时间划分数据集"""
        self._debug_print("=== 划分数据集 ===")
        
        total_time = self.features.shape[0]
        train_size = int(total_time * self.train_ratio)
        val_size = int(total_time * self.val_ratio)
        
        train_end = train_size
        val_end = train_size + val_size
        
        self._debug_print(f"总时间步数: {total_time}")
        self._debug_print(f"训练集: 0 - {train_end} ({train_end} 步)")
        self._debug_print(f"验证集: {train_end} - {val_end} ({val_end - train_end} 步)")
        self._debug_print(f"测试集: {val_end} - {total_time} ({total_time - val_end} 步)")
        
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
        
        self._print(f"训练样本数: {len(self.train_dataset)}")
        self._print(f"验证样本数: {len(self.val_dataset)}")
        self._print(f"测试样本数: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 添加：提高DDP性能
            drop_last=True,           # 添加：确保DDP中batch大小一致
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 添加：提高DDP性能
            drop_last=False,          # 验证集不丢弃最后一个batch
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  # 添加：提高DDP性能
            drop_last=False,          # 测试集不丢弃最后一个batch
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
    
    # ===== 旧版标准化方法（已废弃） =====
    # def _normalize_data(self, data: np.ndarray, stats: Optional[Dict] = None, fit: bool = True) -> Tuple[np.ndarray, Dict]:
    #     """
    #     标准化数据 - 按特征在所有股票上进行标准化 (DEPRECATED)
    #     使用新的 _normalize_single_feature 方法替代
    #     """
    #     pass

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
