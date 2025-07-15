# 新的标准化策略说明

## 概述

根据您的需求，我们实现了一套新的、更适合金融数据的标准化策略，替代了原有的全局Z-score标准化方法。

## 主要变更

### 1. 价格特征标准化（序列级别）
- **策略**：对每个股票的每个序列，使用最后一天的close价格作为基准
- **适用特征**：`open_adj`, `high_adj`, `low_adj`, `close_adj`, `vwap_adj`
- **公式**：`normalized_price = price / last_close`
- **优势**：
  - 消除不同股票价格水平的差异
  - 价格变成相对比率，更有意义
  - 最后一天的close价格约等于1.0，便于模型理解

### 2. 成交量特征标准化（序列级别）
- **策略**：对每个股票的每个序列，使用最后一天的成交量作为基准
- **适用特征**：`volume_adj`, `turnover`
- **公式**：`normalized_volume = volume / last_volume`
- **优势**：
  - 消除不同股票成交量规模的差异
  - 体现相对成交量水平
  - 最后一天的成交量约等于1.0

### 3. 因子特征（不标准化）
- **策略**：保持原值，不进行额外标准化
- **适用特征**：所有factor数据（momentum, beta, size等）
- **原因**：因子数据通常已经经过预处理和标准化

### 4. 收益率标准化（截面标准化）
- **策略**：每天对所有股票的收益率进行标准化
- **公式**：`normalized_return = (return - daily_mean) / daily_std`
- **优势**：
  - 消除市场整体趋势的影响
  - 突出个股相对表现
  - 每天的收益率分布接近标准正态分布

## 代码结构变更

### 新增类和方法

1. **StockDataset._normalize_sequence_features()**
   - 实现序列级别的价格和成交量标准化
   - 在`__getitem__`时动态执行

2. **StockDataModule._cross_section_normalize()**
   - 实现截面标准化逻辑
   - 只使用训练集统计量

3. **特征分类逻辑**
   - `price_features`: 价格相关特征列表
   - `volume_features`: 成交量相关特征列表
   - 自动识别不同类型特征的索引

### 参数变更

- `normalization_method`: 默认为`'sequence_price_cross_section_return'`
- `debug`: 新增调试模式，控制详细输出
- `normalize_features`: 控制是否启用序列级标准化
- `normalize_targets`: 控制是否启用截面标准化

## 使用方法

### 基本使用

```python
# 创建数据模块
datamodule = StockDataModule(
    data_dir='/path/to/data',
    use_factors=True,
    sequence_length=20,
    prediction_horizons=[1, 5, 10, 20],
    normalize_features=True,   # 启用序列级标准化
    normalize_targets=True,    # 启用截面标准化
    normalization_method='sequence_price_cross_section_return',
    debug=True  # 开启调试输出
)

# 准备和设置数据
datamodule.prepare_data()
datamodule.setup()

# 获取数据加载器
train_loader = datamodule.train_dataloader()
```

### 标准化效果验证

```python
# 获取一个批次检查标准化效果
batch = next(iter(train_loader))
features, targets = batch  # [B, L, F, N], [B, H, N]

# 检查价格标准化：最后一天的close应该接近1.0
close_idx = datamodule.train_dataset.close_feature_index
last_day_close = features[0, -1, close_idx, :]
print(f"最后一天close均值: {last_day_close.mean():.4f}")

# 检查收益率截面标准化：每天的均值应该接近0，标准差接近1
for h in range(targets.shape[1]):
    horizon_targets = targets[:, h, :]
    daily_means = horizon_targets.mean(dim=1)
    daily_stds = horizon_targets.std(dim=1)
    print(f"Horizon {h+1}: 日均值范围={daily_means.min():.4f}~{daily_means.max():.4f}")
```

## 关键优势

1. **金融意义明确**：
   - 价格标准化反映相对价格水平
   - 成交量标准化反映相对活跃度
   - 收益率截面标准化突出相对表现

2. **避免前瞻偏差**：
   - 序列级标准化只使用当前序列信息
   - 截面标准化只使用训练集统计量

3. **提升模型性能**：
   - 不同股票特征在相同量级
   - 消除无关的规模效应
   - 突出真正的信号

4. **兼容性好**：
   - 保持原有API接口
   - 支持不同的特征组合
   - 可配置的标准化选项

## 注意事项

1. **数据质量**：确保close价格和成交量数据无异常值
2. **除零保护**：代码中已包含除零检查
3. **调试模式**：首次使用建议开启debug=True查看详细信息
4. **内存使用**：序列级标准化在运行时计算，不增加存储开销

## 与原有方法的对比

| 方面 | 原有方法 | 新方法 |
|------|----------|--------|
| 价格标准化 | 全局Z-score | 序列内相对价格 |
| 成交量标准化 | 全局Z-score | 序列内相对成交量 |
| 因子处理 | 可选标准化 | 保持原值 |
| 收益率处理 | 全局Z-score | 截面标准化 |
| 金融意义 | 一般 | 强 |
| 前瞻偏差 | 可能存在 | 避免 |

这套新的标准化策略更符合量化金融的实践，能够更好地服务于股票风险因子挖掘任务。
