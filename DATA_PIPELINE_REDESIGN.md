# 股票预测数据标准化流程重新设计

## 概述

本次重新设计股票预测数据管道的标准化逻辑，主要目标是：

1. **数据读取后立即标准化**：使用公式 `df_zscore = ((df.ffill().bfill()) - df.values.mean()) / df.values.std()`
2. **尽早进行时间分割**：在任何处理之前先split train/val/test
3. **只使用训练集统计**：确保所有split（train/val/test）都使用训练集的mean和std进行标准化

## 新数据处理流程

### 1. 数据加载阶段 (`_load_and_process_data`)
```python
# 1. 加载价格数据 (price_data)
# 2. 加载因子数据 (factor_data, 如果使用)
# 3. 加载收益率数据 (return_data)
# 4. 调用 _align_split_and_normalize_data
```

### 2. 数据对齐、分割和标准化 (`_align_split_and_normalize_data`)

#### 2.1 数据对齐
- 找到所有数据源的公共股票
- 找到所有数据源的公共时间范围
- 打印对齐后的股票数量和时间范围

#### 2.2 时间分割
```python
# 根据 train_ratio, val_ratio 计算分割点
train_dates = common_dates[:train_end]
val_dates = common_dates[train_end:val_end] 
test_dates = common_dates[val_end:]
```

#### 2.3 特征处理
对每个特征DataFrame：
```python
# 1. 对齐到公共股票和时间
aligned_data = data.reindex(index=common_dates, columns=common_stocks)

# 2. 标准化处理（内部包含填充）
if normalize_features:
    normalized_data = _normalize_single_feature(
        aligned_data, train_dates, val_dates, test_dates, feature_name
    )
else:
    normalized_data = aligned_data.ffill().bfill()

# 3. 添加到特征列表
feature_list.append(normalized_data.values)
```

#### 2.4 目标处理
类似特征处理，但使用不同的填充策略：
```python
if normalize_targets:
    normalized_data = _normalize_single_feature(
        aligned_data, train_dates, val_dates, test_dates, 
        return_key, is_target=True
    )
else:
    normalized_data = aligned_data.fillna(0)  # 目标用0填充
```

### 3. 单特征标准化 (`_normalize_single_feature`)

#### 核心逻辑：
```python
# 1. 填充策略
if is_target:
    filled_df = df.fillna(0)  # 目标：0填充
else:
    filled_df = df.ffill().bfill()  # 特征：前向后向填充

# 2. 只用训练集计算统计量
train_data = filled_df.loc[train_dates]
train_mean = train_data.values.mean()
train_std = train_data.values.std()

# 3. 应用标准化（按您的公式）
normalized_df = (filled_df - train_mean) / train_std
```

#### 验证输出：
- 训练集标准化后应该均值≈0，标准差≈1
- 验证集和测试集的统计量会不同（这是正确的）
- 详细打印每个阶段的统计信息

## 关键优势

### 1. 遵循机器学习最佳实践
- ✅ 只使用训练集统计量
- ✅ 防止数据泄露
- ✅ 时间序列分割正确

### 2. 符合您的具体需求
- ✅ 实现指定的标准化公式
- ✅ 数据读取后立即标准化
- ✅ 特征和目标使用不同填充策略

### 3. 代码质量提升
- ✅ 清晰的方法分离
- ✅ 详细的调试输出
- ✅ 完整的数据质量检查
- ✅ 移除冗余代码

## 数据流示意图

```
原始数据 → 对齐 → 时间分割 → 标准化 → 张量构建
  |         |        |         |         |
价格/因子   公共     train/    只用训练   features:
收益率     股票      val/      集统计     [T,F,N]
数据      时间      test       量        targets:
                                        [T,N,H]
```

## 配置参数

- `normalize_features`: 是否标准化特征
- `normalize_targets`: 是否标准化目标
- `train_ratio`: 训练集比例
- `val_ratio`: 验证集比例

## 调试信息

新流程提供详细的调试输出：
- 每个数据源的加载状态和形状
- 数据对齐后的股票和时间信息
- 时间分割的详细信息
- 每个特征的标准化前后统计量
- 最终数据张量的形状和质量检查

## 废弃的方法

- `_normalize_data`: 旧版本的标准化方法，已注释掉
- `_align_and_process_data`: 被新的`_align_split_and_normalize_data`替代
