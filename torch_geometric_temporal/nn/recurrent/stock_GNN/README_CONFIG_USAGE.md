# 新标准化策略配置使用指南

## 概述

我们已经更新了所有相关的Hydra配置文件和训练脚本，以支持新的标准化策略。以下是详细的使用指南。

## 新增/更新的配置文件

### 数据配置文件 (config/data/)

1. **stock_data.yaml** - 基础配置，已更新为使用新标准化策略
2. **advanced_norm.yaml** - 专门为新标准化策略优化的配置
3. **debug_norm_validation.yaml** - 用于验证新标准化策略的调试配置
4. **debug.yaml** - 更新了调试配置以支持新参数

### 实验配置文件 (config/experiment/)

1. **production.yaml** - 生产环境配置，使用新标准化策略
2. **gat.yaml** - GAT模型配置，使用新标准化策略
3. **norm_validation.yaml** - 新增：专门用于验证标准化策略的实验

## 使用方法

### 1. 验证新标准化策略

首先验证新的标准化策略是否正常工作：

```bash
# 使用专门的验证配置
python train_stock_gnn_hydra.py experiment=norm_validation

# 或者使用调试数据配置
python train_stock_gnn_hydra.py data=debug_norm_validation
```

这将：
- 启用详细的标准化日志输出
- 使用小数据集进行快速验证
- 打印标准化统计信息
- 验证价格特征和收益率的标准化效果

### 2. 生产环境训练

使用新标准化策略进行生产环境训练：

```bash
# GCN + 新标准化策略
python train_stock_gnn_hydra.py experiment=production

# GAT + 新标准化策略
python train_stock_gnn_hydra.py experiment=gat
```

### 3. 自定义配置

#### 使用高级标准化配置

```bash
# 使用优化的标准化配置
python train_stock_gnn_hydra.py data=advanced_norm

# 启用调试日志
python train_stock_gnn_hydra.py data=advanced_norm data.debug=true
```

#### 覆盖特定参数

```bash
# 修改标准化方法（虽然新策略是固定的）
python train_stock_gnn_hydra.py data.normalization_method="sequence_price_cross_section_return"

# 启用/禁用标准化
python train_stock_gnn_hydra.py data.normalize_features=true data.normalize_targets=true

# 启用调试模式
python train_stock_gnn_hydra.py data.debug=true
```

## 新参数说明

### 数据配置新参数

```yaml
# 标准化策略
normalization_method: "sequence_price_cross_section_return"  # 新的标准化方法
normalize_features: true   # 启用序列级价格/成交量标准化
normalize_targets: true    # 启用截面收益率标准化
debug: false              # 启用详细标准化日志
```

### 标准化策略详解

1. **价格特征标准化** (`normalize_features=true`):
   - `open_adj`, `high_adj`, `low_adj`, `close_adj`, `vwap_adj`
   - 每个序列内，用最后一天的`close_adj`进行标准化
   - 公式：`normalized_price = price / last_close`

2. **成交量特征标准化** (`normalize_features=true`):
   - `volume_adj`, `turnover`
   - 每个序列内，用最后一天的对应成交量进行标准化
   - 公式：`normalized_volume = volume / last_volume`

3. **因子特征** (自动处理):
   - 保持原值，不进行额外标准化
   - 因为已经是预处理过的标准化数据

4. **收益率标准化** (`normalize_targets=true`):
   - 截面标准化：每天对所有股票进行标准化
   - 公式：`normalized_return = (return - daily_mean) / daily_std`

## 验证标准化效果

### 检查价格标准化

运行验证后，检查输出：

```
价格特征标准化检查:
Close价格特征索引: 3
标准化后最后一天close价格统计: 均值=1.0000, 标准差=0.0000
```

最后一天的close价格应该均值为1.0，标准差为0.0（因为用自己标准化）。

### 检查收益率截面标准化

```
收益率截面标准化检查:
  Horizon 1: 日均值范围=[-0.0001, 0.0001], 日标准差范围=[0.9980, 1.0020]
```

每天的收益率均值应该接近0，标准差应该接近1。

## 调试和故障排除

### 启用详细日志

```bash
# 启用数据加载的详细日志
python train_stock_gnn_hydra.py data.debug=true

# 使用调试配置
python train_stock_gnn_hydra.py data=debug_norm_validation
```

### 常见问题

1. **除零错误**：已在代码中处理，无效值会保持原值
2. **内存使用**：序列级标准化在运行时计算，不会增加存储开销
3. **性能影响**：标准化在数据加载时进行，对训练速度影响很小

## 与旧版本的兼容性

- 旧的配置文件仍然可用，但建议使用新的配置
- 如果使用旧的`normalization_method="zscore"`，将使用旧的标准化逻辑
- 新的标准化策略通过`normalization_method="sequence_price_cross_section_return"`启用

## 推荐配置

### 快速验证

```bash
python train_stock_gnn_hydra.py experiment=norm_validation
```

### 生产训练

```bash
# GCN
python train_stock_gnn_hydra.py experiment=production

# GAT
python train_stock_gnn_hydra.py experiment=gat
```

### 开发调试

```bash
python train_stock_gnn_hydra.py data=debug_norm_validation trainer=cpu_debug
```

这样，您就可以充分利用新的标准化策略来提升模型的金融意义和性能了！
