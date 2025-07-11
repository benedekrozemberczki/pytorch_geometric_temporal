# Converting from GCN to GAT Layers

## Summary of Changes Made

To convert from GCN (Graph Convolutional Network) to GAT (Graph Attention Network) layers, the following changes were implemented:

### 1. Import Changes
```python
# Before
from torch_geometric.nn import GCNConv

# After  
from torch_geometric.nn import GCNConv, GATConv
```

### 2. Model Parameters Added
```python
def __init__(
    # ... existing parameters ...
    gnn_type: str = "gcn",  # Options: "gcn", "gat"
    gat_heads: int = 4,     # Number of attention heads for GAT
    gat_dropout: float = 0.1, # Dropout for GAT attention
):
```

### 3. Layer Construction Logic
```python
if self.gnn_type == "gat":
    # GAT layers with multi-head attention
    self.gnn1 = GATConv(
        gru_hidden_dim, 
        gnn_hidden_dim // gat_heads,  # Output dim per head
        heads=gat_heads,
        dropout=gat_dropout,
        add_self_loops=add_self_loops,
        concat=True  # Concatenate multi-head outputs
    )
    self.gnn2 = GATConv(
        gnn_hidden_dim,  # Input is concatenated output from gnn1
        gnn_hidden_dim // gat_heads,
        heads=gat_heads,
        dropout=gat_dropout,
        add_self_loops=add_self_loops,
        concat=False  # Average multi-head outputs for final layer
    )
else:
    # Default GCN layers
    self.gnn1 = GCNConv(gru_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
    self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim, add_self_loops=add_self_loops)
```

### 4. Forward Pass Changes
```python
# Add self loops only for GCN (GAT handles them internally)
if self.gnn_type == "gcn" and self.add_self_loops:
    e_idx, e_w = add_self_loops(...)

# Different forward pass for each GNN type
if self.gnn_type == "gat":
    # GAT doesn't use edge weights directly in the same way
    out1 = F.relu(self.gnn1(x_all, e_idx))
    out2 = F.relu(self.gnn2(out1, e_idx))
else:
    # GCN uses edge weights
    out1 = F.relu(self.gnn1(x_all, e_idx, edge_weight=e_w))
    out2 = F.relu(self.gnn2(out1, e_idx, edge_weight=e_w))
```

### 5. Configuration Files

#### Model Config (`config/model/gat.yaml`)
```yaml
gnn_type: "gat"
gat_heads: 4
gat_dropout: 0.1
add_self_loops: false  # GAT handles self-loops internally
```

#### Experiment Config (`config/experiment/gat.yaml`)
```yaml
defaults:
  - override /model: gat

model:
  gat_heads: 8
  gat_dropout: 0.2
  k_nn: 12  # More neighbors since GAT can handle attention weighting
```

## Key Differences: GCN vs GAT

### GCN Characteristics:
- **Fixed aggregation**: Simple mean/sum of neighbor features
- **Edge weights**: Uses explicit edge weights for aggregation
- **Computational efficiency**: Faster, less memory intensive
- **Homogeneous treatment**: All neighbors treated equally

### GAT Characteristics:
- **Attention mechanism**: Learns importance of each neighbor dynamically
- **Self-attention**: Computes attention weights based on node features
- **Multi-head attention**: Multiple attention mechanisms in parallel
- **Adaptive**: Can focus on most relevant neighbors for each node

## Usage Examples

### Train with GCN (default)
```bash
python train_stock_gnn_hydra.py experiment=production
```

### Train with GAT
```bash
python train_stock_gnn_hydra.py experiment=gat
```

### Override GNN type on command line
```bash
# Use GAT with 8 heads
python train_stock_gnn_hydra.py model.gnn_type=gat model.gat_heads=8

# Use GCN
python train_stock_gnn_hydra.py model.gnn_type=gcn
```

## GAT Benefits for Stock Data

1. **Dynamic Relationships**: Stock relationships change over time; attention can adapt
2. **Heterogeneous Features**: Different stocks have different characteristics; attention handles this well
3. **Market Regime Changes**: Attention can focus on different stocks during different market conditions
4. **Interpretability**: Attention weights show which stocks influence each other

## Performance Considerations

### GAT:
- **Memory**: Higher memory usage due to attention computation
- **Speed**: Slower than GCN due to attention mechanisms
- **Batch Size**: May need smaller batches
- **Precision**: Consider using FP32 for attention stability

### Recommended Settings for GAT:
- Batch size: 16-32 (vs 64+ for GCN)
- Learning rate: 1e-4 to 5e-4 (slightly lower)
- Gradient clipping: 0.5 (tighter clipping)
- Attention heads: 4-8 for most cases
- Dropout: 0.1-0.2 for attention regularization

## Monitoring GAT Training

Watch for:
1. **Attention collapse**: All attention weights become uniform
2. **Gradient explosion**: Attention can be unstable
3. **Memory usage**: Monitor GPU memory consumption
4. **Convergence**: GAT may take longer to converge

The implementation provides flexibility to switch between GCN and GAT easily through configuration, allowing you to experiment and compare both approaches for your stock prediction task.
