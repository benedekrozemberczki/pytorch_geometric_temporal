# Stock GNN Training Pipeline with Hydra Configuration

This directory contains a PyTorch Lightning-based training pipeline for dynamic GNN models on stock data, now enhanced with Hydra for hierarchical configuration management.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Hydra and related packages
pip install -r requirements_hydra.txt

# Install other dependencies (PyTorch, PyTorch Geometric, PyTorch Lightning, etc.)
```

### 2. Basic Training
```bash
# Run with default configuration
python train_stock_gnn_hydra.py

# Use different configurations
python train_stock_gnn_hydra.py experiment=debug
python train_stock_gnn_hydra.py experiment=production
```

### 3. Override Configuration Parameters
```bash
# Override specific parameters
python train_stock_gnn_hydra.py model.lr=1e-4 trainer.max_epochs=100

# Use different trainer/data configurations
python train_stock_gnn_hydra.py trainer=single_gpu data=debug

# Override multiple parameters
python train_stock_gnn_hydra.py \
  model.lr=5e-4 \
  model.gru_hidden_dim=128 \
  data.batch_size=64 \
  trainer.max_epochs=200
```

## 📁 Configuration Structure

```
config/
├── config.yaml              # Main configuration file
├── data/                     # Data module configurations
│   ├── stock_data.yaml      # Default stock data settings
│   └── debug.yaml           # Debug data settings
├── model/                    # Model configurations
│   └── dynamic_graph.yaml   # Dynamic graph model settings
├── loss/                     # Loss function configurations
│   └── accumulative_gain.yaml
├── trainer/                  # PyTorch Lightning trainer configurations
│   ├── ddp_gpu.yaml         # Multi-GPU DDP training
│   ├── single_gpu.yaml      # Single GPU training
│   └── cpu_debug.yaml       # CPU debugging
├── callbacks/                # Training callbacks
│   └── default.yaml         # ModelCheckpoint, EarlyStopping, etc.
├── logger/                   # Logging configurations
│   ├── tensorboard.yaml     # TensorBoard logger
│   └── csv.yaml             # CSV logger
└── experiment/               # Pre-configured experiments
    ├── debug.yaml           # Quick debugging setup
    ├── production.yaml      # Full production training
    └── sweep.yaml           # Hyperparameter sweeping
```

## 🎯 Usage Examples

### Debug Mode
```bash
# Quick debugging with small dataset and CPU
python train_stock_gnn_hydra.py experiment=debug

# Debug with GPU but small model
python train_stock_gnn_hydra.py experiment=debug trainer=single_gpu
```

### Production Training
```bash
# Full production training with DDP
python train_stock_gnn_hydra.py experiment=production

# Production with custom learning rate
python train_stock_gnn_hydra.py experiment=production model.lr=1e-4
```

### Single GPU Training
```bash
# Train on single GPU
python train_stock_gnn_hydra.py trainer=single_gpu

# Single GPU with larger model
python train_stock_gnn_hydra.py \
  trainer=single_gpu \
  model.gru_hidden_dim=256 \
  model.gnn_hidden_dim=512
```

### Custom Data Paths
```bash
# Use different data directory
python train_stock_gnn_hydra.py data.data_dir=/path/to/your/data

# Use different prediction horizons
python train_stock_gnn_hydra.py data.prediction_horizons=[1,5,20]
```

## 📊 Hyperparameter Sweeping

### Basic Sweep
```bash
# Run hyperparameter sweep
python train_stock_gnn_hydra.py -m experiment=sweep

# Custom sweep parameters
python train_stock_gnn_hydra.py -m \
  model.lr=1e-4,5e-4,1e-3 \
  model.gru_hidden_dim=32,64,128 \
  trainer.max_epochs=50
```

### Advanced Sweeping with Optuna
```bash
# Install Optuna plugin
pip install hydra-optuna-sweeper

# Configure Optuna in your config
python train_stock_gnn_hydra.py -m \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=20 \
  hydra.sweeper.sampler.n_startup_trials=5
```

## ⚙️ Configuration Details

### Data Configuration
- `data_dir`: Path to stock data
- `use_factors`: Whether to use factor data
- `sequence_length`: Input sequence length
- `prediction_horizons`: List of prediction time horizons
- `batch_size`: Training batch size
- `normalize_features/targets`: Normalization settings

### Model Configuration
- `gru_hidden_dim`: GRU hidden dimension
- `gnn_hidden_dim`: GNN hidden dimension
- `k_nn`: Number of nearest neighbors for graph construction
- `lr`: Learning rate
- `add_self_loops`: Whether to add self loops in GNN

### Trainer Configuration
- `max_epochs`: Maximum training epochs
- `accelerator`: "gpu", "cpu", or "auto"
- `devices`: Number of devices to use
- `strategy`: Training strategy ("ddp", "auto", etc.)
- `precision`: Training precision ("32-true", "16-mixed", etc.)

### Loss Configuration
- `value_decay`: Decay factor for accumulative gain
- `penalty_weight`: Weight for penalty terms
- `eps`: Small epsilon for numerical stability

## 🔧 Advanced Features

### Custom Callbacks
Add custom callbacks in `config/callbacks/`:
```yaml
# config/callbacks/custom.yaml
my_callback:
  _target_: path.to.MyCallback
  param1: value1
  param2: value2
```

### Environment Variables
Use environment variables in configs:
```yaml
data:
  data_dir: ${oc.env:STOCK_DATA_DIR,/default/path}
```

### Conditional Configuration
```yaml
# Use different settings based on conditions
model:
  lr: ${oc.select:trainer.accelerator,"gpu":1e-3,"cpu":1e-4}
```

### Output Directory Structure
```
outputs/
└── stock_gnn_factor_mining/
    └── 2024-01-15_10-30-45/    # Timestamp-based run directory
        ├── .hydra/
        │   ├── config.yaml      # Resolved configuration
        │   └── overrides.yaml   # Applied overrides
        ├── logs/                # Training logs
        ├── checkpoints/         # Model checkpoints
        └── config.yaml          # Final saved configuration
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure Hydra is installed
   ```bash
   pip install hydra-core omegaconf
   ```

2. **Configuration Not Found**: Check file paths and spelling
   ```bash
   # Check available configurations
   python train_stock_gnn_hydra.py --help
   ```

3. **Multi-GPU Issues**: Ensure proper DDP configuration
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.device_count())"
   ```

4. **Memory Issues**: Reduce batch size or model size
   ```bash
   python train_stock_gnn_hydra.py \
     data.batch_size=16 \
     model.gru_hidden_dim=32
   ```

### Debug Tips
- Use `experiment=debug` for quick testing
- Enable verbose logging with `debug.enabled=true`
- Use `trainer.fast_dev_run=true` for very quick testing
- Check logs in the output directory

## 📚 Additional Resources

- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)

## 🔄 Migration from Old Script

The old `train_stock_gnn.py` is still available for comparison. Key differences:

1. **Configuration Management**: All hyperparameters now in YAML files
2. **Experiment Tracking**: Better organization with timestamps and configs
3. **Reproducibility**: Automatic config saving and version control
4. **Flexibility**: Easy parameter overrides and experiment variations
5. **Sweeping**: Built-in hyperparameter optimization support

To migrate your custom settings, create a new experiment config file or override parameters on the command line.
