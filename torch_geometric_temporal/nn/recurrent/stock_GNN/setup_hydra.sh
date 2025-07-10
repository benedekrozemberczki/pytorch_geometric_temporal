#!/bin/bash
# Installation script for Stock GNN with Hydra

echo "🚀 Setting up Stock GNN Training Pipeline with Hydra"
echo "=" * 60

# Install Hydra and dependencies
echo "📦 Installing Hydra and related packages..."
pip install -r requirements_hydra.txt

# Check installation
echo "✅ Checking installations..."
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')"
python -c "import omegaconf; print(f'OmegaConf version: {omegaconf.__version__}')"

# Check PyTorch and related packages
echo "🔍 Checking PyTorch ecosystem..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'PyTorch Lightning version: {pl.__version__}')"

# Test configuration loading
echo "⚙️ Testing configuration loading..."
python -c "
from omegaconf import OmegaConf
import os
config_path = 'config/config.yaml'
if os.path.exists(config_path):
    cfg = OmegaConf.load(config_path)
    print('✓ Main config loaded successfully')
    print(f'  Experiment name: {cfg.experiment.name}')
else:
    print('❌ Config file not found')
"

echo "🎉 Setup complete! You can now run:"
echo "   python train_stock_gnn_hydra.py --help"
echo "   python train_stock_gnn_hydra.py experiment=debug"
