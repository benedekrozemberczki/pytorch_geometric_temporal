#!/usr/bin/env python3
"""
Hydra-based training script for Stock GNN pipeline
Usage:
    python train_stock_gnn_hydra.py
    python train_stock_gnn_hydra.py data=debug trainer=cpu_debug
    python train_stock_gnn_hydra.py model.lr=5e-4 trainer.max_epochs=100
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric_temporal.nn.recurrent.stock_GNN.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning
from torch_geometric_temporal.nn.recurrent.stock_GNN.adp_adj_loss import AccumulativeGainLoss

# Optimize RTX 4090 Tensor Core performance
torch.set_float32_matmul_precision('medium')


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup PyTorch Lightning callbacks from config"""
    callbacks = []
    
    if "callbacks" in cfg and cfg.callbacks is not None:
        for callback_name, callback_cfg in cfg.callbacks.items():
            if callback_cfg is not None and "_target_" in callback_cfg:
                try:
                    callback = instantiate(callback_cfg)
                    callbacks.append(callback)
                    print(f"✓ Added callback: {callback_name}")
                except Exception as e:
                    print(f"⚠ Failed to instantiate callback {callback_name}: {e}")
    
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """Setup PyTorch Lightning logger from config"""
    if "logger" not in cfg or cfg.logger is None:
        return None
    
    try:
        logger = instantiate(cfg.logger)
        print(f"✓ Using logger: {cfg.logger._target_}")
        return logger
    except Exception as e:
        print(f"⚠ Failed to instantiate logger: {e}")
        # Fallback to CSV logger
        try:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger("logs", name="stock_gnn")
            print("📋 Fallback to CSV Logger")
            return logger
        except Exception:
            print("❌ No logger available")
            return None


def create_data_module(cfg: DictConfig) -> StockDataModule:
    """Create the data module from config"""
    data_cfg = cfg.data
    
    # Extract debug settings
    debug_enabled = cfg.get("debug", {}).get("enabled", False)
    verbose_data = cfg.get("debug", {}).get("verbose_data_loading", False)
    
    dm = StockDataModule(
        data_dir=data_cfg.data_dir,
        use_factors=data_cfg.use_factors,
        sequence_length=data_cfg.sequence_length,
        prediction_horizons=data_cfg.prediction_horizons,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        test_ratio=data_cfg.test_ratio,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        normalize_features=data_cfg.normalize_features,
        normalize_targets=data_cfg.normalize_targets,
        normalization_method=data_cfg.normalization_method,
        debug=debug_enabled or verbose_data,
    )
    
    print(f"📊 Created data module:")
    print(f"   - Data dir: {data_cfg.data_dir}")
    print(f"   - Batch size: {data_cfg.batch_size}")
    print(f"   - Sequence length: {data_cfg.sequence_length}")
    print(f"   - Prediction horizons: {data_cfg.prediction_horizons}")
    print(f"   - Debug mode: {debug_enabled or verbose_data}")
    
    return dm


def create_model(cfg: DictConfig, node_feat_dim: int) -> DynamicGraphLightning:
    """Create the model from config"""
    model_cfg = cfg.model
    loss_cfg = cfg.loss
    
    # Create loss function
    loss_fn = instantiate(loss_cfg)
    
    model = DynamicGraphLightning(
        node_feat_dim=node_feat_dim,
        gru_hidden_dim=model_cfg.gru_hidden_dim,
        gnn_hidden_dim=model_cfg.gnn_hidden_dim,
        k_nn=model_cfg.k_nn,
        lr=model_cfg.lr,
        loss_fn=loss_fn,
        add_self_loops=model_cfg.add_self_loops,
        metric_compute_frequency=getattr(model_cfg, 'metric_compute_frequency', 10),
        weight_decay=getattr(model_cfg, 'weight_decay', 1e-4),
        scheduler_config=getattr(model_cfg, 'scheduler', {}),
        gnn_type=getattr(model_cfg, 'gnn_type', 'gcn'),
        gat_heads=getattr(model_cfg, 'gat_heads', 4),
        gat_dropout=getattr(model_cfg, 'gat_dropout', 0.1),
    )
    
    print(f"🧠 Created model:")
    print(f"   - Node features: {node_feat_dim}")
    print(f"   - GRU hidden: {model_cfg.gru_hidden_dim}")
    print(f"   - GNN hidden: {model_cfg.gnn_hidden_dim}")
    print(f"   - K-NN: {model_cfg.k_nn}")
    print(f"   - Learning rate: {model_cfg.lr}")
    print(f"   - Weight decay: {getattr(model_cfg, 'weight_decay', 1e-4)}")
    print(f"   - Scheduler: {getattr(model_cfg, 'scheduler', {}).get('type', 'None')}")
    print(f"   - Metric frequency: {getattr(model_cfg, 'metric_compute_frequency', 10)} epochs")
    print(f"   - Loss function: {loss_cfg._target_}")
    
    return model


def create_trainer(cfg: DictConfig, callbacks: list, logger) -> pl.Trainer:
    """Create PyTorch Lightning trainer from config"""
    trainer_cfg = cfg.trainer
    
    # Handle devices configuration for DDP
    devices = trainer_cfg.devices
    if isinstance(devices, int) and devices > 1 and trainer_cfg.strategy == "ddp":
        print(f"🔥 Using DDP with {devices} GPUs")
    
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        min_epochs=trainer_cfg.get("min_epochs", 1),
        accelerator=trainer_cfg.accelerator,
        devices=devices,
        strategy=trainer_cfg.strategy,
        precision=trainer_cfg.get("precision", "32-true"),
        benchmark=trainer_cfg.get("benchmark", True),
        deterministic=trainer_cfg.get("deterministic", False),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0),
        gradient_clip_algorithm=trainer_cfg.get("gradient_clip_algorithm", "norm"),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        enable_progress_bar=trainer_cfg.get("enable_progress_bar", True),
        enable_model_summary=trainer_cfg.get("enable_model_summary", True),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        limit_test_batches=trainer_cfg.get("limit_test_batches", 1.0),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        enable_checkpointing=trainer_cfg.get("enable_checkpointing", True),
        sync_batchnorm=trainer_cfg.get("sync_batchnorm", False),
        fast_dev_run=trainer_cfg.get("fast_dev_run", False),
        overfit_batches=trainer_cfg.get("overfit_batches", 0),
        detect_anomaly=trainer_cfg.get("detect_anomaly", False),
        callbacks=callbacks,
        logger=logger,
    )
    
    print(f"⚡ Created trainer:")
    print(f"   - Max epochs: {trainer_cfg.max_epochs}")
    print(f"   - Accelerator: {trainer_cfg.accelerator}")
    print(f"   - Devices: {devices}")
    print(f"   - Strategy: {trainer_cfg.strategy}")
    print(f"   - Precision: {trainer_cfg.get('precision', '32-true')}")
    
    return trainer


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Print configuration
    print("=" * 80)
    print("🚀 Stock GNN Training Pipeline with Hydra")
    print("=" * 80)
    print("📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Set random seed for reproducibility
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
        print(f"🌱 Set random seed: {cfg.seed}")
    
    # Create output directories
    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)
    os.makedirs(cfg.get("log_dir", "logs"), exist_ok=True)
    
    try:
        # 1. Setup data module
        print("\n📊 Setting up data module...")
        dm = create_data_module(cfg)
        dm.prepare_data()
        dm.setup("fit")
        
        # Get feature dimension
        node_feat_dim = dm.get_feature_dim()
        stock_num = dm.get_stock_num()
        prediction_horizons = dm.get_prediction_horizons()
        
        print(f"   ✓ Feature dimension: {node_feat_dim}")
        print(f"   ✓ Number of stocks: {stock_num}")
        print(f"   ✓ Prediction horizons: {prediction_horizons}")
        
        # 2. Create model
        print("\n🧠 Creating model...")
        model = create_model(cfg, node_feat_dim)
        
        # 3. Setup callbacks and logger
        print("\n⚙️ Setting up callbacks and logger...")
        callbacks = setup_callbacks(cfg)
        logger = setup_logger(cfg)
        
        # 4. Create trainer
        print("\n⚡ Creating trainer...")
        trainer = create_trainer(cfg, callbacks, logger)
        
        # 5. Start training
        print("\n🎯 Starting training...")
        print("-" * 60)
        trainer.fit(model, datamodule=dm)
        
        # 6. Test the model
        print("\n🧪 Testing model...")
        print("-" * 60)
        trainer.test(model, datamodule=dm)
        
        print("\n✅ Training completed successfully!")
        
        # Save final configuration
        config_path = Path(trainer.logger.log_dir) / "config.yaml"
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)
        print(f"💾 Saved configuration to: {config_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise e


if __name__ == "__main__":
    main()
