# evaluate.py
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning
from torch_geometric_temporal.nn.recurrent.stock_GNN.stock_dataset import StockDataModule

@hydra.main(config_path="torch_geometric_temporal/nn/recurrent/stock_GNN/config", config_name="evaluate", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    Loads a trained model from a checkpoint and evaluates it on the test set.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    print("--- Evaluation Configuration ---")
    print(cfg)
    
    if not cfg.ckpt_path or not torch.cuda.is_available():
        if not cfg.ckpt_path:
            print("Error: Checkpoint path (`ckpt_path`) not provided in config.")
        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This script requires a GPU.")
        return

    # 1. Load Model from Checkpoint
    print(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = DynamicGraphLightning.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location="cuda" # Ensure model is loaded to GPU
    )

    # 2. Initialize DataModule
    # We use the configuration stored in the model's hparams to ensure consistency
    print("Initializing data module...")
    data_module = StockDataModule(**model.hparams.data)
    
    # 3. Initialize Trainer
    # The trainer will handle the test loop
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False # No need to log again, just printing results
    )

    # 4. Run the Test Loop
    print("Starting evaluation on the test set...")
    test_results = trainer.test(model=model, datamodule=data_module)
    
    print("\n--- Final Test Results ---")
    if test_results:
        for result_dict in test_results:
            for key, value in result_dict.items():
                print(f"{key}: {value:.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    evaluate()
