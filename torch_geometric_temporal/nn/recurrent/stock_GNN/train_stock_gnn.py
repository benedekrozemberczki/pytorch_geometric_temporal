# train.py (放在项目根目录)
import pytorch_lightning as pl
from torch_geometric_temporal.nn.recurrent.stock_GNN.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning

def main():
    # 1) 构造 DataModule
    dm = StockDataModule(
        data_dir="/home/xu/clean_data",
        use_factors=True,
        sequence_length=20,
        prediction_horizons=[1,5,10],
        batch_size=32,
        normalize_features=True,
        normalize_targets=True,
    )
    dm.prepare_data()
    dm.setup("fit")

    # 2) 构造模型（node_feat_dim = 特征维度，N = 股票数）
    node_feat_dim = dm.get_feature_dim()
    model = DynamicGraphLightning(
        node_feat_dim=node_feat_dim,
        gru_hidden_dim=64,
        gnn_hidden_dim=64,
        k_nn=8,
        lr=1e-3
    )

    # 3) Trainer
    trainer = pl.Trainer(       
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
    )

    # 4) 训练 + 验证
    trainer.fit(model, datamodule=dm)

    # 5) 测试
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()