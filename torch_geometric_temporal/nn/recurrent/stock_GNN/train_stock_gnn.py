# train.py (放在项目根目录)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
        gnn_hidden_dim=128,
        k_nn=8,
        lr=1e-3
    )

    # 可选：添加回调函数来更好地控制训练过程
    callbacks = [
        # 暂时注释掉这些回调，以确保能看到基本输出
        ModelCheckpoint(
            monitor='val/loss',
            filename='stock-gnn-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min',
        )
    ]

    # 3) Trainer
    trainer = pl.Trainer(       
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,         # 每1步打印一次日志（更频繁）
        enable_progress_bar=True,    # 启用进度条 (默认True)
        enable_model_summary=True,   # 启用模型摘要 (默认True)
        callbacks=callbacks,         # 可以添加自定义回调
        # 添加这些参数来确保能看到损失
        logger=True,                 # 明确启用日志记录器
        enable_checkpointing=True,   # 启用检查点
        check_val_every_n_epoch=1,   # 每个epoch都验证
    )

    # 4) 训练 + 验证
    print("开始训练...")
    print(f"特征维度: {node_feat_dim}")
    print(f"股票数量: {dm.get_stock_num()}")
    print(f"预测期数: {dm.get_prediction_horizons()}")
    
    trainer.fit(model, datamodule=dm)
    
    print("训练完成！")

    # 5) 测试
    print("开始测试...")
    trainer.test(model, datamodule=dm)
    print("测试完成！")

if __name__ == "__main__":
    main()