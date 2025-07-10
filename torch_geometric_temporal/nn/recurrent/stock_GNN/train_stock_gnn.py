# train.py (放在项目根目录)
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch_geometric_temporal.nn.recurrent.stock_GNN.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.adaptive_adj import DynamicGraphLightning

# 优化RTX 4090的Tensor Core性能
torch.set_float32_matmul_precision('medium')  # 或者 'high' 以获得更高性能但稍低精度

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
        debug=False,  # 设置为 True 可以看到详细的数据处理信息
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
        ModelCheckpoint(
            monitor='val_loss',               # 改为 val_loss (不使用斜杠)
            filename='stock-gnn-{epoch:02d}-{val_loss:.4f}',  # 对应修改文件名
            save_top_k=3,
            mode='min',
            auto_insert_metric_name=False,    # 防止自动插入metric名称
            save_last=True,                   # 额外保存最后一个checkpoint
            verbose=True,                     # 显示保存信息
        ),
        EarlyStopping(
            monitor='val_loss',               # 对应修改为 val_loss
            patience=10,
            mode='min',
            verbose=True,                     # 显示早停信息
        )
    ]

    # 配置Logger
    # 选择一个Logger - CSV或TensorBoard
    try:
        # 尝试使用TensorBoard
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger("logs", name="stock_gnn")
        print("使用 TensorBoard Logger")
    except ImportError:
        # 如果TensorBoard不可用，使用CSV Logger
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger("logs", name="stock_gnn")
        print("使用 CSV Logger")

    # 3) Trainer - 使用 DDP
    trainer = pl.Trainer(       
        max_epochs=200,
        accelerator="gpu",
        devices=2,
        strategy="ddp",              # 明确指定使用DDP策略
        log_every_n_steps=1,         # 每1步打印一次日志（更频繁）
        enable_progress_bar=True,    # 启用进度条 (默认True)
        enable_model_summary=True,   # 启用模型摘要 (默认True)
        callbacks=callbacks,         # 可以添加自定义回调
        # 明确指定Logger
        logger=logger,               # 使用配置的logger
        enable_checkpointing=True,   # 启用检查点
        check_val_every_n_epoch=1,   # 每个epoch都验证
        # DDP相关配置
        sync_batchnorm=True,         # 同步BatchNorm
        # replace_sampler_ddp=True,  # 新版本已移除此参数，DDP会自动处理采样
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