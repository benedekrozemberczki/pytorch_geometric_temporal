# Install joblib launcher first
# pip install hydra-joblib-launcher

# Run 2 jobs in parallel (safe for 2-GPU setup)
python train_stock_gnn_hydra.py experiment=sweep hydra/launcher=joblib hydra.launcher.n_jobs=2

# Or reduce batch size and run more jobs
# python train_stock_gnn_hydra.py experiment=sweep hydra/launcher=joblib hydra.launcher.n_jobs=4 data.batch_size=16