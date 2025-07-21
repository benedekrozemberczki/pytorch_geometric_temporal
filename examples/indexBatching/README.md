## PyTorch Geometric Temporal - Index

Index-batching is a technique that reduces the memory cost of training ST-GNNs with spatiotemporal data with no impact on accurary, enabling greater scalability and training on the full PeMS dataset without graph partioning for the first time. Leveraging the reduced memory footprint, this techique also enables GPU-index-batching - a technique that performs preprocessing entirely in GPU memory and utilizes a single CPU-to-GPU mem-copy in place of batch-level CPU-to-GPU transfers throughout training. We implemented GPU-index-batching and index-batching for the following existing datasets and added two new datasets (highlighted in bold) to PyTorch Geometric Temporal (PGT): 

* PeMs-Bay
* Metr-LA
* WindmillLarge
* HungaryChickenpox
* **PeMSAllLA**
* **PeMS**

This folder contains examples with a few existing PGT models. Wherever possible, we follow the existing workflow for a given model. We hope to build out our examples over time. 


Utilizing index-batching requires minimal modifications to the existing PGT workflow. For example, the following is a sample training loop with static graph dataset with temporal signal:

```
train_dataloader, _, _, edges, edge_weights, means, stds = loader.get_index_dataset(batch_size=batch_size)

for batch in train_dataloader:
            X_batch, y_batch = batch

            # Forward pass
            outputs = model(X_batch, edges, edge_weights) 

            # Calculate loss 
            loss = masked_mae_loss((outputs * std) + mean, (y_batch * std) + mean)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


```

The single-GPU examples in this repo can be executed as follows: `python3 <datasetName>.py` and have the following parameters:

| Argument | Short Form | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--epochs` | `-e` | `int` | `30` | The desired number of training epochs. |
| `--batch-size` | `-bs` | `int` | `64` | The desired batch size for training. |
| `--gpu` | `-g` | `str` | `"False"` | Indicates whether data should be preprocessed and migrated directly to the GPU. Use `"True"` to enable GPU processing. |
| `--debug` | `-d` | `str` | `"False"` | Enables debug mode, printing values for debugging. Use `"True"` to enable debugging. |



We also provide a multi-node, multi-GPU Dask-DDP training implementation for PeMS-Bay, PemsAllLA, and the full PeMS dataset. It has the following parameters:

| Argument | Short Form | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--epochs` | `-e` | `int` | `100` | The desired number of training epochs. |
| `--batch-size` | `-bs` | `int` | `64` | The desired batch size for training. |
| `--gpu` | `-g` | `str` | `"False"` | Indicates whether data should be preprocessed and migrated directly to the GPU. Use `"True"` to enable GPU processing. |
| `--debug` | `-d` | `str` | `"False"` | Enables debug mode, printing values for debugging. Use `"True"` to enable debugging. |
| `--dask-cluster-file` | N/A | `str` | `""` | Path to the Dask scheduler file for the Dask CLI interface. |
| `--npar` | `-np` | `int` | `1` | The number of GPUs or workers per node. |
| `--dataset` | N/A | `str` | `"pems-bay"` | Specifies which dataset is in use (e.g., PeMS-Bay, PeMS-All-LA, PeMS). |

To execute in a single node, omit the `--dask-cluster-file` argument. To run multi-node, setup a Dask scheduler and dask workers user the [Dask command line interface](https://docs.dask.org/en/latest/deploying-cli.html) and pass the scheduler file via `--dask-cluster-file`. An example multi-GPU, multi-node script for Argonne's [Polaris supercomputer](https://www.alcf.anl.gov/polaris)  is shown in `submit.sh`. 

