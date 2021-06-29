import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split


class LitDiffConvModel(pl.LightningModule):

    def __init__(self, node_features, filters):
        super().__init__()
        self.recurrent = DCRNN(node_features, filters, 1)
        self.linear = torch.nn.Linear(filters, 1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch.x
        y = train_batch.y.view(-1, 1)
        edge_index = train_batch.edge_index
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.x
        y = val_batch.y.view(-1, 1)
        edge_index = val_batch.edge_index
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        loss = F.mse_loss(h, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics


loader = ChickenpoxDatasetLoader()

dataset_loader = loader.get_dataset(lags=32)

train_loader, val_loader = temporal_signal_split(dataset_loader,
                                                 train_ratio=0.2)

model = LitDiffConvModel(node_features=32,
                         filters=16)

early_stop_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=10,
                                    verbose=False,
                                    mode='max')

trainer = pl.Trainer(callbacks=[early_stop_callback])

trainer.fit(model, train_loader, val_loader)
