import numpy as np
import networkx as nx

import torch
from torch_geometric_temporal.signal import temporal_signal_split

from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicGraphStaticSignal

from torch_geometric_temporal.signal import StaticHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicHeteroGraphStaticSignal

from torch_geometric_temporal.dataset import (METRLADatasetLoader, PemsBayDatasetLoader, 
                                              WindmillOutputLargeDatasetLoader, ChickenpoxDatasetLoader)

def test_index_metrla():
    loader = METRLADatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset(num_timesteps_in=6, num_timesteps_out=6)

    indexLoader = METRLADatasetLoader(raw_data_dir="/tmp/",index=True)
    train_dataloader, _,_, edges, edge_weights, _, _ = indexLoader.get_index_dataset(batch_size=1, shuffle=False, lags=6)
    
    for epoch in range(2):
        for snapshot, indexed_batch in zip(dataset, train_dataloader):
            x,y = indexed_batch
            x = torch.squeeze(x).permute(1,2,0)
            y = torch.squeeze(y)[...,0].permute(1,0)

            assert torch.equal(snapshot.x,x)
            assert torch.equal(snapshot.y,y)
            
            assert torch.equal(snapshot.edge_index,edges)
            assert torch.equal(snapshot.edge_attr,edge_weights)
 
            assert edges.shape == (2, 1722)
            assert edge_weights.shape == (1722,)
            assert x.shape == (207, 2, 6)
            assert y.shape == (207, 6)

def test_index_pemsbay():
    
    loader = PemsBayDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()

    indexLoader = PemsBayDatasetLoader(raw_data_dir="/tmp/",index=True)
    train_dataloader, _,_, edges, edge_weights, _, _ = indexLoader.get_index_dataset(batch_size=1, shuffle=False)
    
    for epoch in range(2):
        
        for snapshot, indexed_batch in zip(dataset, train_dataloader):
            x,y = indexed_batch
            x = torch.squeeze(x).permute(1,2,0)
            y = torch.squeeze(y).permute(1,2,0)
            
            assert torch.equal(snapshot.x,x)
            assert torch.equal(snapshot.y,y)

            assert torch.equal(snapshot.edge_index,edges)
            assert torch.equal(snapshot.edge_attr,edge_weights)

            assert edges.shape == (2, 2694)
            assert edge_weights.shape == (2694,)
            assert x.shape == (325, 2, 12)
            assert y.shape == (325, 2, 12)

def test_index_windmilllarge():

    loader = WindmillOutputLargeDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()
    
    indexLoader = WindmillOutputLargeDatasetLoader(raw_data_dir="/tmp/",index=True)
    train_dataloader, _,_, edges, edge_weights, _, _ = indexLoader.get_index_dataset(batch_size=1, shuffle=False)
    
    for epoch in range(2):
        for snapshot, indexed_batch in zip(dataset, train_dataloader):
            x,y = indexed_batch
            x = torch.squeeze(x).permute(1,0).float()
            y = torch.squeeze(y).permute(1,0).float()[...,0]

            assert torch.equal(snapshot.x,x)
            assert torch.equal(snapshot.y,y)

            assert torch.equal(snapshot.edge_index,edges)
            assert torch.equal(snapshot.edge_attr,edge_weights)
 
            assert edges.shape == (2, 101761)
            assert edge_weights.shape == (101761,)
            assert x.shape == (319, 8)
            assert y.shape == (319,)
        
def test_index_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()

    indexLoader = ChickenpoxDatasetLoader(index=True)
    train_dataloader, _,_, edges, edge_weights = indexLoader.get_index_dataset(batch_size=1, shuffle=False)
    
    for epoch in range(2):
        for snapshot, indexed_batch in zip(dataset, train_dataloader):
            x,y = indexed_batch
            x = torch.squeeze(x).permute(1,0).float()
            y = torch.squeeze(y).float()[0,...]
            
            assert torch.equal(snapshot.x,x) 
            assert torch.equal(snapshot.y,y) 

            assert torch.equal(snapshot.edge_index,edges)
            assert torch.equal(snapshot.edge_attr,edge_weights)
          
            assert edges.shape == (2, 102)
            assert edge_weights.shape == (102,)
            assert x.shape == (20, 4)
            assert y.shape == (20,)        