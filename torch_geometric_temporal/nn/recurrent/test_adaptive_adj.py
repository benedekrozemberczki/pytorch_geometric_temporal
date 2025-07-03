#!/usr/bin/env python3

import torch
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, '/root/pytorch_geometric_temporal')

from torch_geometric_temporal.nn.recurrent.adaptive_adj import DynamicGraphLightning

def test_dynamic_graph():
    print("Testing DynamicGraphLightning...")
    
    # Create model
    DGL = DynamicGraphLightning(node_feat_dim=128)
    
    # Test input matching the error case
    random_input = torch.randn(32, 500, 30, 128)  # Batch size of 32, 500 nodes, 30 seq_len, 128 features
    print(f"Input shape: {random_input.shape}")
    
    try:
        output = DGL(random_input)
        print(f"Success! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dynamic_graph()
