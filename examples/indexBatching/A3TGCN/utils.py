import pickle
import torch
import numpy as np
import pandas as pd
import csv
import os
import time
import scipy.sparse as sp

def masked_mae_loss(y_pred, y_true):

    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean() 



def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def adjacency_to_edge_index(adj_mx):
    """
    Convert an adjacency matrix to edge_index and edge_weight.
    Args:
        adj_mx (np.ndarray): Adjacency matrix of shape (num_nodes, num_nodes).
    Returns:
        edge_index (torch.LongTensor): Shape (2, num_edges), source and target nodes.
        edge_weight (torch.FloatTensor): Shape (num_edges,), edge weights.
    """
    # Convert to sparse matrix
    adj_sparse = sp.coo_matrix(adj_mx)

    # Extract edge indices and weights
    edge_index = torch.tensor(
        np.vstack((adj_sparse.row, adj_sparse.col)), dtype=torch.long
    )
    edge_weight = torch.tensor(adj_sparse.data, dtype=torch.float)

    return edge_index, edge_weight

# standard approach: see https://github.com/chnsh/DCRNN_PyTorch
def benchmark_preprocess(h5File, dataset, key=None):

    if "pems" in dataset.lower():
        df = pd.read_hdf(h5File)

        x_offsets = np.sort(
            # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
            np.concatenate((np.arange(-11, 1, 1),))
        )
        # Predict the next one hour
        y_offsets = np.sort(np.arange(1, 13, 1))
        num_samples, num_nodes = df.shape

        data = np.expand_dims(df.values, axis=-1)
        data_list = [data]
        add_time_in_day= True
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)

        data = np.concatenate(data_list, axis=-1)


        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]

            
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)



        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)

        return torch.tensor(x,dtype=torch.float),torch.tensor(y,dtype=torch.float)

def collect_metrics():
    import psutil
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
    
    try:
        # Initialize NVML for GPU metrics
        nvmlInit()

        # Open the CSV file in append mode
        
        data = []
        max_gpu_mem = -1
        max_system_mem = -1
        while True:
            # Collect system memory usage
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            mem = psutil.virtual_memory()
            total_rss = sum(proc.memory_info().rss for proc in psutil.process_iter(attrs=['memory_info']))
            system_memory_used = total_rss / (1024**2)  # Convert to MB
            system_memory_total = mem.total / (1024**2)  # Convert to MB

            # Collect GPU memory usage
            gpu_metrics = []
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_used = info.used / (1024**2)  # Convert to MB
            gpu_memory_total = info.total / (1024**2)  # Convert to MB

            max_gpu_mem = max(gpu_memory_used, max_gpu_mem)
            max_system_mem = max(system_memory_used, max_system_mem)
            data.append([timestamp,system_memory_used,system_memory_total, gpu_memory_used, gpu_memory_total])
            
            if os.path.isfile("flag.txt"):
                os.remove("flag.txt")
                break

            if os.path.isfile("stats.csv"):
                with open("system_stats.csv", mode="w", newline="") as f:
                    writer = csv.writer(f)

                    # Write headers to the CSV file
                    headers = [
                        "Timestamp",
                        "System_Memory_Used",
                        "System_Memory_Total",
                        "GPU_Memory_Used",
                        "GPU_Memory_Total"
                    ]
                    writer.writerow(headers)
                    writer.writerows(data)

                    
                df = pd.read_csv("stats.csv")
                df['system_memory_total'] = system_memory_total
                df['max_system_mem'] = max_system_mem
                df['gpu_memory_total'] = gpu_memory_total
                df['max_gpu_mem'] = max_gpu_mem
                
                df.to_csv("stats.csv", index=False)
                break
            time.sleep(1)

    except Exception as e:
        print("Error in collecting metrics:", str(e))

    finally:
        # Shutdown NVML
        nvmlShutdown()