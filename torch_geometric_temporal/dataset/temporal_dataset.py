from torch.utils.data import Dataset
from torch_geometric.data import Data

class TemporalData(Data):
    pass

class TemporalDataset(Dataset):
    def __init__(self, data_list, horizon, laplacian_normalization):
        super().__init__()

        if isinstance(data_list, dict):
            self.data = list(data_list.values())
        elif isinstance(data_list, (list, tuple)):
            self.data = data_list
        else:
            self.data = data_list

        self.horizon = horizon
        self.normalization = laplacian_normalization

    def __len__(self):
        return len(self.data) - self.horizon

    def __getitem__(self, idx):
        data_list = self.data[idx:idx + self.horizon]
        # if self.normalization is not None:
        #     data_list = [LaplacianLambdaMax(normalization=self.normalization, is_undirected=d.is_undirected())(d) for d in
        #                  data_list]
        tgt = self.data[idx + self.horizon].y
        return data_list, tgt
