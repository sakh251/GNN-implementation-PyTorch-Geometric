
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import json
class ppiDataset(InMemoryDataset):
    def __init__(self, root, files=None, parocced_file_name=None, transform=None, pre_transform=None):
        self.files = files
        self.parocced_file_name = parocced_file_name
        super(ppiDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'../{self.parocced_file_name}']

    def download(self):
        pass

    def process(self):
        x_feats_train = np.load(self.files[0])
        y_labels_train = np.load(self.files[1])
        x_graphID_train = np.load(self.files[2])

        with open(self.files[3]) as f:
            g = json.load(f)

        edge_list = []
        for link in g['links']:
            edge_list.append((link['source'], link['target']))

        edge_index = np.array(edge_list).transpose()
        e = torch.tensor([edge_index[0],
                                   edge_index[1]], dtype=torch.long)
        x = torch.tensor(x_feats_train, dtype=torch.float)
        y = torch.tensor((y_labels_train))
        id = torch.tensor(x_graphID_train)
        data = Data(x=x, edge_index=e, y=y, id=id)
        data_list = []
        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])