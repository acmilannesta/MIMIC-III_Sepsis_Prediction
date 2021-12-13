import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# borrowed gatech cse6250 hw5 template
class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels):
        self._labels = labels.values.squeeze()
        self._seqs = []
        for i in labels.index:
            self._seqs.append(seqs[seqs.index==i].values)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return self._seqs[idx], self._labels[idx]

# borrowed gatech cse6250 hw5 template
def seq_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
    where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """

    tmp = []
        
    max_length = max([s[0].shape[0] for s in batch])
    
    for seq, label in batch:
        tmp.append((np.vstack([seq, np.zeros((max_length-seq.shape[0], seq.shape[1]))]), seq.shape[0], label))

    tmp.sort(key=lambda x: x[1], reverse=True)


    seqs_tensor = torch.FloatTensor(np.array([t[0] for t in tmp]))
    lengths_tensor = torch.LongTensor(np.array([t[1] for t in tmp]))
    labels_tensor = torch.LongTensor(np.array([t[2] for t in tmp]))

    return (seqs_tensor, lengths_tensor), labels_tensor