import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizer import byteTokenizer


class DatasetLoad(Dataset):
    def __init__(self, path: str, context_length: int):
        self.tokenizer = byteTokenizer(path)
        self.data = torch.tensor(self.tokenizer.encode(self.tokenizer.text), dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data)-self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.context_length]
        y = self.data[idx+1:idx+self.context_length+1]
        return x, y

    
def create_Dataloader(path: str, batch_size: int, context_length: int, train: float=0.8):
    dataset = DatasetLoad(path, context_length)
    n = len(dataset)
    train_size = int(n * train)
    val_size = n - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, dataset
    
