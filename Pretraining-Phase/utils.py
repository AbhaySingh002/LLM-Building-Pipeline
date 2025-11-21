import torch
from pathlib import Path
from tokenizer import byteTokenizer

class DatasetLoad:
    def __init__(self, path: Path, context_length: int, batch_size: int, device: str = "cpu"):
        self.tokenizer = byteTokenizer(path)
        self.data = torch.tensor(self.tokenizer.encode(self.tokenizer.text), dtype=torch.long)
        self.vocab_size = self.tokenizer.vocab_size
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        n = int(0.8 *len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split: str = 'train'):
        """        
        Returns a batch of input-target pairs for next-token prediction.
        split: 'train' or 'val' (for training/validation split)
        """
        
        data = self.val_data if split == "val" else self.train_data
        data = self.data  

        ix = torch.randint(0, len(data) - self.context_length - 1, (self.batch_size,))

        
        x = torch.stack([data[i:i+self.context_length] for i in ix])
        y = torch.stack([data[i+1:i+self.context_length+1] for i in ix])

        return x.to(self.device), y.to(self.device)