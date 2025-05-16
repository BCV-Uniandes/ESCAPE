import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class ESCAPEDataset(Dataset):
    """
    ESCAPE Dataset class.

    Args:
        csv_file (str): Path to CSV with dataset metadata (sequences, map filenames, labels).
        maps_dir (str): Directory containing structural maps (.npy or images).
        seq_max_len (int): Max length for sequence tokenization.
        split_file (str, optional): CSV file to filter dataset rows by index (e.g. train/test split).

    Behavior:
        - Computes global min/max for normalization across all .npy maps.
        - Uses predefined LABEL_COLUMNS if available, else infers from CSV.
        - Tokenizes sequences internally.
    """

    AA_LIST = ['-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    VOCAB = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
    VOCAB['PAD'] = 0

    LABEL_COLUMNS = ["Antibacterial", "Antifungal", "Antiviral", "Antiparasitic", "Antimicrobial"]

    def __init__(
        self, 
        csv_file: str,
        maps_dir: str,
        seq_max_len: int,
        split_file: str = None,
        global_min: float = None,
        global_max: float = None,
        img_size: int = 224
    ):

        df = pd.read_csv(csv_file)

        if split_file is not None:
            splits = pd.read_csv(split_file)['Sequence'].tolist()
            df = df.set_index('Sequence').loc[splits].reset_index()
        df['Hash'] = df['Hash'].fillna('')
        df[self.LABEL_COLUMNS] = df[self.LABEL_COLUMNS].fillna(0.0)
        self.df = df
        self.maps_dir = maps_dir
        self.seq_max_len = seq_max_len
        self.global_min = global_min
        self.global_max = global_max
        self.img_size = img_size

    def seq_to_ids(self, seq):
        """
        Tokenize amino acid sequence string into fixed-length tensor of token IDs.

        Pads with 0 or truncates to seq_max_len.

        Args:
            seq (str): amino acid sequence string.

        Returns:
            torch.LongTensor: tokenized sequence tensor of shape (seq_max_len,).
        """
        seq = seq.upper()  # ensure uppercase for matching VOCAB keys
        ids = [self.VOCAB.get(c, 0) for c in seq[:self.seq_max_len]]
        ids += [0] * (self.seq_max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize sequence
        seq_ids = self.seq_to_ids(row["Sequence"])

        # --- load distance map ---
        npy_path = os.path.join(self.maps_dir, f"{row['Hash']}.npy")
        mat = np.load(npy_path).astype(np.float32)

        # --- normalize globally ---
        mat = (mat - self.global_min) / (self.global_max - self.global_min + 1e-8)
        mat = np.clip(mat, 0.0, 1.0)

        # --- resize to square image ---
        x = torch.from_numpy(mat).unsqueeze(0)  # (1, H, W)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # now (1, img_size, img_size)

        # --- labels ---
        labels = torch.tensor(
            row[self.LABEL_COLUMNS].values.astype(np.float32),
            dtype=torch.float32
        )

        return seq_ids, x, labels
