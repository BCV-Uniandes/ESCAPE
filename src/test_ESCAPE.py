import os
import glob
import copy
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MultiModalClassifier
from dataset import ESCAPEDataset


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seq_ids, dist_map, labels in tqdm(dataloader, desc="Testing ..."):
            seq_ids = seq_ids.to(device)
            dist_map = dist_map.to(device)
            labels = labels.to(device)
            outputs = model(seq_ids, dist_map)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_probs)
    return y_true, y_probs


def ensemble(model_instance, ckpt_path1: str, ckpt_path2: str, dataloader: DataLoader, device: torch.device):
    model1 = copy.deepcopy(model_instance).to(device)
    model2 = copy.deepcopy(model_instance).to(device)

    state_dict1 = torch.load(ckpt_path1, map_location=device)
    state_dict2 = torch.load(ckpt_path2, map_location=device)

    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)

    y_true_1, y_probs_1 = evaluate(model1, dataloader, device)
    y_true_2, y_probs_2 = evaluate(model2, dataloader, device)

    assert np.array_equal(y_true_1, y_true_2), "Mismatch in labels between checkpoint evaluations."
    y_true = y_true_1
    y_probs = (y_probs_1 + y_probs_2) / 2.0
    return y_true, y_probs


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray):
    num_classes = y_true.shape[1]
    aps = [average_precision_score(y_true[:, i], y_probs[:, i]) for i in range(num_classes)]
    f1s = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1 = np.max(2 * precision * recall / (precision + recall + 1e-8))
        f1s.append(f1)
    macro_ap = np.mean(aps)
    macro_f1 = np.mean(f1s)
    return aps, f1s, macro_ap, macro_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble evaluation of MultiModalClassifier")

    # Dataset & paths
    parser.add_argument("--csv_file", type=str, default="../data/metadata.csv", help="Path to dataset CSV file")
    parser.add_argument("--maps_dir", type=str, default="../data/struct_maps", help="Directory with structural maps (.npy)")
    parser.add_argument("--split_file", type=str, default="../data/sequences/Test.csv", help="CSV file listing test split sequences")

    # Checkpoints
    parser.add_argument("--checkpoint1", type=str, default="../checkpoints/ESCAPE/OLD_ESCAPE_Fold1.pth", help="Path to first checkpoint")
    parser.add_argument("--checkpoint2", type=str, default="../checkpoints/ESCAPE/OLD_ESCAPE_Fold2.pth", help="Path to second checkpoint")

    # Dataloader
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sequence and image parameters
    parser.add_argument("--seq_max_len", type=int, default=200, help="Max sequence length")
    parser.add_argument("--img_size", type=int, default=224, help="Size to resize structural maps")

    # Model hyperparameters
    parser.add_argument("--seq_d_model", type=int, default=256, help="Sequence Transformer embedding dimension")
    parser.add_argument("--struct_d_model", type=int, default=192, help="Structure Transformer embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers per branch")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of output classes")
    parser.add_argument("--vocab_size", type=int, default=27, help="Vocabulary size for sequence tokens")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for image embedding")
    parser.add_argument("--img_channels", type=int, default=1, help="Number of image channels")

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Compute global min/max for normalization
    npy_paths = glob.glob(os.path.join(args.maps_dir, "*.npy"))
    global_min = float("inf")
    global_max = -float("inf")
    for p in npy_paths:
        mat = np.load(p)
        m, M = float(mat.min()), float(mat.max())
        if m < global_min:
            global_min = m
        if M > global_max:
            global_max = M

    test_dataset = ESCAPEDataset(
        csv_file=args.csv_file,
        maps_dir=args.maps_dir,
        seq_max_len=args.seq_max_len,
        split_file=args.split_file,
        global_min=global_min,
        global_max=global_max,
        img_size=args.img_size,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_instance = MultiModalClassifier(
        seq_d_model=args.seq_d_model,
        struct_d_model=args.struct_d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        vocab_size=args.vocab_size,
        max_len_seq=args.seq_max_len,
        img_size=args.img_size,
        patch_size=args.patch_size,
        img_channels=args.img_channels,
    )

    y_true, y_probs = ensemble(
        model_instance,
        args.checkpoint1,
        args.checkpoint2,
        test_dataloader,
        device,
    )

    aps, f1s, mean_ap, mean_f1 = compute_metrics(y_true, y_probs)

    print("=== Ensemble Evaluation Results ===")
    for i, ap in enumerate(aps):
        print(f"Class {i}: AP={ap*100:.1f}%, F1={f1s[i]*100:.1f}%")
    print(f"Mean AP: {mean_ap*100:.1f}%, Mean F1: {mean_f1*100:.1f}%")


if __name__ == "__main__":
    main()
