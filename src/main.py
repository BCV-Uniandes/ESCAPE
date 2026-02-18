# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import glob
import random
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from args import get_args
from dataset import ESCAPEDataset
from models import ClassifierTransformer, StructTransformer, MultiModalClassifier

# Amino acid vocabulary and sequence converter
AA_LIST = ['-','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
VOCAB = {aa: i+1 for i, aa in enumerate(AA_LIST)}
VOCAB['PAD'] = 0

def seq_to_ids(seq: str, max_len: int) -> torch.LongTensor:
    """
    Convert amino-acid sequence to fixed-length ID tensor.
    """
    ids = [VOCAB.get(c, 0) for c in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def evaluate(model, loader, device, mode, args):
    """
    Evaluate model and compute loss + metrics.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    y_true, y_probs = [], []

    with torch.no_grad():
        for seq_ids, dist_map, labels in tqdm(loader, desc="Evaluating"):
            seq_ids = seq_ids.to(device)
            dist_map = dist_map.to(device)
            labels   = labels.to(device)
            if mode == 'sequence':
                out = model(seq_ids)
            elif mode == 'distance':
                out = model(dist_map)
            else:
                out = model(seq_ids, dist_map)
            loss = criterion(out, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(out).cpu().numpy()
            y_true.append(labels.cpu().numpy())
            y_probs.append(probs)

    y_true = np.vstack(y_true)
    y_probs = np.vstack(y_probs)
    per_ap = [average_precision_score(y_true[:,i], y_probs[:,i]) for i in range(y_true.shape[1])]
    per_f1 = []
    for i in range(y_true.shape[1]):
        prec, rec, _ = precision_recall_curve(y_true[:,i], y_probs[:,i])
        f1_scores = 2*(prec*rec)/(prec+rec+1e-8)
        per_f1.append(np.max(f1_scores))
    macro_ap  = np.mean(per_ap)
    macro_f1  = np.mean(per_f1)
    macro_auc = roc_auc_score(y_true, y_probs, average='macro')

    return total_loss/len(loader), macro_auc, macro_ap, macro_f1, per_ap, per_f1

if __name__ == '__main__':
    args = get_args()

    # Set seeds for reproducibility
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


    wandb.init(
        project=args.project,
        name=args.run_name or f"run_{args.mode}",
        config=vars(args),
        mode="disabled" if args.wandb==False else "online",
    )

    if args.fold == 1:
        train_split = args.fold1_file
        val_split   = args.fold2_file
    else:
        train_split = args.fold2_file
        val_split   = args.fold1_file

    # Dataset and loaders
    #seq_transform = lambda s: seq_to_ids(s, args.seq_max_len)

    ds_train = ESCAPEDataset(
        maps_dir=args.maps_dir,
        seq_max_len=args.seq_max_len,
        test_file=train_split,      # CSV of train fold 
        global_min=global_min,
        global_max=global_max,
        img_size=args.dist_max_len  
    )

    ds_val = ESCAPEDataset(
        maps_dir=args.maps_dir,
        seq_max_len=args.seq_max_len,
        test_file=val_split,        # CSV of val fold 
        global_min=global_min,
        global_max=global_max,
        img_size=args.dist_max_len
    )

    tr_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size)

    # Model instantiation
    if args.mode == 'sequence':
        model = ClassifierTransformer(
            d_model     = args.seq_d_model,
            n_heads     = args.seq_n_heads,
            num_layers  = args.seq_n_layers,
            num_classes = args.num_classes,
            vocab_size  = len(VOCAB),
            max_len     = args.seq_max_len
        )

    elif args.mode == 'distance':
        model = StructTransformer(
            img_size    = args.dist_max_len,
            patch_size  = 16,                # o args.patch_size si lo tienes
            d_model     = args.dist_d_model,
            n_heads     = args.dist_n_heads,
            n_layers    = args.dist_n_layers,
            img_channels= 1,
            num_classes = args.num_classes   
        )

    elif args.mode == 'MultiModal':
        model = MultiModalClassifier(
            seq_d_model    = args.seq_d_model,
            struct_d_model = args.dist_d_model,
            n_heads        = args.seq_n_heads,
            num_layers     = args.seq_n_layers,
            num_classes    = args.num_classes,
            vocab_size     = len(VOCAB),
            max_len_seq    = args.seq_max_len,
            img_size       = args.dist_max_len,
            patch_size     = 16,
            img_channels   = 1
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")



    # Print model summary
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and scheduler
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None

    criterion = nn.BCEWithLogitsLoss()

    # Training loop

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        model.train()
        train_loss = 0.0
        for seq_ids, dist_map, labels in tqdm(tr_loader, desc="Training"):
            seq_ids = seq_ids.to(device)
            dist_map = dist_map.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if args.mode == 'sequence':
                out = model(seq_ids)
            elif args.mode == 'distance':
                out = model(dist_map)
            elif args.mode == 'MultiModal':
                out = model(seq_ids, dist_map)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        # Validation and logging
        #validate every 10 epochs
        if epoch % 10 == 0:
            print("Validating...")
            val_loss, val_auc, val_ap, val_f1, per_ap, per_f1 = evaluate(model, val_loader, device, args.mode, args)
            wandb.log({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Epoch": epoch
            })
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            wandb.log({
                "Train Loss": train_loss,
                "Epoch": epoch
            })
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")
        if scheduler:
            scheduler.step()

        # Full metrics logging
        if epoch % args.eval_interval == 0:
            metrics = {
                "Val AUC": val_auc,
                "Val AP": val_ap,
                "Val F1": val_f1
            }
            for i, label in enumerate(ESCAPEDataset.LABEL_COLUMNS):
                metrics[f"AP_{label}"] = per_ap[i]
                metrics[f"F1_{label}"] = per_f1[i]
            wandb.log(metrics)
            print("-- Full Evaluation Metrics --")
            for k,v in metrics.items():
                print(f" {k}: {v:.4f}")
        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            # save it on the mode folder
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            if not os.path.exists(os.path.join("outputs", args.mode)):
                os.makedirs(os.path.join("outputs", args.mode))
            if not os.path.exists(os.path.join("outputs", args.mode, str(args.fold))):
                os.makedirs(os.path.join("outputs", args.mode, str(args.fold)))
            
            torch.save(model.state_dict(), os.path.join("outputs", args.mode, str(args.fold), f"model_seed1665_{args.mode}_Fold{args.fold}_{args.run_name}_epoch{epoch}.pth"))
            print(f"Model checkpoint saved at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), os.path.join("outputs", args.mode, str(args.fold), f"model_seed1665_Fold{args.fold}_{args.run_name}.pth"))
