import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="AMP Classification with Sequence and Distance Transformers"
    )
    parser.add_argument(
        '--maps_dir', type=str, default="/media/SSD5/naparicioc/Distance_Maps",
        help="Directory containing distance map .npy files"
    )
    parser.add_argument(
        '--fold', type=int, choices=[1, 2], required=False,
        help="Specify the fold number for cross-validation (1-2)"
    )
    parser.add_argument(
        '--num_classes', type=int, default=5,
        help="Number of classes for classification"
    )

    # Mode selection
    parser.add_argument(
        '--mode', choices=['sequence','distance','MultiModal'], required=True,
        help="Model mode: 'sequence' for sequence-only, 'distance' for distance-only, 'MultiModal' for combined cross-attention"
    )

    # Sequence Transformer parameters
    seq_group = parser.add_argument_group('Sequence Transformer Parameters')
    seq_group.add_argument(
        '--seq_max_len', type=int, default=200,
        help="Maximum token length for the sequence transformer"
    )
    seq_group.add_argument(
        '--seq_d_model', type=int, default=256,
        help="Embedding dimension for the sequence transformer"
    )
    seq_group.add_argument(
        '--seq_n_heads', type=int, default=8,
        help="Number of attention heads in the sequence transformer"
    )
    seq_group.add_argument(
        '--seq_n_layers', type=int, default=4,
        help="Number of encoder layers in the sequence transformer"
    )

    # Distance Transformer parameters
    dist_group = parser.add_argument_group('Distance Transformer Parameters')
    dist_group.add_argument(
        '--dist_max_len', type=int, default=224,
        help="Dimension size (rows/cols) for the distance transformer input"
    )
    dist_group.add_argument(
        '--dist_d_model', type=int, default=256,
        help="Embedding dimension for the distance transformer"
    )
    dist_group.add_argument(
        '--dist_n_heads', type=int, default=8,
        help="Number of attention heads in the distance transformer"
    )
    dist_group.add_argument(
        '--dist_n_layers', type=int, default=4,
        help="Number of encoder layers in the distance transformer"
    )

    # Optimization parameters
    parser.add_argument(
        '--epochs', type=int, default=100,
        help="Total number of training epochs"
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help="Training batch size"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        '--optimizer', choices=['adam','adamw'], default='adamw',
        help="Optimizer choice"
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help="Weight decay for the optimizer"
    )
    parser.add_argument(
        '--scheduler', choices=['none','step','cosine'], default='none',
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        '--step_size', type=int, default=10,
        help="Step size (epochs) for StepLR scheduler"
    )
    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help="Gamma value for StepLR scheduler"
    )
    parser.add_argument(
        '--eval_interval', type=int, default=10,
        help="Interval (in epochs) for performing full evaluation"
    )

    parser.add_argument(
        '--seed', type=int, default=42,
        help="Random seed for reproducibility"
    )

    # Logging parameters
    parser.add_argument(
        '--project', type=str, default='AMP-Multimodal',
        help="Weights & Biases project name"
    )
    parser.add_argument(
        '--run_name', type=str, default=None,
        help="Weights & Biases run name (optional)"
    )

    parser.add_argument(
        '--wandb', type=bool, default=False,
        help="Disable Weights & Biases logging"
    )

    parser.add_argument(
        '--model_fold1', type=str, default=None,
        help="Path to the first model checkpoint for ensemble"
    )
    parser.add_argument(
        '--model_fold2', type=str, default=None,
        help="Path to the second model checkpoint for ensemble"
    )

    parser.add_argument(
        '--fold1_file', type=str,
        default="/media/SSD5/sojedaa2/ESCAPE/data/Fold1_reconstructed.csv",
        help="CSV file for Fold 1 (train/val split)"
    )

    parser.add_argument(
        '--fold2_file', type=str,
        default="/media/SSD5/sojedaa2/ESCAPE/data/Fold2_reconstructed.csv",
        help="CSV file for Fold 2 (train/val split)"
    )


    parser.add_argument(
    '--test_file', type=str,
    default="/media/SSD3/sojedaa/sojedaa2/ESCAPE/data/Test_reconstructed.csv",
    help="CSV file for test evaluation"
    )


    return parser.parse_args()