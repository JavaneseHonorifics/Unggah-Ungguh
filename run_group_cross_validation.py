import argparse
from pathlib import Path
from src.cross_validation import group_cross_validation


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune JavaneseHonorifics/Unggah-Ungguh classification")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Directory to save outputs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_names",
        nargs='+',
        default=[
            "w11wo/javanese-bert-small-imdb-classifier",
            "w11wo/javanese-gpt2-small-imdb-classifier",
            "w11wo/javanese-distilbert-small-imdb-classifier"
        ],
        help="List of HuggingFace model identifiers"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    group_cross_validation(args)
