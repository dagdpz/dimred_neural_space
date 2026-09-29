import argparse
from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(
    data_dir="data/new_data/flaffus",
    *,
    transform="sqrt",
    zscore=False,
):
    """Build processed trial DataFrame and save it to disk."""
    data_dir = Path(data_dir)

    df = build_processed_trials(
        data_dir=data_dir,
        transform=transform,
        zscore=zscore,
    )

    save_processed_trials(df, path=data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/new_data/flaffus",
        help="Folder containing new-format population_*.mat and trials_*.mat files.",
    )
    parser.add_argument(
        "--using_old_data",
        action="store_true",
        default=False,
        help="Use old-format data from data/old_data.",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="sqrt",
        help="Apply square-root transform to firing rates.",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        default=False,
        help="Apply per-unit z-scoring.",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        transform=args.transform,
        zscore=args.zscore,
    )
