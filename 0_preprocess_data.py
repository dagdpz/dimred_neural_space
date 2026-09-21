import argparse
from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(
    data_dir="data/new_data/flaffus",
    *,
    using_old_data=False,
    sqrt_transform=True,
    zscore=False,
    plot=False,
):
    """Build processed trial DataFrame and save it to disk."""
    data_dir = Path(data_dir)

    df = build_processed_trials(
        data_dir=data_dir,
        using_old_data=using_old_data,
        sqrt_transform=sqrt_transform,
        zscore=zscore,
        plot=plot,
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
        "--sqrt_transform",
        action="store_true",
        default=False,
        help="Apply square-root transform to firing rates.",
    )

    parser.add_argument(
        "--zscore",
        action="store_true",
        default=False,
        help="Apply per-unit z-scoring.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate preprocessing diagnostic plots.",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        using_old_data=args.using_old_data,
        sqrt_transform=args.sqrt_transform,
        zscore=args.zscore,
        plot=args.plot,
    )
