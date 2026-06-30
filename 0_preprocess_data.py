import argparse

from scipy.sparse import data

from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(
    data_dir="data/new_data/flaffus",
    *,
    use_old_data=False,
    old_data_dir="data/old_data",
    sqrt_transform=True,
    zscore=False,
    plot=False,
):
    """Build processed trial DataFrame and save it to disk."""
    data_dir = Path(data_dir)
    old_data_dir = Path(old_data_dir)

    df = build_processed_trials(
        data_dir=data_dir,
        use_old_data=use_old_data,
        old_data_dir=old_data_dir,
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
        "--old_data_dir",
        type=str,
        default="data/old_data",
        help="Folder containing old-format population_*.mat and trials_*.mat files.",
    )

    parser.add_argument(
        "--old_data",
        action="store_true",
        default=False,
        help="Use old-format data from data/old_data.",
    )

    parser.add_argument(
        "--no_sqrt",
        action="store_true",
        default=False,
        help="Do not apply square-root transform to firing rates.",
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
        use_old_data=args.old_data,
        old_data_dir=args.old_data_dir,
        sqrt_transform=not args.no_sqrt,
        zscore=args.zscore,
        plot=args.plot,
    )
