import argparse

from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(
    *,
    use_old_data=False,
    sqrt_transform=True,
    zscore=False,
    plot=False,
):
    """Build processed trial DataFrame and save it to disk."""
    df = build_processed_trials(
        use_old_data=args.old_data,
        sqrt_transform=not args.no_sqrt,
        zscore=args.zscore,
        plot=args.plot,
    )
    save_processed_trials(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        use_old_data=args.old_data,
        sqrt_transform=not args.no_sqrt,
        zscore=args.zscore,
        plot=args.plot,
    )
