import argparse

from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(*, normalize=False, plot=False):
    """Build processed trial DataFrame and save it to disk."""
    df = build_processed_trials(
        normalize=normalize,
        plot=plot,
    )
    save_processed_trials(df, normalize=normalize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Save sqrt-transformed, z-scored firing rates.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Generate preprocessing plots.",
    )
    args = parser.parse_args()
    main(
        normalize=args.normalize,
        plot=args.plot,
    )
