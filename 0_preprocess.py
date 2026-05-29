from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main(normalize=True):
    df = build_processed_trials(normalize=normalize)
    save_processed_trials(df, normalize=normalize)


if __name__ == "__main__":
    main(normalize=False)
