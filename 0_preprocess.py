from scripts.plotting import *
from scripts.utils import *
from scripts.preprocess import *


def main():
    df = build_processed_trials()
    save_processed_trials(df)


if __name__ == "__main__":
    main()
