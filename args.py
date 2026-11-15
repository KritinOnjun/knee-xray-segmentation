import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Knee Segmentation Project')

    parser.add_argument("-csv_dir", default="data/CSVs/")  # FIX: CSVs, not CSVS

    parser.add_argument("-out_dir", default="sessions")

    # FIX: type=int / type=float so numbers from the terminal are numbers, not text
    parser.add_argument("-epochs", type=int, default=30)

    parser.add_argument("-batch_size", type=int, default=16)

    parser.add_argument("-lr", type=float, default=1e-3, choices=[1e-2, 1e-3, 1e-4, 1e-5])

    parser.add_argument("-wd", type=float, default=1e-5, choices=[1e-5, 1e-7])

    args = parser.parse_args()
    return args
