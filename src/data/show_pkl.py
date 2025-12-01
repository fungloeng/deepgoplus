#!/usr/bin/env python3
# show_pkl.py
import pandas as pd
import argparse
import os

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quick peek at the first N rows of a pandas pickle file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the .pkl file to inspect")
    parser.add_argument("-n", "--rows", type=int, default=5, help="Number of rows to display (default: 5)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file not found -> {args.file}")
        exit(1)

    df = pd.read_pickle(args.file)

    print(f"--- first {args.rows} rows / total {len(df)} rows ---")
    print(df.head(args.rows))
    print("\n--- column info ---")
    print(df.info())

if __name__ == "__main__":
    main()