import argparse
import os

def get_size_mb(path):
    bytes_size = os.path.getsize(path)
    return bytes_size / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch checkpoint (.pt) file sizes in MB"
    )
    parser.add_argument("paths", nargs="+", help="Paths to .pt files")
    parser.add_argument("--sort", action="store_true",
                        help="Sort files by size (descending)")
    parser.add_argument("--diff", action="store_true",
                        help="Show size difference relative to first file")
    args = parser.parse_args()

    results = []
    for p in args.paths:
        if not os.path.exists(p):
            print(f"Warning: {p} does not exist, skipping.")
            continue
        size_mb = get_size_mb(p)
        results.append((p, size_mb))

    if not results:
        print("No valid files provided.")
        return

    # Sort if requested
    if args.sort:
        results.sort(key=lambda x: x[1], reverse=True)

    base_size = results[0][1]

    print("\nCheckpoint Sizes:\n")
    for path, size in results:
        line = f"{path}: {size:.2f} MB"
        if args.diff:
            diff = size - base_size
            line += f"  ({diff:+.2f} MB vs first)"
        print(line)

    print()

if __name__ == "__main__":
    main()