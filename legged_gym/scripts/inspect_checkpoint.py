import argparse
import torch
from collections.abc import Mapping

def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")

def describe_tensor(name, tensor, file=None):
    log(f"{name}:", file)
    log(f"  shape: {tuple(tensor.shape)}", file)
    log(f"  dtype: {tensor.dtype}", file)
    try:
        log(f"  min: {tensor.min().item():.4f}", file)
        log(f"  max: {tensor.max().item():.4f}", file)
        log(f"  mean: {tensor.float().mean().item():.4f}", file)
    except Exception:
        log("  stats: unavailable", file)
    log("", file)

def inspect(obj, prefix="", file=None, ignore_keys=None):
    if isinstance(obj, torch.Tensor):
        describe_tensor(prefix, obj, file)

    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k

            # Skip ignored keys (partial match)
            if ignore_keys and any(ik in full_key for ik in ignore_keys):
                continue

            inspect(v, full_key, file, ignore_keys)

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            full_key = f"{prefix}[{i}]"

            if ignore_keys and any(ik in full_key for ik in ignore_keys):
                continue

            inspect(v, full_key, file, ignore_keys)

    else:
        log(f"{prefix}: {type(obj)}", file)

def main():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch .pt checkpoint")
    parser.add_argument("path", help="Path to .pt file")
    parser.add_argument("--keys-only", action="store_true",
                        help="Only print top-level keys")
    parser.add_argument("--device", default="cpu",
                        help="Device to load checkpoint on (default: cpu)")
    parser.add_argument("--save-to", type=str, default=None,
                        help="Path to save inspection output (optional)")
    parser.add_argument(
        "--ignore-keys",
        nargs="*",
        default=[],
        help="List of key substrings to ignore during inspection"
    )
    args = parser.parse_args()

    # Open file if requested
    out_file = open(args.save_to, "w") if args.save_to else None

    try:
        checkpoint = torch.load(args.path, map_location=args.device)

        log(f"\nLoaded: {args.path}", out_file)
        log(f"Type: {type(checkpoint)}\n", out_file)

        if args.keys_only:
            if isinstance(checkpoint, dict):
                log("Top-level keys:", out_file)
                for k in checkpoint.keys():
                    log(f"  {k}", out_file)
            else:
                log("Checkpoint is not a dict.", out_file)
            return

        inspect(checkpoint, file=out_file, ignore_keys=args.ignore_keys)

    finally:
        if out_file:
            out_file.close()

if __name__ == "__main__":
    main()