import argparse
import sys
import os
import json
import math
from safetensors import safe_open

def format_params(num_params: int) -> str:
    """Formats a number into a human-readable string (e.g., 1.75B, 70.5M)."""
    if num_params == 0:
        return "0"
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f} B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    else:
        return str(num_params)

def print_safetensors_keys(filepath):
    """
    Loads a .safetensors file or a sharded checkpoint directory and prints the tensor keys,
    shapes, dtypes, and total parameters.

    Args:
        filepath (str): The path to a .safetensors file or a directory
                        containing a model.safetensors.index.json file.
    """
    tensor_info_list = []
    try:
        # --- Case 1: Handle sharded checkpoints (directory) ---
        if os.path.isdir(filepath):
            index_path = os.path.join(filepath, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                print(f"[!] Error: Directory provided but 'model.safetensors.index.json' not found in '{filepath}'.")
                sys.exit(1)
            
            print(f"[*] Found sharded model index: {index_path}")
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            if "metadata" in index_data and "weight_map" in index_data:
                metadata = index_data["metadata"]
                # Sort keys for consistent output
                sorted_keys = sorted(index_data["weight_map"].keys())
                for key in sorted_keys:
                    if key in metadata:
                        info = metadata[key]
                        tensor_info_list.append({
                            "key": key,
                            "shape": info.get("shape", "N/A"),
                            "dtype": info.get("dtype", "N/A")
                        })
                print(f"[*] Successfully extracted metadata for {len(tensor_info_list)} tensors from index file.")
            else:
                print("[!] Error: 'metadata' or 'weight_map' not found in the index JSON file.")
                sys.exit(1)

        # --- Case 2: Handle a single .safetensors file ---
        elif os.path.isfile(filepath):
            if not filepath.endswith('.safetensors'):
                print(f"[!] Warning: File does not have a .safetensors extension: '{filepath}'")

            print(f"[*] Loading single checkpoint from: {filepath}")
            with safe_open(filepath, framework="pt", device="cpu") as f:
                # Iterate through sorted keys for consistent output
                for key in sorted(f.keys()):
                    info = f.get_tensor_info(key)
                    tensor_info_list.append({
                        "key": key,
                        "shape": info.shape,
                        "dtype": str(info.dtype) # Convert torch.dtype to string
                    })
            print("[*] Checkpoint metadata loaded successfully.")
        
        else:
            raise FileNotFoundError

        # --- Enhanced Printing Logic ---
        total_keys = len(tensor_info_list)
        if total_keys == 0:
            print("\n--- No tensors found in the checkpoint. ---")
            return
            
        max_key_len = max(len(info["key"]) for info in tensor_info_list)
        total_params = 0

        print(f"\n--- Found {total_keys} tensors in the checkpoint ---\n")
        print(f"{'Tensor Name':<{max_key_len}}   {'Shape':<25} {'Dtype':<15} {'Parameters'}")
        print(f"{'-' * max_key_len}   {'-' * 25} {'-' * 15} {'-' * 10}")

        for info in tensor_info_list:
            key = info["key"]
            shape = info["shape"]
            dtype = info["dtype"]
            
            shape_str = str(shape)
            num_params = math.prod(shape) if isinstance(shape, list) or isinstance(shape, tuple) else 0
            total_params += num_params
            
            print(f"{key:<{max_key_len}}   {shape_str:<25} {dtype:<15} {format_params(num_params)}")
        
        print("\n" + "=" * 80)
        print(f"Total Parameters: {total_params:,} (~{format_params(total_params)})")
        print("=" * 80 + "\n")

    except FileNotFoundError:
        print(f"[!] Error: File or directory not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")
        print("[!] The file might be corrupted, not a valid safetensors file, or the directory structure is incorrect.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a .safetensors file or sharded checkpoint directory and print its tensor keys, shapes, dtypes, and total parameters.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the .safetensors file or the directory containing the sharded model."
    )

    args = parser.parse_args()
    print_safetensors_keys(args.filepath)