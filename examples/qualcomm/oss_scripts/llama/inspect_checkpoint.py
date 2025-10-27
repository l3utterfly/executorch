import torch
import argparse
import sys

def format_params(num_params: int) -> str:
    """Formats a number into a human-readable string (e.g., 1.75B, 70.5M)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f} B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    else:
        return str(num_params)

def print_state_dict_keys(filepath):
    """
    Loads a .pth file and prints the keys, tensor shapes, and dtypes of its state_dict.

    Args:
        filepath (str): The path to the .pth file.
    """
    try:
        # Load the checkpoint. Using map_location='cpu' is crucial
        # as it prevents GPU memory allocation and potential errors
        # if the model was saved on a different device.
        print(f"[*] Loading checkpoint from: {filepath}")
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        print("[*] Checkpoint loaded successfully.")

        # The state_dict can be the checkpoint itself or nested within it.
        # We'll try to find it intelligently.
        state_dict = None
        
        # Common keys for state_dicts in checkpoints
        possible_keys = ['model', 'state_dict', 'model_state_dict', 'net']

        if isinstance(checkpoint, dict):
            # Check for common keys
            for key in possible_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"[*] Found state_dict under the key: '{key}'")
                    break
            
            # If no common key is found, assume the checkpoint is the state_dict
            if state_dict is None:
                print("[*] No common key found, assuming the entire file is the state_dict.")
                state_dict = checkpoint
        else:
            # If the file is not a dictionary, it's likely the state_dict itself
            print("[*] Loaded file is not a dictionary, assuming it is the state_dict.")
            state_dict = checkpoint

        # Final check to ensure we have a dictionary-like object with keys
        if not hasattr(state_dict, 'keys'):
            print(f"[!] Error: The object loaded is not a state_dict (type: {type(state_dict)}).")
            print("[!] It does not have keys to print.")
            return

        # --- Enhanced Printing Logic ---
        items = list(state_dict.items())
        total_keys = len(items)

        if total_keys == 0:
            print("\n--- State dict is empty ---")
            return
            
        # Calculate padding for aligned printing
        max_key_len = max(len(key) for key, _ in items)
        total_params = 0

        print(f"\n--- Found {total_keys} tensors in the state_dict ---\n")
        print(f"{'Tensor Name':<{max_key_len}}   {'Shape':<25} {'Dtype':<15} {'Parameters'}")
        print(f"{'-' * max_key_len}   {'-' * 25} {'-' * 15} {'-' * 10}")

        for key, tensor in items:
            if isinstance(tensor, torch.Tensor):
                shape_str = str(list(tensor.shape))
                dtype_str = str(tensor.dtype)
                num_params = tensor.numel()
                total_params += num_params
                
                print(f"{key:<{max_key_len}}   {shape_str:<25} {dtype_str:<15} {format_params(num_params)}")
            else:
                # Handle cases where a value might not be a tensor
                print(f"{key:<{max_key_len}}   (Not a tensor, type: {type(tensor).__name__})")
        
        print("\n" + "=" * 80)
        print(f"Total Parameters: {total_params:,} (~{format_params(total_params)})")
        print("=" * 80 + "\n")

    except FileNotFoundError:
        print(f"[!] Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"[!] An error occurred while loading or processing the file: {e}")
        print("[!] The file might be corrupted or not a valid PyTorch checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    # Set up argument parser for command-line usage
    parser = argparse.ArgumentParser(
        description="Inspect a .pth file and print its tensor keys, shapes, dtypes, and total parameters.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the .pth model checkpoint file."
    )

    args = parser.parse_args()
    print_state_dict_keys(args.filepath)