import torch
import argparse
import sys

def print_state_dict_keys(filepath):
    """
    Loads a .pth file and prints the keys of its state_dict.

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

        all_keys = list(state_dict.keys())
        total_keys = len(all_keys)

        print(f"\n--- Found {total_keys} keys in the state_dict ---")
        for key in all_keys:
            print(key)
        print("--------------------------------------------------\n")

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
        description="Inspect a .pth file and print the keys of its state_dict.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the .pth model checkpoint file."
    )

    args = parser.parse_args()
    print_state_dict_keys(args.filepath)