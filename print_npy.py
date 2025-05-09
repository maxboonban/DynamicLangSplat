import numpy as np
import sys

def print_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Print basic information about the array
        print(f"\nFile: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Number of elements: {data.size}")
        
        # Print the actual data
        print("\nData contents:")
        print(data)
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_npy.py <path_to_npy_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print_npy_file(file_path) 