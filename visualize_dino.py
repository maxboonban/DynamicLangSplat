import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def visualize_dino_features(npy_path, output_path=None, method='pca'):
    """
    Visualize DINO features from a .npy file
    
    Args:
        npy_path: Path to the .npy file containing DINO features
        output_path: Where to save the visualization (if None, will display)
        method: 'pca' for PCA visualization, 'first3' for first 3 channels
    """
    # Load the features
    features = np.load(npy_path)
    print(f"Feature shape: {features.shape}")
    
    # Reshape to 2D for PCA if needed
    h, w, c = features.shape
    features_2d = features.reshape(-1, c)
    
    if method == 'pca':
        # Use PCA to reduce to 3 channels
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(features_2d)
        # Reshape back to image
        img = reduced.reshape(h, w, 3)
        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min())
    elif method == 'first3':
        # Take first 3 channels
        img = features[:, :, :3]
        # Normalize to [0,1]
        img = (img - img.min()) / (img.max() - img.min())
    
    # Display or save
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DINO features")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the .npy file")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the visualization")
    parser.add_argument("--method", type=str, default='pca', choices=['pca', 'first3'], 
                       help="Visualization method: 'pca' for PCA visualization, 'first3' for first 3 channels")
    args = parser.parse_args()
    
    visualize_dino_features(args.npy_path, args.output_path, args.method) 