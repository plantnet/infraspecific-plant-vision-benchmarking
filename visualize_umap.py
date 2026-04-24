import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import umap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test',
                        help='Which data split to visualize')
    args = parser.parse_args()

    # 1. Load the requested split
    feat_path = os.path.join(args.results_dir, f'{args.split}_feats.npy')
    label_path = os.path.join(args.results_dir, f'{args.split}_labels.npy')
    
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} not found. Run test.py first.")
        return

    features = np.load(feat_path)
    labels = np.load(label_path)
    with open(os.path.join(args.results_dir, 'classes.txt'), 'r') as f:
        classes = f.read().splitlines()

    # 2. Run UMAP
    print(f"Running UMAP on {args.split} features ({features.shape[0]} samples)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(features)

    # 3. Plot
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(classes):
        mask = labels == i
        if np.any(mask):
            plt.scatter(embedding[mask, 0], embedding[mask, 1], label=class_name, alpha=0.6, s=40)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.title(f"UMAP Projection: {args.split.upper()} set (DinoV2 features)")
    plt.tight_layout()
    
    output_path = os.path.join(args.results_dir, f'umap_{args.split}.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()