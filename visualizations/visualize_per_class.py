import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import argparse

def plot_accuracy_vs_training_size(csv_path, output_path, x_scale='linear'):
    """
    Génère un nuage de points : Précision vs Taille du jeu d'entraînement.
    Force les axes de l'échelle logarithmique à être strictement identiques 
    entre tous les graphiques (min, max, et graduations figés).
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Nettoyage
    df = df.dropna(subset=['Train_Images', 'Observation_Top1_Acc']).copy()
    
    # Configuration de la police type LaTeX
    plt.rcParams.update({
        'font.size': 11, 
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']
    })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = df['Train_Images']
    y = df['Observation_Top1_Acc']
    
    # Nuage de points global
    ax.scatter(x, y, alpha=0.5, c='#7fb2d5', edgecolors='gray', s=40, linewidth=0.5, zorder=2)
    
    # --- 1. Calcul des 4 profils extrêmes ---
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min if x_max > x_min else 1
    
    x_norm = (x - x_min) / x_range
    y_norm = y / 100.0

    idx_bad = np.sqrt(x_norm**2 + y_norm**2).idxmin()                 # Low N, Low Acc
    idx_surprise = np.sqrt(x_norm**2 + (1 - y_norm)**2).idxmin()      # Low N, High Acc
    idx_disappoint = np.sqrt((1 - x_norm)**2 + y_norm**2).idxmin()    # High N, Low Acc
    idx_good = np.sqrt((1 - x_norm)**2 + (1 - y_norm)**2).idxmin()    # High N, High Acc

    # --- 2. Configuration et placement explicite en ESCALIER ---
    y_above_grid_1 = 106 
    y_below_grid_1 = -6 
    y_above_grid_2 = 114 
    y_below_grid_2 = -14 
    
    color_good = 'tab:blue'
    color_disappoint = 'tab:orange'
    color_surprise = 'tab:green'
    color_bad = 'tab:red'
    
    configs = {
        'bad': {
            'idx': idx_bad, 'color': color_bad, 
            'x_text': df.loc[idx_bad, 'Train_Images'], 
            'y_text': y_below_grid_1, 'va': 'top'
        },
        'surprise': {
            'idx': idx_surprise, 'color': color_surprise, 
            'x_text': df.loc[idx_surprise, 'Train_Images'], 
            'y_text': y_above_grid_1, 'va': 'bottom'
        },
        'disappoint': {
            'idx': idx_disappoint, 'color': color_disappoint, 
            'x_text': df.loc[idx_disappoint, 'Train_Images'], 
            'y_text': y_below_grid_2, 'va': 'top'
        },
        'good': {
            'idx': idx_good, 'color': color_good, 
            'x_text': df.loc[idx_good, 'Train_Images'], 
            'y_text': y_above_grid_2, 'va': 'bottom'
        }
    }

    # --- 3. Tracé des points spécifiques et annotations ---
    plotted_indices = set()
    for key, cfg in configs.items():
        idx = cfg['idx']
        if idx in plotted_indices: continue
        plotted_indices.add(idx)
            
        row = df.loc[idx]
        c_name = str(row['Class_Name']).replace('_', ' ')
        
        ax.scatter(row['Train_Images'], row['Observation_Top1_Acc'], 
                   color=cfg['color'], edgecolors='black', s=90, zorder=3, linewidth=1)
        
        ax.annotate(c_name, 
                    (row['Train_Images'], row['Observation_Top1_Acc']),
                    xytext=(cfg['x_text'], cfg['y_text']), textcoords='data', 
                    fontsize=10, color=cfg['color'], fontweight='bold',
                    ha='center', va=cfg['va'], zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.8))

    # --- 4. Légende factuelle ---
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_good, markersize=9, 
                      label="High Sample Size, High Accuracy"),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_surprise, markersize=9, 
                      label="Low Sample Size, High Accuracy"),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_disappoint, markersize=9, 
                      label="High Sample Size, Low Accuracy"),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bad, markersize=9, 
                      label="Low Sample Size, Low Accuracy")
    ]
    ax.legend(handles=legend_elements, loc='lower right', title="Class Performance Profiles", 
              fontsize=9, title_fontsize=10, framealpha=0.9, edgecolor='gray')

    # --- 5. Formatage des axes ---
    ax.set_ylabel("Observation Top-1 Accuracy (%)", fontweight='bold')
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-20, 120) 
    
    # Application de l'échelle et calcul des marges
    if x_scale == 'log':
        ax.set_xlabel("Number of Training Images (Log Scale)", fontweight='bold')
        ax.set_xscale('log')
        
        # Forçage strict des ticks pour la comparabilité
        ax.set_xticks([1, 10, 100, 1000])
        
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        
        # Forçage strict des limites : identique pour tous les graphes
        ax.set_xlim(0.5, 2000)
    else:
        ax.set_xlabel("Number of Training Images", fontweight='bold')
        ax.set_xlim(x_min - (x_range * 0.15), x_max + (x_range * 0.15))
    
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help="Directory containing per_class_metrics.csv")
    parser.add_argument('--dataset-name', type=str, default="", help="Ignored")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "per_class_metrics.csv")

    out_linear = os.path.join(args.results_dir, f"{args.dataset_name}_per_class_accuracy_vs_size_linear.pdf")
    out_log = os.path.join(args.results_dir, f"{args.dataset_name}_per_class_accuracy_vs_size_log.pdf")
    
    plot_accuracy_vs_training_size(csv_path, out_linear, x_scale='linear')
    plot_accuracy_vs_training_size(csv_path, out_log, x_scale='log')

if __name__ == "__main__":
    main()