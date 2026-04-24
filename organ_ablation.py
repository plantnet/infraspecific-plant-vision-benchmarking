import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax

def calculate_all_metrics(df, prob_cols):
    if df.empty:
        return 0.0, 0.0, 0.0
        
    class_ids = np.array([int(c.split('_')[-1]) for c in prob_cols])
    
    # 1. Image Top-1
    img_pred_indices = df[prob_cols].values.argmax(axis=1)
    img_preds = class_ids[img_pred_indices]
    img_top1_acc = np.mean(df['true_class_id'].values == img_preds) * 100
    
    # Agrégation par observation
    obs_df = df.groupby(['observation_id', 'true_class_id'])[prob_cols].mean().reset_index()
    
    # 2. Observation Top-1
    obs_pred_indices = obs_df[prob_cols].values.argmax(axis=1)
    obs_preds = class_ids[obs_pred_indices]
    obs_df['is_correct'] = (obs_df['true_class_id'].values == obs_preds)
    obs_top1_acc = obs_df['is_correct'].mean() * 100
    
    # 3. Macro Observation Top-1
    macro_obs_top1_acc = obs_df.groupby('true_class_id')['is_correct'].mean().mean() * 100
    
    return img_top1_acc, obs_top1_acc, macro_obs_top1_acc

def main(logits_path, labels_path, metadata_csv):
    logits = np.load(logits_path)
    labels = np.load(labels_path)
    probs = softmax(logits, axis=1)
    
    # Chargement avec détection de l'absence potentielle de header
    df_check = pd.read_csv(metadata_csv, nrows=0)
    
    if 'observation_id' not in df_check.columns and 'organ' not in df_check.columns:
        df = pd.read_csv(metadata_csv, header=None)
        # Mapping spécifique (index 1 -> observation_id, index 3 -> organ)
        if len(df.columns) > 3:
            df.rename(columns={1: 'observation_id', 3: 'organ'}, inplace=True)
    else:
        df = pd.read_csv(metadata_csv)
    
    # Fallback si observation_id reste introuvable
    if 'observation_id' not in df.columns:
        if 'image_name' in df.columns:
            df['observation_id'] = df['image_name']
        elif 'image_path' in df.columns:
            df['observation_id'] = df['image_path']
        else:
            df['observation_id'] = df.index

    # Arrêt si organ est introuvable
    if 'organ' not in df.columns:
        print(f"⚠️ 'organ' introuvable dans {metadata_csv}. Ablation impossible pour ce dataset.")
        return
        
    if len(df) != len(logits):
        raise ValueError(f"Erreur d'alignement: {len(df)} images dans le CSV vs {len(logits)} prédictions numpy.")
    
    df['true_class_id'] = labels
    num_classes = probs.shape[1]
    prob_cols = []
    
    for i in range(num_classes):
        col_name = f'prob_class_{i}'
        df[col_name] = probs[:, i]
        prob_cols.append(col_name)
        
    organs = df['organ'].dropna().unique()
    
    # Calcul Baseline
    base_img, base_obs, base_macro = calculate_all_metrics(df, prob_cols)
    print(f"\n{'='*90}")
    print("BASELINE (All Organs)")
    print(f"{'='*90}")
    print(f"Image Top-1:       {base_img:.2f}%")
    print(f"Observation Top-1: {base_obs:.2f}%")
    print(f"Macro Obs Top-1:   {base_macro:.2f}%\n")
    
    print(f"{'='*90}")
    print(f"{'Ablated Organ':<15} | {'Image Top-1 (%)':<22} | {'Observation Top-1 (%)':<22} | {'Macro Obs Top-1 (%)':<22}")
    print("-" * 90)
    
    # Calcul Ablation
    for organ in organs:
        df_ablated = df[df['organ'] != organ]
        
        abl_img, abl_obs, abl_macro = calculate_all_metrics(df_ablated, prob_cols)
        
        # Calcul du Delta (Nouveau - Ancien)
        delta_img = abl_img - base_img
        delta_obs = abl_obs - base_obs
        delta_macro = abl_macro - base_macro
        
        # Formatage de la chaîne : Score (Delta)
        str_img = f"{abl_img:>5.2f} ({delta_img:+.2f})"
        str_obs = f"{abl_obs:>5.2f} ({delta_obs:+.2f})"
        str_macro = f"{abl_macro:>5.2f} ({delta_macro:+.2f})"
        
        print(f"{organ:<15} | {str_img:<22} | {str_obs:<22} | {str_macro:<22}")
    print(f"{'='*90}\n")

    print(f"\n{'='*115}")
    print("SINGLE-ORGAN EVALUATION (Intrinsic Discriminative Power)")
    print(f"{'='*115}")
    print(f"{'Isolated Organ':<15} | {'Image Top-1 (%)':<30} | {'Observation Top-1 (%)':<30} | {'Macro Obs Top-1 (%)':<30}")
    print("-" * 115)
    
    for organ in organs:
        # On garde UNIQUEMENT les images de l'organe ciblé
        df_isolated = df[df['organ'] == organ]
        
        iso_img, iso_obs, iso_macro = calculate_all_metrics(df_isolated, prob_cols)
        
        # Calcul du Delta (Nouveau - Ancien)
        delta_img = iso_img - base_img
        delta_obs = iso_obs - base_obs
        delta_macro = iso_macro - base_macro
        
        # Comptage pour le contexte statistique
        img_count = len(df_isolated)
        obs_count = df_isolated['observation_id'].nunique()
        
        # Formatage avec le score, le delta et le nombre d'échantillons (n)
        str_img = f"{iso_img:>5.2f} ({delta_img:+.2f}) [n={img_count}]"
        str_obs = f"{iso_obs:>5.2f} ({delta_obs:+.2f}) [n={obs_count}]"
        str_macro = f"{iso_macro:>5.2f} ({delta_macro:+.2f})"
        
        print(f"{organ:<15} | {str_img:<30} | {str_obs:<30} | {str_macro:<30}")
    print(f"{'='*115}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    args = parser.parse_args()
    
    main(args.logits, args.labels, args.metadata)