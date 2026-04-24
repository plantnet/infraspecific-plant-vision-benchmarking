import os
import json
import pandas as pd
import argparse

def get_model_and_history(folder_name, full_path):
    fn = folder_name.lower()
    path = full_path.lower()
    
    # --- 1. CNNs (ResNet, ConvNeXt) ---
    if "resnet50" in fn: return "resnet50", "pt(a1_in1k)"
    if "convnext" in fn: return "convnext_base", "pt(in22k)_ft(in1k)"
    
    # --- 2. BioCLIP ---
    if "bioclip" in fn: return "bioclip_v2", "TreeOfLife-200M"

    # --- 3. VISION TRANSFORMERS (ViT) ---
    if "vit_base" in fn or "vit_large" in fn:
        
        # A. DINOv3
        if "dinov3" in fn:
            base_hist = "ssl_pt(lvd1689m)"
            if "vit_large" in fn:
                 model_clean = "vit_large_patch16_dinov3.lvd1689m"
            else:
                 model_clean = "vit_base_patch16_dinov3.lvd1689m"
            return model_clean, base_hist

        # B. DINOv2 Large
        if "dinov2" in fn and "vit_large" in fn:
            base_hist = "ssl_pt(lvd142m)"
            model_clean = "vit_large_patch14_dinov2.lvd142m"
            return model_clean, base_hist

        # C. ViT Supervisé (AugReg) - Baseline
        if "augreg" in fn:
            base_hist = "pt(in21k)_ft(in1k)"
            if "384" in fn:
                 model_clean = "vit_base_patch16_384.augreg_in21k_ft_in1k"
            else:
                 model_clean = "vit_base_patch16_224.augreg_in21k_ft_in1k"
            return model_clean, base_hist

        # D. DINOv2 Base et ses dérivés
        base_hist = "ssl_pt(lvd142m)"
        model_clean = "vit_base_patch14_dinov2" # Par défaut
        
        # Cas spécifique : Architecture Reg4
        if "reg4" in fn:
            model_clean = "vit_base_patch14_reg4_dinov2"
            
            # 1. Pl@ntNet 14M SSL
            if "pn170125" in fn or "dinov2sslft" in path:
                return model_clean, f"{base_hist}_sslft(Pl@ntNet14M)"
            
            # 2. PlantCLEF 2024 (Fallback absolu pour les reg4 restants)
            else:
                return model_clean, f"{base_hist}_ft(PlantCLEF2024_1.4M)"
        
        # DÉTECTION DE L'HISTORIQUE (Pour les modèles standard sans reg4)
        if "pn" in fn or "plantnet" in path:
            return model_clean, f"{base_hist}_ft(Pl@ntNet16M)"
        
        if "pc22" in fn:
             return model_clean, f"{base_hist}_sslft(PlantCLEF2022)"

        # 5. DINOv2 Base Generic
        return model_clean, base_hist

    return folder_name, "UNKNOWN"
    

def safe_get(d, keys, default=None):
    for key in keys:
        if d is None or not isinstance(d, dict):
            return default
        d = d.get(key)
    return d if d is not None else default

def collect_results(results_dir, output_file):
    all_rows = []
    
    print(f"🔍 Scanning results in: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"❌ Error: Directory not found: {results_dir}")
        return

    for root, dirs, files in os.walk(results_dir):
        if 'test_metrics.json' in files:
            dataset = os.path.basename(os.path.dirname(root))
            folder_name = os.path.basename(root)
            
            try:
                with open(os.path.join(root, 'test_metrics.json'), 'r') as f:
                    m = json.load(f)
                
                model_clean, history = get_model_and_history(folder_name, root)
                mode = "frozen" if "frozen" in folder_name.lower() else "full"
                
                res = "???"
                if "_res" in folder_name:
                    res = folder_name.split('_res')[1].split('_')[0]

                row = {
                    'Dataset': dataset,
                    'Model': model_clean,
                    'History': history,
                    'Mode': mode,
                    'Res': res,
                    
                    'LP_Obs_T1': safe_get(m, ['linear_probing', 'observation', 'top1']),
                    'LP_Obs_Macro': safe_get(m, ['linear_probing', 'observation', 'macro']),
                    'kNN_Obs_T1': safe_get(m, ['knn', 'observation', 'top1']),
                    'kNN_Obs_Macro': safe_get(m, ['knn', 'observation', 'macro']),
                    
                    'LP_Img_T1': safe_get(m, ['linear_probing', 'image', 'top1']),
                    'LP_Img_Macro': safe_get(m, ['linear_probing', 'image', 'macro']),
                    'kNN_Img_T1': safe_get(m, ['knn', 'image', 'top1']),
                    'kNN_Img_Macro': safe_get(m, ['knn', 'image', 'macro']),
                    
                    'Folder': folder_name 
                }
                all_rows.append(row)
            except Exception as e:
                print(f"⚠️ Erreur: {e}")

    if not all_rows:
        print("❌ Aucun résultat trouvé.")
        return

    df = pd.DataFrame(all_rows)
    
    cols_order = [
        'Dataset', 'Model', 'History', 'Mode', 'Res', 
        'LP_Obs_Macro', 'LP_Obs_T1', 'LP_Img_Macro', 'LP_Img_T1',
        'kNN_Obs_Macro', 'kNN_Obs_T1', 'kNN_Img_Macro',  'kNN_Img_T1'
    ]
    cols_final = [c for c in cols_order if c in df.columns]
    df = df[cols_final]

    df = df.sort_values(by=['Dataset', 'Model', 'History', 'Mode'], ascending=[True, True, True, True])
    df = df.round(2)

    output_path = os.path.join(results_dir, output_file)
    df.to_csv(output_path, index=False)
    
    print("-" * 100)
    print(f"✅ Synthèse terminée : {len(df)} lignes.")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print("\n📊 --- APERÇU COMPLET ---")
    print(df.head(1000).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--output-file', type=str, default='final_benchmarking_results.csv')
    args = parser.parse_args()
    
    collect_results(args.results_dir, args.output_file)