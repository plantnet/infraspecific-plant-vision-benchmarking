import os
import pandas as pd
import shutil
from pathlib import Path

def create_mini_dataset(source_dir, target_dir, label_col='class_name', n_per_group=10):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # 1. Création de l'arborescence
    print(f"📁 Initialisation du mini-dataset ({n_per_group} images max/groupe) vers : {target_dir}")
    (target_dir / "images").mkdir(parents=True, exist_ok=True)

    # 2. Copie du mapping (Source de vérité pour le benchmarking)
    if (source_dir / "class_mapping.txt").exists():
        shutil.copy(source_dir / "class_mapping.txt", target_dir / "class_mapping.txt")
        print("✅ class_mapping.txt copié.")

    # 3. Traitement des CSV
    for csv_name in ['train.csv', 'test.csv']:
        csv_path = source_dir / csv_name
        if not csv_path.exists():
            continue
            
        print(f"📄 Traitement de {csv_name}...")
        df = pd.read_csv(csv_path)
        
        # Détection automatique de la colonne du nom de fichier
        file_col = 'image_name' if 'image_name' in df.columns else 'file_name'
        if file_col not in df.columns:
            print(f"❌ Erreur : Colonne de fichier non trouvée dans {csv_name}")
            continue

        # Détermination des groupes d'échantillonnage
        group_cols = [label_col]
        if 'learn_tag' in df.columns:
            group_cols.append('learn_tag')
            
        # Échantillonnage : N images par groupe (Variété + Split)
        # On utilise head(n_per_group) pour prendre jusqu'à 10 images si elles existent
        mini_df = df.groupby(group_cols).head(n_per_group).copy()
        
        # 4. Copie physique des images
        copy_count = 0
        for _, row in mini_df.iterrows():
            file_name = str(row[file_col]).strip()
            cls_name = str(row[label_col]).strip()
            
            # Détermination du split pour le dossier (train/val/test)
            split = row['learn_tag'] if 'learn_tag' in df.columns else 'test'
            
            # Chemins possibles selon l'organisation du dossier images
            possible_src_paths = [
                source_dir / "images" / split / cls_name / file_name,
                source_dir / "images" / split / file_name,
                source_dir / "images" / cls_name / file_name
            ]
            
            src_path = None
            for p in possible_src_paths:
                if p.exists():
                    src_path = p
                    break
                
            if src_path:
                dest_path = target_dir / "images" / split / cls_name / file_name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if not dest_path.exists():
                    shutil.copy(src_path, dest_path)
                copy_count += 1

        # Sauvegarde du nouveau CSV
        mini_df.to_csv(target_dir / csv_name, index=False)
        print(f"✅ {csv_name} généré ({len(mini_df)} lignes au total, {copy_count} images copiées).")

if __name__ == "__main__":
    # Définition des datasets à traiter en mini
    datasets = ["banana", "plant-market-garden"]
    
    for ds in datasets:
        create_mini_dataset(
            source_dir=f"/fastdata/datasets/tibi/zenodo/{ds}",
            target_dir=f"/fastdata/datasets/tibi/zenodo/mini-{ds}",
            label_col="class_name",
            n_per_group=10
        )