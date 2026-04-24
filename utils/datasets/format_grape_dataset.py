import pandas as pd
import os
import argparse

def process_grape_csv(train_src_path, test_src_path, dir_dest):
    """
    Harmonise les fichiers CSV du dataset Grape Variety.
    Compatible avec dataset.py (utilise 'id' pour l'observation).
    """
    
    
    if not os.path.exists(train_src_path) or not os.path.exists(test_src_path):
        print(f"❌ Erreur : Fichiers introuvables dans {train_src_path, test_src_path}")
        print(train_src_path, test_src_path)
        return

    os.makedirs(dir_dest, exist_ok=True)
    #print(f"🔄 Traitement : {dir_src} -> {dir_dest}")

    try:
        df_train = pd.read_csv(train_src_path, sep=';')
        df_test = pd.read_csv(test_src_path, sep=';')
    except Exception as e:
        print(f"❌ Erreur CSV : {e}")
        return

    def standardize_df(df):
        # Mapping strict pour compatibilité dataset.py
        rename_map = {
            'variety': 'class_name',
            'fake_obs': 'id',             # <--- COMPATIBILITÉ ASSURÉE
            'image_name': 'image_name',
            # 'local_path' ignoré
        }
        df = df.rename(columns=rename_map)

        if 'organ' in df.columns:
            df['organ'] = df['organ'].replace({
                'feuille': 'leaf',
                'grappe': 'fruit',
                'rameau': 'branch',
                'souche': 'trunk'
            })

        if 'image_name' in df.columns:
            df['image_name'] = df['image_name'].astype(str).str.strip()

        target_cols = ['image_name', 'id', 'organ', 'class_name', 'learn_tag']
        cols_to_keep = [c for c in target_cols if c in df.columns]
        print(cols_to_keep)
        return df[cols_to_keep]

    df_train_clean = standardize_df(df_train)
    df_test_clean = standardize_df(df_test)

    out_train = os.path.join(dir_dest, 'train.csv')
    out_test = os.path.join(dir_dest, 'test.csv')

    df_train_clean.to_csv(out_train, index=False)
    df_test_clean.to_csv(out_test, index=False)

    print("-" * 40)
    print(f"✅ Fichiers générés avec header compatible (id) :")
    print(f"   Structure : {list(df_train.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_val_src', type=str, required=True)
    parser.add_argument('--test_src', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()
    process_grape_csv(args.train_val_src, args.test_src, args.dest)