import os
import pandas as pd
import shutil
from tqdm import tqdm
import re

MODEL_TO_AUTHOR = {
    "Pixel 6a": "Charlotte Tibi",
    "iPhone 14": "Pierre Bonnet",
    "SM-G980F": "Mathias Chouet",
    "ILCE-7RM3": "Hervé Goëau",
    "Pixel 7": "Hervé Goëau",
    "Unknown": "AMAP Team"
}

# --- 1. FONCTIONS UTILITAIRES ---

def sanitize_filename(name):
    """Nettoie une chaîne pour l'utiliser dans un nom de fichier."""
    name = str(name).lower()
    name = re.sub(r'[àáâãäå]', 'a', name)
    name = re.sub(r'[èéêë]', 'e', name)
    name = re.sub(r'[ìíîï]', 'i', name)
    name = re.sub(r'[òóôõö]', 'o', name)
    name = re.sub(r'[ùúûü]', 'u', name)
    name = re.sub(r'[ç]', 'c', name)
    name = re.sub(r'[^a-z0-9\.]', '_', name) 
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def load_inaturalist_metadata(external_csv_path):
    """
    Charge le fichier d'observations iNaturalist.
    Récupère l'auteur (Nom > Login) et la licence.
    """
    print(f"🔍 Chargement des métadonnées iNaturalist depuis : {external_csv_path}")
    if not os.path.exists(external_csv_path):
        print("⚠️ Fichier iNaturalist introuvable !")
        return {}
    
    try:
        df = pd.read_csv(external_csv_path, sep=',')
        metadata_map = {}
        
        for _, row in df.iterrows():
            obs_id = str(row['id']).replace('.0', '').strip()
            
            # Sélection de l'auteur : Nom Complet > Login
            full_name = str(row['user_name']).strip() if pd.notna(row['user_name']) and str(row['user_name']).strip() != "" else ""
            login = str(row['user_login']).strip() if pd.notna(row['user_login']) else "unknown"
            
            metadata_map[obs_id] = {
                'license': str(row['license']).lower() if pd.notna(row['license']) else "all rights reserved",
                'author': full_name if full_name else login,
                'date': str(row['observed_on']) if pd.notna(row['observed_on']) else "",
                'latitude': str(row['latitude']) if pd.notna(row['latitude']) else "",
                'longitude': str(row['longitude']) if pd.notna(row['longitude']) else "",
                'source_url': str(row['url']) if pd.notna(row['url']) else ""
            }
            
        print(f"✅ {len(metadata_map)} observations iNaturalist chargées.")
        return metadata_map

    except Exception as e:
        print(f"❌ Erreur lecture fichier métadonnées : {e}")
        return {}

def load_camera_mapping(exif_file_path):
    """Charge le fichier généré par exiftool (Model \t FileName)."""
    mapping = {}
    if os.path.exists(exif_file_path):
        with open(exif_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    model, fname = parts
                    mapping[fname] = model
    return mapping

def load_grape_metadata(csv_path):
    print(f"🔍 Chargement Grape Metadata : {csv_path}")
    if not os.path.exists(csv_path): return {}
    # Lecture avec séparateur point-virgule
    df = pd.read_csv(csv_path, sep=';', quotechar='"')
    metadata = {}
    for _, row in df.iterrows():
        img_name = str(row['image_name']).strip()
        metadata[img_name] = {
            'author': str(row['author']),
            'date': str(row['date']),
            'location': str(row['location']),
            'organ': str(row['organ'])
        }
    return metadata

def parse_exif_date(date_str):
    """Nettoie le format EXIF (2012:06:27) en format standard (2012-06-27)."""
    if not date_str or date_str == "-" or date_str == "0000:00:00 00:00:00":
        return ""
    # On garde juste la date YYYY:MM:DD
    match = re.match(r'(\d{4}):(\d{2}):(\d{2})', date_str)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return date_str

def load_grape_test_dates(exif_date_file):
    """Charge et sélectionne la meilleure date disponible pour le test set Grape."""
    date_mapping = {}
    if os.path.exists(exif_date_file):
        with open(exif_date_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    fname = parts[0]
                    # Stratégie de sélection : DateTimeOriginal > CreateDate > ModifyDate
                    best_date = ""
                    for i in range(1, len(parts)):
                        val = parse_exif_date(parts[i])
                        if val:
                            best_date = val
                            break
                    date_mapping[fname] = best_date
    return date_mapping
    
def generate_class_mapping(df, output_dir):
    """Génère class_mapping.txt"""
    if 'class_name' not in df.columns: return
    classes = sorted(df['class_name'].unique().astype(str))
    with open(os.path.join(output_dir, 'class_mapping.txt'), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"    🏷️  class_mapping.txt généré ({len(classes)} classes).")

def format_date_iso(date_str):
    """
    Transforme les formats DD/MM/YYYY, DD-MM-YYYY, AAAA:MM:JJ en AAAA-MM-JJ
    sans déclencher de UserWarning pandas.
    """
    if not date_str or str(date_str).lower() in ['nan', 'none', '-', '']:
        return ""
    
    # Nettoyage : on ne garde que la partie date si une heure est présente
    date_clean = str(date_str).strip().split(' ')[0]
    # Standardisation du séparateur
    date_clean = date_clean.replace(':', '-').replace('/', '-')
    
    try:
        # Si la date commence déjà par l'année (ex: 2019-04-17)
        if re.match(r'^\d{4}-', date_clean):
            dt = pd.to_datetime(date_clean, errors='coerce') # Pas de dayfirst
        # Sinon (ex: 28-11-2022 ou 22/06/2012)
        else:
            dt = pd.to_datetime(date_clean, dayfirst=True, errors='coerce')
            
        if pd.isna(dt):
            return ""
        return dt.strftime('%Y-%m-%d')
    except:
        return ""
        
# --- 2. CŒUR DU SCRIPT DE MIGRATION ---

def process_dataset_migration(dataset_name, src_root, csv_train_path, csv_test_path, dest_root, mapping_out_path, metadata_map=None, imagename_to_camera_model=None, imagename_to_date=None):
    print(f"\n📦 Packaging Dataset : {dataset_name}")
    
    # A. Dossiers
    dir_train = os.path.join(dest_root, 'train')
    dir_test = os.path.join(dest_root, 'test')
    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    # B. Chargement CSV Source
    try:
        df_train_src = pd.read_csv(csv_train_path, sep=',', dtype=str)
        df_test_src = pd.read_csv(csv_test_path, sep=',', dtype=str)
    except Exception as e:
        print(f"❌ Erreur lecture CSV source : {e}")
        return

    # C. Harmonisation Colonnes
    col_maps = {
        'variety': 'class_name', 'taxon_name': 'class_name',
        'id': 'observation_id', 'sample_no': 'observation_id', 'tidy': 'observation_id',
        'file_name': 'image_name', 'local_path': 'image_name',
        'learn_tag': 'split',
        'date(dd-mm-yyyy)': 'date', 'date.dd.mm.yyyy.': 'date',
        'lat.wgs84.': 'latitude', 'lon.wgs84.': 'longitude',
        'gps_accuracy.m.': 'accuracy'
    }

    def safe_rename(df):
        for old, new in col_maps.items():
            if old in df.columns:
                if new in df.columns and old != new:
                    df.drop(columns=[old], inplace=True)
                else:
                    df.rename(columns={old: new}, inplace=True)
        return df.loc[:, ~df.columns.duplicated()]

    df_train_src = safe_rename(df_train_src)
    df_test_src = safe_rename(df_test_src)
    
    if 'split' not in df_train_src.columns: df_train_src['split'] = 'train'
    df_test_src['split'] = 'test'

    # D. Indexation Fichiers
    file_index = {}
    for r, d, f in os.walk(src_root):
        for filename in f:
            file_index[filename] = os.path.join(r, filename)

    global_untraced = []
    obs_registry = {}
    class_img_counter = {}

    # E. Fonction de Migration
    def migrate_subset(df, target_base_dir, is_flat_structure, subset_name):
        new_metadata = []
        missing_files = 0

        def get_scalar(row, col_name, default='unknown'):
            val = row.get(col_name, default)
            if isinstance(val, pd.Series):
                return val.iloc[0]
            return val
        
        for idx_row, row in tqdm(df.iterrows(), total=len(df), desc=f"Migration {subset_name}"):
            
            # 1. Infos Base
            old_fname_base = os.path.basename(str(row.get('image_name', '')).strip())
            cls_name = sanitize_filename(get_scalar(row, 'class_name'))
            
            # Gestion du compteur global
            if cls_name not in class_img_counter:
                class_img_counter[cls_name] = 0
            current_img_idx = class_img_counter[cls_name]
            class_img_counter[cls_name] += 1
            
            obs_id_raw = str(get_scalar(row, 'observation_id', 'unk')).replace('.0', '').strip()
            organ = sanitize_filename(str(row.get('organ', 'na')))
            
            # Assignation d'un ID court pour Market et Grape
            if dataset_name in ["market", "grape"]:
                if obs_id_raw not in obs_registry:
                    obs_registry[obs_id_raw] = str(len(obs_registry) + 1)
                obs_id_clean = obs_registry[obs_id_raw]
            else:
                obs_id_clean = sanitize_filename(obs_id_raw)
            
            # 2. Enrichissement Traçabilité
            author_val, license_val, url_val = "Unknown", "cc-by-sa", ""
            date_val = str(row.get('date', ''))
            
            # NOUVEAU : Nettoyage des NaN pour latitude / longitude
            lat_val = str(row.get('latitude', '')).strip()
            lon_val = str(row.get('longitude', '')).strip()
            if lat_val.lower() in ['nan', 'none', 'null', 'na']: lat_val = ""
            if lon_val.lower() in ['nan', 'none', 'null', 'na']: lon_val = ""

            if dataset_name == "banana":
                if metadata_map and obs_id_raw in metadata_map:
                    m = metadata_map[obs_id_raw]
                    author_val, license_val, url_val = m['author'], m['license'], m['source_url']
                    date_val, lat_val, lon_val = m['date'], m['latitude'], m['longitude']
                elif "Jenny" in str(row.get('user_id', '')): 
                    author_val = "Christophe Jenny"
                    url_val = "All Rights Reserved (Research Use Only)"
                else:
                    global_untraced.append({'split': os.path.basename(target_base_dir), 'id': obs_id_raw, 'img': old_fname_base})

            elif dataset_name == "market":
                model = imagename_to_camera_model.get(old_fname_base, "Unknown")
                author_val = MODEL_TO_AUTHOR.get(model, "Unknown")
                url_val = "https://identify.plantnet.org/en/varieties/species/tags/variety"
                license_val = "cc-by-sa"

            elif dataset_name == "grape":
                if subset_name == "train" and old_fname_base in metadata_map:
                    g = metadata_map[old_fname_base]
                    author_val, date_val, url_val = g['author'], g['date'], "https://identify.plantnet.org/en/varieties/species/tags/variety"
                    license_val = "cc-by-sa"
                else: 
                    author_val, url_val = "IFV / INRAE / L'Institut Agro (PlantGrape)", "https://www.plantgrape.fr/en"
                    if old_fname_base in imagename_to_date:
                        date_val = imagename_to_date[old_fname_base]
                    license_val = "All Rights Reserved (Research Use Only)"
            
            # 3. Nouveau Nom & Copie
            ext = os.path.splitext(old_fname_base)[1].lower()
            if ext == '.jpeg':
                ext = '.jpg'
            elif not ext:
                ext = '.jpg'
                
            new_fname = f"{dataset_name}_{cls_name}_{current_img_idx:05d}{ext}"

            src_file_path = file_index.get(old_fname_base)
            if src_file_path:
                final_dir = target_base_dir if is_flat_structure else os.path.join(target_base_dir, cls_name)
                os.makedirs(final_dir, exist_ok=True)
                shutil.copy2(src_file_path, os.path.join(final_dir, new_fname))

                date_val = format_date_iso(date_val)
                
                # 4. Métadonnées Finales
                meta_row = row.to_dict()
                meta_row.update({
                    'image_name': new_fname, 
                    'observation_id': obs_id_clean,        # Le nouvel ID (simplifié ou iNat)
                    'class_name': cls_name,
                    'author': author_val, 
                    'license': license_val, 
                    'source_url': url_val,
                    'date': date_val, 
                    'latitude': lat_val, 
                    'longitude': lon_val,
                    'original_filename': old_fname_base,   # L'ancien nom d'image
                    'original_observation_id': obs_id_raw  # L'ancien ID d'observation
                })
                new_metadata.append(meta_row)
            else:
                missing_files += 1
        
        return pd.DataFrame(new_metadata)

    # F. Exécution
    df_tv = migrate_subset(df_train_src, dir_train, False, "train")
    df_test = migrate_subset(df_test_src, dir_test, True, "test")

    # G. Sauvegarde CSV Standardisé
    def save_csv(df, path):
        if df.empty: return
        prio = ['image_name', 'observation_id', 'class_name', 'organ', 'split', 'author', 'license', 'source_url', 'date', 'latitude', 'longitude']
        final = [c for c in prio if c in df.columns]
        df[final].to_csv(path, index=False)

    save_csv(df_tv, os.path.join(dest_root, 'train_val.csv'))
    save_csv(df_test, os.path.join(dest_root, 'test.csv'))
    generate_class_mapping(df_tv, dest_root)
    
    if global_untraced:
        pd.DataFrame(global_untraced).to_csv(os.path.join(dest_root, 'untraced_observations.csv'), index=False)

    # H. Sauvegarde du Mapping de renommage HORS de release_candidate
    df_all = pd.concat([df_tv, df_test], ignore_index=True)
    if not df_all.empty and 'original_filename' in df_all.columns and 'image_name' in df_all.columns:
        # Ajout de original_observation_id dans l'export
        mapping_df = df_all[[
            'original_filename', 
            'image_name', 
            'original_observation_id', 
            'observation_id', 
            'split'
        ]]
        os.makedirs(os.path.dirname(mapping_out_path), exist_ok=True)
        mapping_df.to_csv(mapping_out_path, index=False)
        print(f"    📝 Dictionnaire de renommage sauvegardé : {mapping_out_path}")

# --- 3. LANCEMENT ---

if __name__ == "__main__":
    BASE_OUT = "/data/infraspecies/vegetables/zenodo/release_candidate/"
    MAPPINGS_DIR = "/data/infraspecies/vegetables/zenodo/filename_mappings/" # Dossier externe
    
    # 1. BANANA
    banana_meta_file = "/data/infraspecies/vegetables/zenodo/data_in/observations-680356.csv"
    banana_full_metadata = load_inaturalist_metadata(banana_meta_file)

    process_dataset_migration(
        dataset_name="banana", 
        src_root="/data/infraspecies/vegetables/zenodo/data_in/banana", 
        csv_train_path="/data/infraspecies/vegetables/zenodo/data_in/banana/train.csv",
        csv_test_path="/data/infraspecies/vegetables/zenodo/data_in/banana/test.csv", 
        dest_root=os.path.join(BASE_OUT, "banana"),
        mapping_out_path=os.path.join(MAPPINGS_DIR, "banana_mapping.csv"),
        metadata_map=banana_full_metadata,
        imagename_to_camera_model=None
    )
    
    # 2. MARKET
    imagename_to_camera_model_file = "/data/infraspecies/vegetables/zenodo/data_in/plant-market-garden/market_camera_models.txt"
    imagename_to_camera_model = load_camera_mapping(imagename_to_camera_model_file)
    
    process_dataset_migration(
        dataset_name="market", 
        src_root="/data/infraspecies/vegetables/zenodo/data_in/plant-market-garden",
        csv_train_path="/data/infraspecies/vegetables/zenodo/data_in/plant-market-garden/train.csv",
        csv_test_path="/data/infraspecies/vegetables/zenodo/data_in/plant-market-garden/test.csv", 
        dest_root=os.path.join(BASE_OUT, "plant-market-garden"),
        mapping_out_path=os.path.join(MAPPINGS_DIR, "market_mapping.csv"),
        metadata_map=None,
        imagename_to_camera_model=imagename_to_camera_model
    )
    
    # 3. GRAPE
    grape_meta_file = "/data/infraspecies/vignes/plantagroeco_summer_school_2024_vine_varieties_identification_kit/datasets/PlantNet2012IFV/plantnet_2012_ifv.csv"
    grape_extra_meta = load_grape_metadata(grape_meta_file)
    
    grape_test_date_exif = "/data/infraspecies/vegetables/zenodo/data_in/grape_variety/grape_test_dates.txt"
    imagename_to_date = load_grape_test_dates(grape_test_date_exif)
    
    process_dataset_migration(
        "grape", 
        src_root="/data/infraspecies/vegetables/zenodo/data_in/grape_variety",
        csv_train_path="/data/infraspecies/vegetables/zenodo/data_in/grape_variety/train.csv",
        csv_test_path="/data/infraspecies/vegetables/zenodo/data_in/grape_variety/test.csv",
        dest_root=os.path.join(BASE_OUT, "grape-variety"),
        mapping_out_path=os.path.join(MAPPINGS_DIR, "grape_mapping.csv"),
        metadata_map=grape_extra_meta,
        imagename_to_camera_model=None,
        imagename_to_date=imagename_to_date
    )

    