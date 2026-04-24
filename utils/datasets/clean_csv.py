import pandas as pd

def clean_metadata(file_path, output_path):
    df = pd.read_csv(file_path)
    
    # On supprime la colonne file_path si elle existe
    if 'file_path' in df.columns:
        df = df.drop(columns=['file_path'])
    if 'image_path' in df.columns:
        df = df.drop(columns=['image_path'])
    
    # Optionnel : On peut aussi supprimer d'autres colonnes très spécifiques au projet initial
    # pour ne garder que le nécessaire à la vision par ordinateur
    cols_to_keep = [
        'file_name', 'sample_no', 'organ', 'taxon_name', 
        'class_name', 'learn_tag', 'observation_tags'
    ]
    # Si tu veux garder tout le reste sauf file_path, ignore la ligne ci-dessus
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned {file_path} -> {output_path}")

# Application
clean_metadata('/data/infraspecies/vegetables/zenodo/plant-market-garden/train.csv', '/data/infraspecies/vegetables/zenodo/plant-market-garden/train_clean.csv')
clean_metadata('/data/infraspecies/vegetables/zenodo/plant-market-garden/test.csv', '/data/infraspecies/vegetables/zenodo/plant-market-garden/test_clean.csv')
clean_metadata('/data/infraspecies/vegetables/zenodo/banana/train.csv', '/data/infraspecies/vegetables/zenodo/banana/train_clean.csv')
clean_metadata('/data/infraspecies/vegetables/zenodo/banana/test.csv', '/data/infraspecies/vegetables/zenodo/banana/test_clean.csv')