import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class InfraspecificPlantDataset(Dataset):
    def __init__(self, root_dir, metadata, transform=None, exclude_organ=None, 
                 label_col='class_name', split=None, class_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col
        self.split = split

        # 1. Loading
        df = pd.read_csv(metadata) if isinstance(metadata, str) else metadata.copy()
        
        # 2. Mapping
        if class_mapping and os.path.exists(class_mapping):
            with open(class_mapping, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = sorted(df[self.label_col].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 3. Filtering by Split
        if split is not None and 'learn_tag' in df.columns:
            df = df[df['learn_tag'] == split].reset_index(drop=True)

        # --- 4. AUTO-CLEANING (Pre-flight Check) ---
        print(f"🔍 Checking integrity for split '{split}'...")
        valid_indices = []
        for i in range(len(df)):
            row = df.iloc[i]
            file_col = 'image_name' if 'image_name' in row else 'file_name'
            if self._find_image(row[self.label_col], row[file_col]):
                valid_indices.append(i)
        
        diff = len(df) - len(valid_indices)
        if diff > 0:
            print(f"⚠️ {diff} missing images on disk were removed from the dataset.")
        
        self.metadata = df.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ Dataset ready: {len(self.metadata)} valid images.")

    def __len__(self):
        return len(self.metadata)

    def _find_image(self, class_name, file_name):
        """Robust search including handling of extension case sensitivity."""
        c_raw = str(class_name).strip()
        variants = [c_raw, c_raw.replace(" ", "_"), c_raw.replace("_", " ")]
        
        # Handle finicky extensions (.jpg vs .JPG)
        name_root, _ = os.path.splitext(file_name)
        file_variants = [file_name, f"{name_root}.JPG", f"{name_root}.jpg", f"{name_root}.jpeg", f"{name_root}.JPEG"]

        bases = [self.root_dir]
        if not self.root_dir.endswith('images'):
            bases.append(os.path.join(self.root_dir, "images"))
        
        for base in bases:
            for v in variants:
                for f_v in file_variants:
                    path = os.path.join(base, self.split if self.split else "", v, f_v)
                    if os.path.exists(path):
                        return path
        return None

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_name = str(row.get('file_name', row.get('image_name', ''))).strip()
        class_name = str(row[self.label_col]).strip()
        
        img_path = self._find_image(class_name, file_name)
        # img_path should never be None here thanks to the constructor cleaning
        image = Image.open(img_path).convert('RGB')
        
        info = {
            'file_name': file_name,
            'observation_id': row.get('sample_no', row.get('id', 'unknown')),
            'label_name': class_name
        }

        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[class_name], info