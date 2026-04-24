import os
import torch
import argparse
from collections import OrderedDict

def reconstruct_backbone_heads(state_dict):
    """
    Extrait les poids du backbone et des têtes à partir du dictionnaire spécifique Pl@ntNet.
    """
    backbone_state_dict = OrderedDict()
    heads = OrderedDict()

    for k, x in state_dict.items():
        if k.startswith("backbone_."):
            # On retire le préfixe 'backbone_.' (10 caractères)
            backbone_state_dict[k[10:]] = x
        elif k.startswith("heads_."):
            if k.endswith("._extra_state"):
                name = x["name"]
                if name == "species":
                    heads = x
                    key = k[:-12]
                    this_feature_state_dict = OrderedDict()
                    for kk, xx in state_dict.items():
                        if kk != k and kk.startswith(key):
                            this_feature_state_dict[kk[len(key) :]] = xx
                    heads["state_dict"] = this_feature_state_dict
                    heads["state_dict"] = {
                        k.replace("extra_fc.", ""): v 
                        for k, v in heads["state_dict"].items() 
                        if k.startswith("extra_fc.")
                    }
        else:
            print(f"⚠️ Warning: Unknown key <{k}> while loading PN state")

    return backbone_state_dict, heads

def convert_and_save_pn_checkpoint(input_path, output_path):
    """
    Charge le modèle Pl@ntNet, extrait le backbone, et le sauvegarde
    dans un format standard compatible avec timm.
    """
    print(f"📥 Loading Pl@ntNet checkpoint from: {input_path}")
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"❌ Error: File not found at {input_path}")
        
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    if "state_dict" not in checkpoint:
        raise KeyError("❌ Error: The checkpoint does not contain a 'state_dict' key.")

    state_dict = checkpoint["state_dict"]
    
    # Extraction du backbone propre
    print("⚙️ Extracting backbone weights...")
    backbone_state_dict, _ = reconstruct_backbone_heads(state_dict)
    
    if not backbone_state_dict:
        raise ValueError("❌ Error: Failed to extract backbone weights. Check the checkpoint structure.")

    # Création du dictionnaire final (on ne garde QUE le state_dict pour timm)
    final_checkpoint = {
        'state_dict': backbone_state_dict
    }

    print(f"💾 Saving clean backbone to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_checkpoint, output_path)
    
    print("✅ Conversion successful! The new checkpoint is ready for timm.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pl@ntNet checkpoint to timm compatible format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original pn_model.pth.tar")
    parser.add_argument("--output", type=str, required=True, help="Path to save the clean timm-compatible .pth")
    
    args = parser.parse_args()
    
    convert_and_save_pn_checkpoint(args.input, args.output)