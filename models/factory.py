import torch
import timm
import os
import open_clip

def create_infraspecific_model(model_name, num_classes, img_size=None, frozen=False, checkpoint_path=''):
    model = None

    # --- CAS 1 : BioCLIP v2 (via open_clip) ---
    if "bioclip_v2" in model_name.lower():
        print(f"🚀 [FACTORY] Loading BioCLIP v2 Visual Encoder via open_clip...")
        
        # Le repo officiel imageomics/bioclip-2 utilise un ViT-L/14
        full_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        model = full_clip.visual 
        
        # Détection de la dimension de sortie (ViT-L/14 = 1024)
        # On utilise getattr avec 1024 comme fallback pour les architectures Large
        output_dim = getattr(model, 'output_dim', 1024)
        print(f"✅ Detected Feature Dimension: {output_dim}")
        
        model.classifier = torch.nn.Linear(output_dim, num_classes)

        # Injection de forward_features pour test.py et UMAP
        # Sur OpenCLIP, le forward du backbone renvoie les features globales
        model.forward_features = model.forward 

        # FIX FORWARD : On chaîne le backbone et le classifier pour train.py
        model.forward_backbone = model.forward 

        def hybrid_forward(x):
            features = model.forward_backbone(x)
            # Gestion du cas où CLIP renvoie un tuple (features, intermediate)
            if isinstance(features, tuple):
                features = features[0]
            return model.classifier(features)

        model.forward = hybrid_forward
        model.get_classifier = lambda: model.classifier

    # --- CAS 2 : TIMM (DINOv2, ConvNeXt, ResNet, etc.) ---
    else:
        print(f"🚀 [FACTORY] Creating model via TIMM: {model_name}")
        kwargs = {'pretrained': True, 'num_classes': num_classes}
        
        if img_size is not None:
            vit_keywords = ['vit', 'dinov2', 'swin', 'deit', 'eva', 'beit']
            if any(k in model_name.lower() for k in vit_keywords):
                kwargs['img_size'] = img_size
        
        model = timm.create_model(model_name, **kwargs)

    # --- ÉTAPE A : CHARGEMENT DU CHECKPOINT LOCAL ---
    # On charge avant le gel pour éviter tout conflit de requires_grad
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"📂 Loading Local Weights: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = state_dict.get('state_dict', state_dict.get('model', state_dict))
        
        # Filtrage de la tête pour éviter les crashs si num_classes diffère
        for key in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
            if key in state_dict and key in model.state_dict():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    print(f"⚠️ Removing {key} from checkpoint (shape mismatch)")
                    state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=False)

    # --- ÉTAPE B : STRATÉGIE DE GEL (FROZEN) ---
    if frozen:
        print(f"🔒 MODE FROZEN : Gel du backbone...")
        for param in model.parameters():
            param.requires_grad = False
            
        # Dégel de la tête uniquement
        target_head = None
        if hasattr(model, 'get_classifier'):
            target_head = model.get_classifier()
        
        if target_head:
            for param in target_head.parameters():
                param.requires_grad = True
            print("✅ Head identified and thawed.")
        else:
            # Fallback par nom pour les modèles timm standards
            for name, param in model.named_parameters():
                if any(k in name.lower() for k in ['head', 'fc', 'classifier']):
                    param.requires_grad = True

    # --- RÉSUMÉ FACTUEL DES PARAMÈTRES ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("-" * 30)
    print(f"📊 PARAMETER SUMMARY ({model_name})")
    print(f"    • Total       : {total_params:,}")
    print(f"    • Frozen      : {frozen_params:,}")
    print(f"    • Trainable   : {trainable_params:,}")
    print(f"    • Train ratio : {(trainable_params/total_params)*100:.4f}%")
    print("-" * 30)

    return model
    
    

def create_infraspecific_model_old(model_name, num_classes, img_size=None, frozen=False, checkpoint_path=''):
    """
    Creates a suitable timm model, handles backbone freezing, and loads local weights.
    """
    # 1. Basic argument configuration
    kwargs = {
        'pretrained': True, # <--- LE PIÈGE : Charge les poids génériques par défaut
        'num_classes': num_classes
    }

    # 2. Adaptive img_size management
    if img_size is not None:
        vit_models = ['vit', 'dinov2', 'swin', 'deit', 'eva', 'beit']
        if any(x in model_name.lower() for x in vit_models):
            kwargs['img_size'] = img_size
            print(f"🎯 Image size forced to {img_size} for Transformer architecture.")
        else:
            print(f"ℹ️ CNN model detected: img_size ignored at initialization.")

    # 3. Model creation via timm
    print(f"🚀 Creating model: {model_name}")
    model = timm.create_model(model_name, **kwargs)

    # 4. Local checkpoint loading with CRASH TEST
    if checkpoint_path:
        # --- SÉCURITÉ 1 : Le fichier existe-t-il vraiment ? ---
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"❌ CRITICAL ERROR: Checkpoint not found at: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        # weights_only=True est recommandé maintenant, mais False est ok pour la compatibilité
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Gestion des formats .tar / .pth
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict: # Parfois c'est 'model'
            state_dict = state_dict['model']

        # --- HEAD FILTERING ---
        # Nettoyage des têtes incompatibles (inchangé)
        for key in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']:
            if key in state_dict and key in model.state_dict():
                if state_dict[key].shape != model.state_dict()[key].shape:
                    print(f"⚠️ Removing {key} (shape mismatch)")
                    state_dict.pop(key)

        # --- SÉCURITÉ 2 : Vérification du chargement ---
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"ℹ️ Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
        # Si on a trop de missing keys, c'est suspect (sauf pour la tête)
        if len(msg.missing_keys) > 20:
            print(f"⚠️ WARNING: High number of missing keys! Check if the architecture matches the checkpoint.")

    elif frozen:
        # --- SÉCURITÉ 3 : Alerte Frozen sur Generic ---
        print("⚠️⚠️ ATTENTION : Mode FROZEN activé SANS checkpoint personnalisé !")
        print("⚠️ Le modèle utilisera les poids 'pretrained=True' (Generic/ImageNet/Timm).")
        print("⚠️ Si vous vouliez tester Pl@ntNet ou PC24, C'EST UNE ERREUR.")

    # 5. Robust freezing (Frozen) strategy
    if frozen:
        print(f"🔒 MODE FROZEN : Gel du backbone...")
        
        for param in model.parameters():
            param.requires_grad = False
            
        try:
            classifier = model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True
            print("✅ Tête identifiée et dégelée via get_classifier().")
        except:
            print("⚠️ Utilisation du filtrage par nom pour la tête...")
            for name, param in model.named_parameters():
                if any(k in name.lower() for k in ['head', 'fc', 'classifier']):
                    param.requires_grad = True
                    
    # 6. Factual parameter check (inchangé)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("-" * 30)
    print(f"📊 PARAMETER SUMMARY ({model_name})")
    print(f"   • Total       : {total_params:,}")
    print(f"   • Frozen      : {frozen_params:,}")
    print(f"   • Trainable   : {trainable_params:,}")
    print(f"   • Train ratio : {(trainable_params/total_params)*100:.4f}%")
    print("-" * 30)

    return model