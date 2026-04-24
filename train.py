import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
import time
import random
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

# Timm tools
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, AverageMeter

# Custom modules
from data.dataset import InfraspecificPlantDataset
from models.factory import create_infraspecific_model
from utils.metrics import aggregate_observation_results, compute_final_metrics

import warnings

# Filtre les avertissements de type UserWarning provenant de sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Ou de manière encore plus ciblée pour le message sur la régression :
warnings.filterwarnings("ignore", message=".*represent a regression problem.*")


def set_seed(seed=42):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU
    
    # Guarantee (as much as possible) determinism for convolution/cuDNN algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

    
def main():
    parser = argparse.ArgumentParser(description='Infraspecific Plant Vision Benchmarking')

    parser.add_argument('--seed', type=int, default=42)

    # --- Dataset Parameters ---
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--class-mapping', type=str, default=None)
    parser.add_argument('--train-split', type=str, default='train')
    parser.add_argument('--val-split', type=str, default='val')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--label-col', type=str, default='class_name')
    
    # --- Model Parameters ---
    parser.add_argument('--model', type=str, default='vit_base_patch14_dinov2.lvd142m')
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--model-ema', action='store_true', help='Enable Model EMA')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998, help='Decay rate for model EMA')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('-j', '--workers', type=int, default=8)
    
    # --- Augmentation & Regularization ---
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1',  
                        help='Auto-augmentation policy (ex: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--re-prob', type=float, default=0.25, 
                        help='Random erasing probability (default: 0.25)')
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision')
    parser.add_argument('--output-dir', type=str, default='./output')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # --- STEP 1: Metadata & Class Count ---
    temp_set = InfraspecificPlantDataset(
        root_dir=args.root_dir,
        metadata=args.metadata,
        class_mapping=args.class_mapping,
        label_col=args.label_col,
        split=args.train_split
    )
    num_classes = len(temp_set.classes)
    
    # --- STEP 2: Model & EMA ---
    model = create_infraspecific_model(
        model_name=args.model, 
        num_classes=num_classes,
        frozen=args.frozen, 
        checkpoint_path=args.checkpoint, 
        img_size=args.img_size
    ).to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)

        # --- FIX BIOCLIP EMA ---
        # Si c'est BioCLIP, on doit s'assurer que le module EMA a les mêmes méthodes greffées
        if "bioclip" in args.model.lower():
            print("🔧 Patching ModelEMA for BioCLIP compatibility...")
            # On ré-injecte les méthodes sur le module interne de l'EMA
            model_ema.module.classifier = model.classifier
            model_ema.module.forward_backbone = model.forward_backbone
            model_ema.module.forward = model.forward 
            model_ema.module.forward_features = model.forward_features
            
        print(f"📈 Model EMA enabled with decay: {args.model_ema_decay}")

    # --- STEP 3: Transforms & Configuration ---
    config = getattr(model, 'default_cfg', {})
    input_size = config.get('input_size', (3, 224, 224))[-1]
    interpolation = config.get('interpolation', 'bicubic')

    train_transform = create_transform(
        input_size=args.img_size,
        is_training=True,
        use_prefetcher=False,      
        no_aug=False,
        scale=(0.08, 1.0),         
        ratio=(3./4., 4./3.),
        hflip=0.5,                 
        vflip=0.5,                 
        color_jitter=0.4,
        auto_augment=args.aa,      
        interpolation='bicubic',
        re_prob=args.re_prob,      
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    val_transform = create_transform(
        input_size=args.img_size,
        is_training=False,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # --- STEP 4: DataLoaders ---
    train_set = InfraspecificPlantDataset(
        root_dir=args.root_dir, 
        metadata=args.metadata, 
        transform=train_transform,
        class_mapping=args.class_mapping, 
        split=args.train_split, 
        label_col=args.label_col
    )
    val_set = InfraspecificPlantDataset(
        root_dir=args.root_dir, 
        metadata=args.metadata, 
        transform=val_transform,
        class_mapping=args.class_mapping, 
        split=args.val_split, 
        label_col=args.label_col
    )
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    # --- STEP 5: Optimizer (Differential LR Logic) ---
    if args.frozen:
        optimizer = create_optimizer_v2(
            model, 
            opt=args.opt, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        optimizer = create_optimizer_v2(
            model, 
            opt=args.opt, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        

    scheduler, _ = create_scheduler_v2(
        optimizer, 
        sched='cosine', 
        num_epochs=args.epochs, 
        min_lr=1e-6
    )
    
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            label_smoothing=args.smoothing, 
            num_classes=num_classes
        )

    criterion = SoftTargetCrossEntropy() if mixup_fn else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    # --- STEP 6: Training Loop ---
    history = []
    history_path = os.path.join(args.output_dir, 'history.csv')
    
    # Initialisation des records pour la logique imbriquée
    best_macro = -1.0
    best_obs_acc = -1.0
    best_img_acc = -1.0
    epochs_no_improve = 0
    
    print(f"🚀 Training started: {args.model} | Mode: {'Frozen' if args.frozen else 'Full'}")
    
    start_time = time.time() # Start global timer

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        losses_m = AverageMeter()
        batch_time_m = AverageMeter()
        end = time.time()

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=args.amp):
                if mixup_fn: 
                    images, labels = mixup_fn(images, labels)
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if model_ema: 
                model_ema.update(model)
        
            losses_m.update(loss.item(), images.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logique d'affichage corrigée (Intervalle ou Dernier batch)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                throughput = (args.batch_size * 60) / batch_time_m.avg
                print(f'Train: Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {losses_m.avg:.4f} | Speed: {throughput:.0f} img/min | Batch: {batch_time_m.avg:.3f}s')

        scheduler.step(epoch + 1)
        
        # --- Validation ---
        metrics_std = validate(model, val_loader, device)
        metrics_ema = validate(model_ema.module, val_loader, device) if model_ema else None
        
        # --- LOGIQUE DE SÉLECTION IMBRIQUÉE SÉCURISÉE (M > o > i) ---
        ref = metrics_ema if metrics_ema else metrics_std
        curr_m, curr_o, curr_i = ref['obs_macro'], ref['obs_acc'], ref['image_acc']
        
        eps = 1e-8 
        is_best = False

        # 1. Priorité 1 : Macro Accuracy
        if curr_m > best_macro + eps:
            is_best = True
        
        # 2. Priorité 2 : Observation Accuracy (si Macro égale)
        elif abs(curr_m - best_macro) < eps and curr_m > eps:
            if curr_o > best_obs_acc + eps:
                is_best = True
            
            # 3. Priorité 3 : Image Accuracy (si Macro et Obs égales)
            elif abs(curr_o - best_obs_acc) < eps:
                if curr_i > best_img_acc + eps:
                    is_best = True

        # --- EXÉCUTION DE LA SAUVEGARDE ---
        if is_best or epoch == 0:
            if is_best:
                # Mise à jour synchronisée de tous les records
                best_macro = curr_m
                best_obs_acc = curr_o
                best_img_acc = curr_i
                epochs_no_improve = 0  # Reset de la patience
                status_str = f"⭐ BEST: M={best_macro:.2f}% | o={best_obs_acc:.2f}%"
            else:
                # Cas particulier de l'époque 0
                status_str = "📦 Initial Save"
            
            # Sauvegarde physique des fichiers
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_best.pth'))
            if model_ema:
                torch.save(model_ema.module.state_dict(), os.path.join(args.output_dir, 'model_best_ema.pth'))
        else:
            # Pas de progression détectée sur la hiérarchie
            epochs_no_improve += 1
            status_str = f"⌛ {epochs_no_improve}/{args.patience}"

        # --- Time and Estimations ---
        epoch_duration = time.time() - epoch_start_time
        elapsed_total = time.time() - start_time
        avg_epoch_time = elapsed_total / (epoch + 1)
        eta_seconds = (args.epochs - (epoch + 1)) * avg_epoch_time
        finish_date = datetime.now() + timedelta(seconds=eta_seconds)

        # --- COMPACT LOG ---
        std_scores = f"{metrics_std['image_acc']:.1f}/{metrics_std['obs_acc']:.1f}/{metrics_std['obs_macro']:.1f}%"
        ema_info = ""
        if metrics_ema:
            ema_scores = f"{metrics_ema['image_acc']:.1f}/{metrics_ema['obs_acc']:.1f}/{metrics_ema['obs_macro']:.1f}%"
            ema_info = f" | EMA (i/o/M): {ema_scores}"
        
        curr_lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch {epoch:03d} | {format_time(epoch_duration)} | LR: {curr_lr:.6f} | Std (i/o/M): {std_scores}{ema_info} | {status_str}")
        print(f"📊 Progress: {format_time(elapsed_total)} elapsed | ETA: {format_time(eta_seconds)} (End: {finish_date.strftime('%H:%M')})")
        
        # --- STEP 7: History Logging ---
        history_entry = {
            'epoch': epoch, 'train_loss': losses_m.avg, 'lr': curr_lr,
            'std_img_acc': metrics_std['image_acc'], 'std_obs_acc': metrics_std['obs_acc'], 'std_obs_macro': metrics_std['obs_macro']
        }
        if metrics_ema:
            history_entry.update({
                'ema_img_acc': metrics_ema['image_acc'], 'ema_obs_acc': metrics_ema['obs_acc'], 'ema_obs_macro': metrics_ema['obs_macro']
            })
        
        history.append(history_entry)
        pd.DataFrame(history).to_csv(history_path, index=False)
        
        if epochs_no_improve >= args.patience:
            print(f"🛑 Early stopping at {epoch}. Total: {format_time(time.time() - start_time)}. Best Macro: {best_macro:.2f}%")
            break


def validate(model, loader, device):
    model.eval()
    image_results = []
    correct_images, total_images = 0, 0
    
    with torch.no_grad():
        for images, labels, info in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            correct_images += (preds == labels).sum().item()
            total_images += labels.size(0)
            
            for i in range(len(labels)):
                image_results.append({
                    'observation_id': str(info['observation_id'][i]),
                    'probs': probs[i].cpu().numpy(),
                    'label': labels[i].item()
                })

    # --- MANUAL AGGREGATION ---
    obs_data = {}
    for res in image_results:
        oid = res['observation_id']
        if oid not in obs_data:
            obs_data[oid] = {'probs_list': [], 'label': res['label']}
        obs_data[oid]['probs_list'].append(res['probs'])

    final_obs_ids, final_obs_avg_probs, final_obs_labels, final_obs_preds = [], [], [], []
    is_correct_top1_list = []

    for oid, data in obs_data.items():
        avg_p = np.mean(data['probs_list'], axis=0)
        pred = np.argmax(avg_p)
        label = data['label']
        
        final_obs_ids.append(oid)
        final_obs_avg_probs.append(avg_p)
        final_obs_labels.append(label)
        final_obs_preds.append(pred)
        
        # Prepare columns for compute_final_metrics
        is_correct_top1_list.append(pred == label)

    # Create "full" DataFrame to satisfy utils/metrics.py
    df_obs = pd.DataFrame({
        'observation_id': final_obs_ids,
        'probs': final_obs_avg_probs,
        'label': final_obs_labels,
        'prediction': final_obs_preds,
        'is_correct_top1': is_correct_top1_list # <--- FIX HERE
    })

    # Calculate Top-1 / Top-5 metrics via tools (won't crash anymore)
    try:
        obs_top1, _ = compute_final_metrics(df_obs)
    except Exception as e:
        # Fallback if compute_final_metrics requires more columns
        obs_top1 = np.mean(is_correct_top1_list) * 100.0
    
    # Calculate Macro Accuracy (Mean Recall)
    obs_macro = recall_score(final_obs_labels, final_obs_preds, average='macro', zero_division=0) * 100
    
    return {
        'image_acc': 100.0 * correct_images / total_images, 
        'obs_acc': obs_top1,
        'obs_macro': obs_macro
    }

if __name__ == '__main__':
    main()