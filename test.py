import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.data import create_transform
from sklearn.metrics import recall_score

# Custom modules
from data.dataset import InfraspecificPlantDataset
from models.factory import create_infraspecific_model
from utils.metrics import aggregate_observation_results, compute_final_metrics

def accuracy_topk(output, target, topk=(1, 5)):
    """
    Computes Image-level Top-1, Top-5, and Macro-Accuracy.
    Macro-Accuracy is the average of precision per variety.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Classic Top-K
    _, pred = output.topk(maxk, 1, True, True)
    pred_t = pred.t()
    correct = pred_t.eq(target.view(1, -1).expand_as(pred_t))

    res_metrics = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res_metrics.append(correct_k.mul_(100.0 / batch_size).item())
    
    # Image Macro Accuracy
    pred_top1 = pred[:, 0].cpu().numpy()
    target_np = target.cpu().numpy()
    macro_acc = recall_score(target_np, pred_top1, average='macro', zero_division=0) * 100
    res_metrics.append(macro_acc)
    
    return res_metrics # [Top-1, Top-5, Macro]


def knn_classifier(train_features, train_labels, test_features, k=20, T=0.07):
    """
    Probabilistic k-NN classification. 
    Converts distances into probabilities to allow observation-level voting.
    """
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    # Cosine similarity
    sim = torch.mm(test_features, train_features.t())
    topk_sim, topk_idx = sim.topk(k, dim=1)
    topk_labels = train_labels[topk_idx]
    
    # Temperature weighting
    weights = torch.exp(topk_sim / T)
    num_test = test_features.size(0)
    num_classes = train_labels.max() + 1
    
    one_hot = torch.zeros(num_test * k, num_classes).to(train_features.device).scatter_(1, topk_labels.view(-1, 1), 1.0)
    weighted_probs = (one_hot.view(num_test, k, num_classes) * weights.unsqueeze(-1)).sum(1)
    
    return F.normalize(weighted_probs, p=1, dim=1)


def extract_features(model, loader, device):
    """Extracts both logits and features simultaneously with error handling."""
    model.eval()
    all_logits, all_feats, all_labels = [], [], []
    all_obs_ids = []
    missing_count = 0
    
    print(f"🚀 Starting extraction on {len(loader.dataset)} images...")
    
    with torch.no_grad():
        iterator = iter(loader)
        while True:
            try:
                batch = next(iterator)
                images, labels, info = batch
                
                images = images.to(device)
                logits = model(images)
                feats = model.forward_features(images)
                
                if feats.ndim == 4:
                    feats = feats.mean(dim=(2, 3))
                elif feats.ndim == 3:
                    feats = feats.mean(dim=1)
                
                all_logits.append(logits.cpu())
                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())
                
                # Conversion explicite des tenseurs PyTorch en types Python natifs (int/float/str)
                obs_ids_list = [v.item() if torch.is_tensor(v) else v for v in info['observation_id']]
                all_obs_ids.extend(obs_ids_list)
                
            except StopIteration:
                break
            except FileNotFoundError as e:
                missing_count += 1
                if missing_count <= 5:
                    print(f"⚠️ Skipping missing image: {e}")
                continue
            except Exception as e:
                print(f"🔥 Unexpected error: {e}")
                continue
                
    if missing_count > 0:
        print(f"📢 Total missing images on disk: {missing_count}")
        
    if not all_logits:
        raise RuntimeError("❌ No data could be extracted. Check your paths.")
            
    return torch.cat(all_logits), torch.cat(all_feats), torch.cat(all_labels), all_obs_ids
    

def run_observation_metrics(logits_or_probs, obs_ids, labels, is_logits=True):
    """
    Handles aggregation by plant and computes observation-level metrics.
    Returns standard metrics AND the underlying dataframe for per-class analysis.
    """
    if is_logits:
        probs = torch.softmax(logits_or_probs, dim=1).numpy()
    else:
        probs = logits_or_probs.numpy() if torch.is_tensor(logits_or_probs) else logits_or_probs

    agg_dict = {}
    for i in range(len(labels)):
        oid = obs_ids[i]
        lbl = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
        if oid not in agg_dict:
            agg_dict[oid] = {'probs_list': [], 'label': lbl}
        agg_dict[oid]['probs_list'].append(probs[i])

    rows = []
    for oid, data in agg_dict.items():
        avg_probs = np.mean(data['probs_list'], axis=0)
        pred = np.argmax(avg_probs)
        
        top5_preds = np.argsort(avg_probs)[-5:]
        is_correct_t5 = data['label'] in top5_preds
        
        rows.append({
            'observation_id': oid,
            'probs': avg_probs,
            'label': data['label'],
            'pred': pred,
            'is_correct_top1': (pred == data['label']),
            'is_correct_top5': is_correct_t5            
        })
    
    df_obs = pd.DataFrame(rows).set_index('observation_id')
    
    t1, t5 = compute_final_metrics(df_obs)
    macro = recall_score(df_obs['label'], df_obs['pred'], average='macro', zero_division=0) * 100
    
    return t1, t5, macro, df_obs

def main():
    parser = argparse.ArgumentParser(description='Infraspecific Benchmarking - Exhaustive Test')
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--metadata-train', type=str, required=True)
    parser.add_argument('--metadata-test', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--class-mapping', type=str, default=None)
    parser.add_argument('--model', type=str, default='vit_base_patch14_dinov2.lvd142m')
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--results-dir', type=str, default='./results')
    parser.add_argument('--label-col', type=str, default='class_name')
    parser.add_argument('--train-split', type=str, default='train')
    parser.add_argument('--test-split', type=str, default='test')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true', default=False, help='Force recalculation even if metrics exist')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    
    # Early Exit logic
    csv_path = os.path.join(args.results_dir, 'per_class_metrics.csv')
    json_path = os.path.join(args.results_dir, 'test_metrics.json')
    if os.path.exists(csv_path) and os.path.exists(json_path) and not args.force:
        print(f"⏭️ Results already exist in {args.results_dir}. Skipping. Use --force to recalculate.")
        return

    device = torch.device(args.device)

    # --- 1. Dataset Initialization ---
    temp_set = InfraspecificPlantDataset(
        root_dir=args.root_dir, metadata=args.metadata_train, 
        class_mapping=args.class_mapping, label_col=args.label_col, split=None
    )
    classes = temp_set.classes
    num_classes = len(classes)

    # --- 2. Create and Load Model ---
    model = create_infraspecific_model(
        model_name=args.model, 
        num_classes=num_classes, 
        img_size=args.img_size,
        checkpoint_path=args.checkpoint 
    ).to(device)

    model.eval()

    # --- 3. Dataloaders ---
    config = getattr(model, 'default_cfg', {})
    transform = create_transform(input_size=args.img_size, is_training=False, interpolation=config.get('interpolation', 'bicubic'))
    
    train_loader = DataLoader(
        InfraspecificPlantDataset(root_dir=args.root_dir, metadata=args.metadata_train, transform=transform, class_mapping=args.class_mapping, split=args.train_split, label_col=args.label_col), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        InfraspecificPlantDataset(root_dir=args.root_dir, metadata=args.metadata_test, transform=transform, class_mapping=args.class_mapping, split=args.test_split, label_col=args.label_col), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # --- 4. Extraction and Inference ---
    train_feats_path = os.path.join(args.results_dir, 'train_feats.npy')
    train_labels_path = os.path.join(args.results_dir, 'train_labels.npy')
    
    test_feats_path = os.path.join(args.results_dir, 'test_feats.npy')
    test_labels_path = os.path.join(args.results_dir, 'test_labels.npy')
    test_logits_path = os.path.join(args.results_dir, 'test_logits.npy')
    test_obs_path = os.path.join(args.results_dir, 'test_obs_ids.json')

    # Chargement ou Extraction Train
    if os.path.exists(train_feats_path) and os.path.exists(train_labels_path) and not args.force:
        print("📥 Loading train features from disk...")
        train_feats = torch.from_numpy(np.load(train_feats_path))
        train_labels = torch.from_numpy(np.load(train_labels_path))
    else:
        print(f"🧪 Extracting train features on {device}...")
        _, train_feats, train_labels, _ = extract_features(model, train_loader, device)

    # Chargement ou Extraction Test
    if all(os.path.exists(p) for p in [test_feats_path, test_labels_path, test_logits_path, test_obs_path]) and not args.force:
        print("📥 Loading test features and logits from disk...")
        test_feats = torch.from_numpy(np.load(test_feats_path))
        test_labels = torch.from_numpy(np.load(test_labels_path))
        test_logits = torch.from_numpy(np.load(test_logits_path))
        with open(test_obs_path, 'r') as f:
            test_obs_ids = json.load(f)
    else:
        print(f"🧪 Extracting test features on {device}...")
        test_logits, test_feats, test_labels, test_obs_ids = extract_features(model, test_loader, device)

    # --- 5. Calculate Metrics ---
    print("📊 Computing scores (Linear Probing & k-NN)...")
    
    # A. Linear Probing (Head)
    img_h_t1, img_h_t5, img_h_macro = accuracy_topk(test_logits, test_labels)
    obs_h_t1, obs_h_t5, obs_h_macro, df_obs_h = run_observation_metrics(test_logits, test_obs_ids, test_labels, is_logits=True)

    # B. k-NN
    knn_probs_torch = knn_classifier(train_feats, train_labels, test_feats)
    img_k_t1, img_k_t5, img_k_macro = accuracy_topk(knn_probs_torch, test_labels)
    obs_k_t1, obs_k_t5, obs_k_macro, df_obs_k = run_observation_metrics(knn_probs_torch, test_obs_ids, test_labels, is_logits=False)

    # --- 6. Per-Class Detailed Metrics Calculation ---
    print("📈 Extracting per-class detailed metrics...")
    
    # Image predictions (Top-1 and Top-5)
    test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    
    # Sécurité au cas où un dataset aurait moins de 5 classes
    k_val = min(5, test_logits.size(1))
    _, test_top5_preds = test_logits.topk(k_val, dim=1)
    test_top5_preds = test_top5_preds.cpu().numpy()
    
    test_labels_np = test_labels.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    
    # Matrice booléenne : True si le vrai label est dans le Top-5 prédit
    is_correct_img_top5 = np.any(test_top5_preds == test_labels_np[:, None], axis=1)
    
    detailed_rows = []
    for c_idx, c_name in enumerate(classes):
        # Train Support
        train_supp = np.sum(train_labels_np == c_idx)
        
        # Image Level Data
        img_mask = (test_labels_np == c_idx)
        img_supp = np.sum(img_mask)
        img_acc = np.mean(test_preds[img_mask] == c_idx) * 100 if img_supp > 0 else np.nan
        img_top5_acc = np.mean(is_correct_img_top5[img_mask]) * 100 if img_supp > 0 else np.nan
        
        # Observation Level Data
        obs_mask = (df_obs_h['label'] == c_idx)
        obs_supp = np.sum(obs_mask)
        obs_acc = np.mean(df_obs_h[obs_mask]['pred'] == c_idx) * 100 if obs_supp > 0 else np.nan
        obs_top5_acc = np.mean(df_obs_h[obs_mask]['is_correct_top5']) * 100 if obs_supp > 0 else np.nan
        
        detailed_rows.append({
            'Class_ID': c_idx,
            'Class_Name': c_name,
            'Train_Images': train_supp,
            'Test_Images': img_supp,
            'Test_Observations': obs_supp,
            'Image_Top1_Acc': img_acc,
            'Image_Top5_Acc': img_top5_acc,
            'Observation_Top1_Acc': obs_acc,
            'Observation_Top5_Acc': obs_top5_acc
        })
        
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(csv_path, index=False)

    # --- 7. Final Display ---
    print(f"\n✅ Benchmark completed: {args.model}")
    print(f"{'Method':<15} | {'Img T1':<7} | {'Img Mac':<7} | {'Obs T1':<7} | {'Obs Mac':<7}")
    print("-" * 65)
    print(f"{'Linear/Full':<15} | {img_h_t1:>6.1f}% | {img_h_macro:>6.1f}% | {obs_h_t1:>6.1f}% | {obs_h_macro:>6.1f}%")
    print(f"{'k-NN':<15} | {img_k_t1:>6.1f}% | {img_k_macro:>6.1f}% | {obs_k_t1:>6.1f}% | {obs_k_macro:>6.1f}%")

    # --- 8. Save JSON ---
    if not os.path.exists(json_path) or args.force:
        final_metrics = {
            "linear_probing": {
                "image": {"top1": img_h_t1, "top5": img_h_t5, "macro": img_h_macro},
                "observation": {"top1": obs_h_t1, "top5": obs_h_t5, "macro": obs_h_macro}
            },
            "knn": {
                "image": {"top1": img_k_t1, "top5": img_k_t5, "macro": img_k_macro},
                "observation": {"top1": obs_k_t1, "top5": obs_k_t5, "macro": obs_k_macro}
            }
        }
        with open(json_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)

    # --- 9. Export UMAP Data ---
    data_to_export = [
        ('train_feats.npy', train_feats),
        ('train_labels.npy', train_labels),
        ('test_feats.npy', test_feats),
        ('test_labels.npy', test_labels),
        ('test_logits.npy', test_logits) # Ajout de la sauvegarde des logits
    ]
    
    for fname, tensor_data in data_to_export:
        fpath = os.path.join(args.results_dir, fname)
        if not os.path.exists(fpath) or args.force:
            np.save(fpath, tensor_data.cpu().numpy())

    # Ajout de la sauvegarde des observation_ids
    if not os.path.exists(test_obs_path) or args.force:
        with open(test_obs_path, 'w') as f:
            json.dump(test_obs_ids, f)

    classes_path = os.path.join(args.results_dir, 'classes.txt')
    if not os.path.exists(classes_path) or args.force:
        with open(classes_path, 'w') as f:
            f.write('\n'.join(classes))

    print(f"✅ Process completed. Results located in: {args.results_dir}")

if __name__ == '__main__':
    main()