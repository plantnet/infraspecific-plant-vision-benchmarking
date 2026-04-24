import torch
import numpy as np
import pandas as pd
from collections import defaultdict

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """
    Computes accuracy@k for the specified values of k.
    
    Args:
        output (torch.Tensor): Logits or probabilities from the model.
        target (torch.Tensor): Ground truth labels.
        topk (tuple): The 'k' values to calculate (e.g., 1 and 5).
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def aggregate_observation_results(image_results_list):
    """
    Aggregates image-level probabilities into observation-level predictions.
    Based on the strategy of averaging softmax probabilities per individual.
    
    Args:
        image_results_list (list): List of dicts containing 'observation_id', 
                                   'probs' (numpy array), and 'label'.
    """
    obs_data = defaultdict(lambda: {'probs': [], 'label': None})
    
    # Group probabilities by observation ID
    for item in image_results_list:
        obs_id = item['observation_id']
        obs_data[obs_id]['probs'].append(item['probs'])
        obs_data[obs_id]['label'] = item['label'] # Assuming all images in an obs have the same label

    obs_final_results = []
    
    for obs_id, data in obs_data.items():
        # Average the probabilities across all images in the observation
        avg_probs = np.mean(data['probs'], axis=0)
        avg_probs_torch = torch.from_numpy(avg_probs).unsqueeze(0)
        
        # Get Top-1 and Top-5 indices
        top1_pred = np.argmax(avg_probs)
        top5_preds = np.argsort(avg_probs)[-5:][::-1]
        
        obs_final_results.append({
            'observation_id': obs_id,
            'ground_truth': data['label'],
            'top1_prediction': int(top1_pred),
            'top5_predictions': top5_preds.tolist(),
            'is_correct_top1': int(top1_pred == data['label']),
            'is_correct_top5': int(data['label'] in top5_preds)
        })
        
    return pd.DataFrame(obs_final_results)

def compute_final_metrics(df_obs):
    """
    Calculates final percentages from the aggregated observation DataFrame.
    """
    top1_acc = df_obs['is_correct_top1'].mean() * 100.0
    top5_acc = df_obs['is_correct_top5'].mean() * 100.0
    return top1_acc, top5_acc

