import os
import json
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate summary table of experiments')
    parser.add_argument('--results-dir', type=str, required=True, help='Parent directory containing all experiments')
    parser.add_argument('--output-file', type=str, default='summary_results.csv')
    args = parser.parse_args()

    summary_data = []

    # 1. Recursive walk to find all test_metrics.json files
    for root, dirs, files in os.walk(args.results_dir):
        if 'test_metrics.json' in files:
            exp_path = os.path.join(root, 'test_metrics.json')
            folder_name = os.path.basename(root)
            
            # Attempt to infer dataset name from the path
            dataset_name = os.path.basename(os.path.dirname(root))
            
            try:
                with open(exp_path, 'r') as f:
                    data = json.load(f)
                
                # Data extraction
                row = {
                    'Dataset': dataset_name,
                    'Experience': folder_name,
                    # Head - Observation (Highest priority)
                    'Head_Obs_T1': data['linear_probing']['observation']['top1'],
                    'Head_Obs_Mac': data['linear_probing']['observation']['macro'],
                    'Head_Obs_T5': data['linear_probing']['observation']['top5'],
                    # k-NN - Observation
                    'kNN_Obs_T1': data['knn']['observation']['top1'],
                    'kNN_Obs_Mac': data['knn']['observation']['macro'],
                    # Head - Image
                    'Head_Img_T1': data['linear_probing']['image']['top1'],
                    'Head_Img_Mac': data['linear_probing']['image']['macro']
                }
                summary_data.append(row)
            except Exception as e:
                print(f"⚠️ Error reading {exp_path}: {e}")

    if not summary_data:
        print("❌ No data found. Check the --results-dir path.")
        return

    # 2. DataFrame creation and sorting
    df = pd.DataFrame(summary_data)
    df = df.sort_values(by=['Dataset', 'Head_Obs_T1'], ascending=[True, False])

    # 3. Saving
    output_path = os.path.join(args.results_dir, args.output_file)
    df.to_csv(output_path, index=False)

    # 4. Formatted console output
    print("\n📊 --- EXPERIMENT RESULTS SUMMARY TABLE ---")
    if len(df) > 0:
        # to_string() does not require 'tabulate'
        print(df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.2f}%"))
    else:
        print("No data to display.")
    
    print(f"\n💾 Full table saved to: {output_path}")
    
if __name__ == '__main__':
    main()