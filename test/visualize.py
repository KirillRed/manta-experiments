import numpy as np
import argparse
import torch
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

parser = argparse.ArgumentParser()

parser.add_argument("--action", type=str, default="Forgot to specify the action")
parser.add_argument("--dataset", type=str, default="breakfast")
parser.add_argument("--name", type=str, default="unnamed")
# type 0 is qualitative, 1 is shuffling
parser.add_argument("--test_type", type=int, default=0)

args = parser.parse_args()

dataset = "50salads" if args.dataset.startswith("5") else "breakfast"
print(f"Visualizing using {dataset}")

def load_mappings(mapping_file_path):
    id_to_action = {}
    action_to_id = {}
    with open(mapping_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                class_id = int(parts[0])
                action_name = parts[1]
                
                id_to_action[class_id] = action_name
                action_to_id[action_name] = class_id
                
    return id_to_action, action_to_id

def load_ground_truth_text(gt_file_path, action_to_id):
    gt_ids = []
    with open(gt_file_path, 'r') as f:
        for line in f:
            action_name = line.strip()
            if action_name in action_to_id:
                gt_ids.append(action_to_id[action_name])
            else:
                print(f"Warning: Action '{action_name}' not found in mapping!")
                gt_ids.append(0) 
                
    return np.array(gt_ids)

def extract_samples(file_path, sample_indices=[0, 1, 2], window_size=51):
    try:
        preds = torch.load(file_path)
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
    except:
        preds = np.load(file_path, allow_pickle=True)

    print(f"Loaded {file_path} with shape: {preds.shape}")

    extracted_arrays = []
    for idx in sample_indices:
        raw_sample = preds[idx, :]
        smoothed_sample = signal.medfilt(raw_sample, kernel_size=window_size)
        extracted_arrays.append(smoothed_sample)
        
    return extracted_arrays

def visualize_action_ribbons(ribbons_dict, id_to_action, title, save_path=None):
    names = list(ribbons_dict.keys())
    arrays = list(ribbons_dict.values())
    
    all_unique_classes = np.unique(np.concatenate(arrays))
    num_unique = len(all_unique_classes)

    print(num_unique)
    
    cmap = plt.get_cmap('tab20', max(num_unique, 1))
    
    color_dict = {cls_id: cmap(i) for i, cls_id in enumerate(all_unique_classes)}

    fig, ax = plt.subplots(figsize=(15, 1.5 * len(names)))
    
    for i, (name, arr) in enumerate(zip(names, arrays)):
        colored_ribbon = np.array([color_dict[int(val)] for val in arr])
        
        y_pos = len(names) - 1 - i 
        ax.imshow([colored_ribbon], extent=[0, len(arr), y_pos, y_pos + 0.7], aspect='auto')
        ax.text(-0.02 * len(arr), y_pos + 0.35, name, va='center', ha='right', fontsize=12, fontweight='bold')


    max_len = max([len(arr) for arr in arrays])
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, len(names))
    ax.set_yticks([]) 
    ax.set_xlabel('Time (Frames)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    legend_patches = []
    for cls_id in all_unique_classes:
        action_name = id_to_action.get(cls_id, f"Unknown ({cls_id})")
        patch = mpatches.Patch(color=color_dict[cls_id], label=action_name)
        legend_patches.append(patch)
        
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
              ncol=4, fancybox=True, shadow=True, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    id_to_action, action_to_id = load_mappings(f"./datasets/breakfast/data/{dataset}/mapping.txt")

    file_path = "/data1/kredensk/anticipation/manta/test/test.bundle"
    with open(file_path, "r") as file:
        file_name = file.read().strip()
    
    gt_array = load_ground_truth_text(f"./datasets/breakfast/data/{dataset}/groundTruth/{file_name}", action_to_id)
    
    if args.test_type == 0:
        ctrl_samples = extract_samples("./test/full_sample.pt", sample_indices=[0, 1, 2])
        splice_samples = extract_samples("./test/sliced_sample.pt", sample_indices=[0, 1, 2])
        
        my_ribbons = {
            "Ground Truth": gt_array,
            "Full Sample 0": ctrl_samples[0],
            "Full Sample 1": ctrl_samples[1],
            "Full Sample 2": ctrl_samples[2],
            "Spliced Sample 0": splice_samples[0],
            "Spliced Sample 1": splice_samples[1],
            "Spliced Sample 2": splice_samples[2]
        }
        
        title = f'MANTA Action Prediction Variance (Full vs Spliced (after removing {args.action}))' 

        visualize_action_ribbons(my_ribbons, id_to_action, title, save_path=f"test/results/{args.name}")
    else:
        shuffled_gt = load_ground_truth_text("./test/shuffled_sample.txt", action_to_id)
        ctrl_samples = extract_samples("./test/full_sample.pt", sample_indices=[0, 1, 2])
        samples = extract_samples("./test/shuffled_sample.pt", sample_indices=[0, 1, 2])
        my_ribbons = {
            "Ground Truth": gt_array,
            "Original Sample 0": ctrl_samples[0],
            "Original Sample 1": ctrl_samples[1],
            "Original Sample 2": ctrl_samples[2],
            "Shuffled Ground Truth": shuffled_gt,
            "Shuffled Sample 0": samples[0],
            "Shuffled Sample 1": samples[1],
            "Shuffled Sample 2": samples[2]
        }
        
        title = f'MANTA Action Prediction Variance (Original vs Shuffled)' 

        visualize_action_ribbons(my_ribbons, id_to_action, title, save_path=f"test/results/{args.name}.png")