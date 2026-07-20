import numpy as np
import os
import re
import json
from tqdm import tqdm

def get_idx(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else 0

root_dir = 'results'
ams_dir_path = '../'
output_dir = 'aggregated_results'
os.makedirs(output_dir, exist_ok=True)

ams_folders = sorted([d for d in os.listdir(root_dir) if d.startswith('ams_')], key=get_idx)
n_ams = len(ams_folders)

if n_ams == 0:
    raise FileNotFoundError("Aucun dossier ams_ trouvé.")

first_folder = os.path.join(root_dir, ams_folders[0])
rep_files_init = sorted([f for f in os.listdir(first_folder) if f.endswith('_score.npy')], key=get_idx)
n_rep = len(rep_files_init)

sample_data = np.load(os.path.join(first_folder, rep_files_init[0]))
p = sample_data.shape[1]

scores_tensor = np.empty((n_ams, n_rep, p))
probs_list = []

with tqdm(total=n_ams, desc="Traitement des dossiers ams") as pbar:
    for i, ams_folder in enumerate(ams_folders):
        # 1. Extraction des scores (dans root_dir)
        current_score_path = os.path.join(root_dir, ams_folder)
        rep_files = sorted([f for f in os.listdir(current_score_path) if f.endswith('_score.npy')], key=get_idx)
        
        for j, rep_file in enumerate(rep_files):
            file_path = os.path.join(current_score_path, rep_file)
            score = np.load(file_path)
            scores_tensor[i, j, :] = score[:]

        # 2. Extract probabilities (inside ams_dir_path)
        checkpoint_path = os.path.join(ams_dir_path, ams_folder, "ams_checkpoint.txt")
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    if "current_p" in data:
                        probs_list.append(float(data["current_p"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        pbar.update(1)

np.save(os.path.join(output_dir, 'final_scores.npy'), scores_tensor)

if probs_list:
    probs_array = np.array(probs_list)
    np.save(os.path.join(output_dir, 'final_probs.npy'), probs_array)
    
    mean_val = np.mean(probs_array)
    std_err = np.std(probs_array) / np.sqrt(len(probs_array))
    
    with open(os.path.join(output_dir, "results_prob.txt"), "w") as out:
        out.write(f"Mean: {mean_val}, SEM: {std_err}\n")
