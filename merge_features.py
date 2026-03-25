import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# --- Constants ---
R_CONSTANT = 1.987204e-3 

def parse_mutations(mut_str):
    results = []
    parts = str(mut_str).split(',')
    for p in parts:
        match = re.search(r"([A-Za-z0-9])([A-Z])(\d+)([A-Z])", p.strip())
        if match:
            results.append((match.group(1), int(match.group(3))))
    return results

def find_file_case_insensitive(directory, filename_pattern):
    if not os.path.exists(directory): return None
    files = os.listdir(directory)
    pattern_lower = filename_pattern.lower()
    for f in files:
        if f.lower() == pattern_lower:
            return os.path.join(directory, f)
    return None

# --- Setup Paths ---
mut_path = 'data/processed/multiple_mutations_all.csv'
dyn_dir = 'data/features/dynamics'
env_dir = 'data/features/residue_env'
sig_dir = 'data/features/signatures'

df = pd.read_csv(mut_path)
df.columns = [c.strip() for c in df.columns]
final_data = []

# Define the full set of expected structural columns
STRUCTURAL_COLS = ['atomic_fluctuation', 'deformation_energy', 'phi', 'psi', 'rsa', 'residue_depth']

print(f"Merging features for {len(df)} SKEMPI entries...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pdb_id_full = str(row['#Pdb']).strip()
    pdb_code = pdb_id_full.split('_')[0]
    
    # 1. Signature Check
    sig_filename = f"row_{idx}_{pdb_id_full}_sig.csv"
    sig_path = find_file_case_insensitive(sig_dir, sig_filename)
    if not sig_path: continue

    # 2. Affinity Check
    try:
        kd_wt = pd.to_numeric(row['Affinity_wt_parsed'], errors='coerce')
        kd_mut = pd.to_numeric(row['Affinity_mut_parsed'], errors='coerce')
        temp = pd.to_numeric(row['Temperature'], errors='coerce')
        if np.isnan(temp): temp = 298.15
        if np.isnan(kd_wt) or np.isnan(kd_mut) or kd_wt <= 0 or kd_mut <= 0: continue
        ddG_target = R_CONSTANT * temp * np.log(kd_mut / kd_wt)
    except: continue

    # 3. Aggregation Block
    dyn_path = find_file_case_insensitive(dyn_dir, f"{pdb_code}_dynamics.csv")
    env_path = find_file_case_insensitive(env_dir, f"{pdb_code}_env.csv")
    
    res_features = {col: 0.0 for col in STRUCTURAL_COLS} # Initialize with zeros
    
    if dyn_path and env_path:
        d_df, e_df = pd.read_csv(dyn_path), pd.read_csv(env_path)
        mut_list = parse_mutations(row['Mutation(s)_PDB'])
        site_vals = []
        
        for ch, r_id in mut_list:
            d_match = d_df[d_df['residue_index'] == r_id]
            e_match = e_df[(e_df['res_id'] == r_id) & (e_df['chain'].str.upper() == ch.upper())]
            
            # Combine whatever data is available
            row_dict = {}
            if not d_match.empty:
                row_dict.update(d_match.iloc[0].to_dict())
            if not e_match.empty:
                row_dict.update(e_match.iloc[0].to_dict())
            
            if row_dict:
                site_vals.append(row_dict)
        
        if site_vals:
            agg_df = pd.DataFrame(site_vals)
            # Calculate mean only for columns that exist, others stay 0.0
            for col in STRUCTURAL_COLS:
                if col in agg_df.columns:
                    res_features[col] = agg_df[col].mean()

# 4. Final Success Merge (Including the human-readable mutation label)
    sig_features = pd.read_csv(sig_path).iloc[0].to_dict()
    final_data.append({
        'pdb_id': pdb_id_full, 
        'mutation': row['Mutation(s)_PDB'], # The "Label" you wanted
        'location': row['iMutation_Location(s)'], # Also very useful for debugging!
        'ddG_target': ddG_target, 
        **res_features, 
        **sig_features
    })

if final_data:
    master_df = pd.DataFrame(final_data)
    master_df.to_csv('data/processed/master_features.csv', index=False)
    print(f"\n[Success] Final dataset: {len(master_df)} samples, {master_df.shape[1]} columns.")
else:
    print("\n[Error] No rows were merged.")