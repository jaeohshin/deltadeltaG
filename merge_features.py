import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import re

# Constants
R_CONSTANT = 1.987204e-3
STRUCT_COLS = ['atomic_fluctuation', 'deformation_energy', 'phi', 'psi', 'rsa', 'residue_depth']
DIRS = {
    'sig': 'data/features/signatures',
    'dyn': 'data/features/dynamics',
    'env': 'data/features/residue_env',
    'pharma': 'data/features/pharma_deltas',
    'evo': 'data/features/evolutionary'
}

def get_struct_values(df_dyn, df_env, mut_string):
    """
    Robust matching that handles string/int mismatches and chain whitespace.
    """
    vals = []
    # Standardize: RA83Q or R A 83 Q -> WT: R, Chain: A, Res: 83
    mut_list = str(mut_string).split(',')
    
    # Pre-clean columns to ensure fast, reliable matching
    if not df_env.empty:
        df_env['res_id'] = df_env['res_id'].astype(str).str.strip()
        df_env['chain'] = df_env['chain'].astype(str).str.strip()
    if not df_dyn.empty:
        df_dyn['residue_index'] = df_dyn['residue_index'].astype(str).str.strip()

    for m in mut_list:
        m = m.strip()
        # Use Regex to extract WT, Chain, and ResNum
        match = re.search(r'([A-Za-z])\s*([A-Za-z])\s*(\d+)', m)
        if not match: continue
            
        chain_id = match.group(2).strip().upper()
        res_num_str = match.group(3).strip()
        
        # 1. Environment Match (RSA/Depth)
        env_match = df_env[(df_env['res_id'] == res_num_str) & (df_env['chain'] == chain_id)]
        
        # 2. Dynamics Match (NMA)
        # Often sequential, so we try the ResNum directly first
        dyn_match = df_dyn[df_dyn['residue_index'] == res_num_str]
        
        res_data = {}
        if not env_match.empty:
            res_data.update(env_match.iloc[0].to_dict())
        if not dyn_match.empty:
            res_data.update(dyn_match.iloc[0].to_dict())
            
        if res_data:
            vals.append(res_data)
                
    if vals:
        # Convert to numeric for averaging, ignore non-numeric columns
        mean_vals = pd.DataFrame(vals).apply(pd.to_numeric, errors='coerce').mean().to_dict()
        return {c: round(mean_vals.get(c, 0.0), 5) for c in STRUCT_COLS}
    return {c: 0.0 for c in STRUCT_COLS}

# --- 1. Load Master SKEMPI ---
print("📖 Loading SKEMPI Master...")
skempi = pd.read_csv('data/raw/skempi_v2.csv', sep=';')
for col in ['Temperature', 'Affinity_mut_parsed', 'Affinity_wt_parsed']:
    skempi[col] = pd.to_numeric(skempi[col], errors='coerce')
skempi['ddG_target'] = R_CONSTANT * skempi['Temperature'] * np.log(skempi['Affinity_mut_parsed'] / skempi['Affinity_wt_parsed'])

# --- 2. Hybrid Merge Logic ---
sig_files = glob.glob(os.path.join(DIRS['sig'], "row_*_sig.csv"))
print(f"🔎 Found {len(sig_files)} samples. Merging...")

final_rows = []
found_struct_count = 0
diagnostic_done = False

for f_sig in tqdm(sig_files):
    fname = os.path.basename(f_sig)
    parts = fname.split('_')
    row_idx = int(parts[1])
    pdb_id = parts[2].upper()
    mut_prefix = "_".join(parts[:-1])

    if row_idx >= len(skempi): continue
    skempi_row = skempi.iloc[row_idx]
    if pd.isna(skempi_row['ddG_target']): continue
    
    mut_str = skempi_row['Mutation(s)_PDB']

    # Define paths
    paths = {
        'evo': os.path.join(DIRS['evo'], f"{mut_prefix}_evo.csv"),
        'delta': os.path.join(DIRS['pharma'], f"{mut_prefix}_delta.csv"),
        'dyn': os.path.join(DIRS['dyn'], f"{pdb_id}_dynamics.csv"),
        'env': os.path.join(DIRS['env'], f"{pdb_id}_env.csv")
    }

    if not all(os.path.exists(p) for p in paths.values()): continue
        
    try:
        struct_feats = get_struct_values(pd.read_csv(paths['dyn']), pd.read_csv(paths['env']), mut_str)
        
        # Diagnostic: Show us the first match to confirm it's working
        if not diagnostic_done and struct_feats['rsa'] > 0:
            print(f"\n✅ Diagnostic Match for {pdb_id} {mut_str}:")
            print(f"   RSA: {struct_feats['rsa']}, Fluctuation: {struct_feats['atomic_fluctuation']}")
            diagnostic_done = True

        if struct_feats['rsa'] > 0: found_struct_count += 1
        
        combined = {
            'pdb_id': pdb_id, 'mutation': mut_str, 'ddG_target': round(skempi_row['ddG_target'], 5),
            **pd.read_csv(f_sig).iloc[0].to_dict(),
            **struct_feats,
            **pd.read_csv(paths['delta']).iloc[0, 2:].to_dict(),
            **pd.read_csv(paths['evo']).iloc[0, 2:].to_dict()
        }
        final_rows.append(combined)
    except: continue

# --- 3. Final Save ---
if final_rows:
    master_df = pd.DataFrame(final_rows)
    os.makedirs('data/processed', exist_ok=True)
    master_df.to_csv('data/processed/mmcsm_features.csv', index=False)
    print(f"\n✅ Success! Dataset assembled: {len(master_df)} samples.")
    print(f"📊 Structural data successfully mapped for {found_struct_count} samples.")