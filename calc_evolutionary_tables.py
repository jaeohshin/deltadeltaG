import os
import pandas as pd
import numpy as np
from Bio.Align import substitution_matrices
from tqdm import tqdm

# --- 1. Load Standard Substitution Matrices ---
# These describe how likely nature is to swap one amino acid for another.
# High scores = common/tolerated; Low/Negative scores = rare/forbidden.
blosum62 = substitution_matrices.load("BLOSUM62")
pam250 = substitution_matrices.load("PAM250")

# --- 2. AAINDEX(3) Definitions ---
# [Hydrophobicity (Kyte-Doolittle), Molecular Weight (Da), pI (Isoelectric Point)]
# These represent the "Physical Cost" of the mutation.
AA_PROPS = {
    'A': [ 1.8,  89.1, 6.00], 'R': [-4.5, 174.2, 10.76], 'N': [-3.5, 132.1, 5.41],
    'D': [-3.5, 133.1, 2.77], 'C': [ 2.5, 121.2, 5.07], 'Q': [-3.5, 146.2, 5.65],
    'E': [-3.5, 147.1, 3.22], 'G': [-0.4,  75.1, 5.97], 'H': [-3.2, 155.2, 7.59],
    'I': [ 4.5, 131.2, 6.02], 'L': [ 3.8, 131.2, 5.98], 'K': [-3.9, 146.2, 9.74],
    'M': [ 1.9, 149.2, 5.74], 'F': [ 2.8, 165.2, 5.48], 'P': [-1.6, 115.1, 6.30],
    'S': [-0.8, 105.1, 5.68], 'T': [-0.7, 119.1, 5.60], 'W': [-0.9, 204.2, 5.89],
    'Y': [-1.3, 181.2, 5.66], 'V': [ 4.2, 117.1, 5.96]
}

def get_evo_scores(mut_string):
    """
    Calculates evolutionary deltas and substitution scores.
    For multiple mutations, it averages the scores across all sites.
    """
    # [d_hydro, d_mw, d_pi, blosum62, pam250]
    total_scores = np.zeros(5)
    
    try:
        muts = str(mut_string).split(',')
        count = 0
        for m in muts:
            m = m.strip()
            if len(m) < 3: continue
            
            wt_aa, mut_aa = m[0], m[-1]
            
            if wt_aa in AA_PROPS and mut_aa in AA_PROPS:
                # 1. AAINDEX Deltas (Mutant - WildType)
                delta_props = np.array(AA_PROPS[mut_aa]) - np.array(AA_PROPS[wt_aa])
                total_scores[0:3] += delta_props
                
                # 2. Evolutionary Matrix Scores
                # Note: Bio.Align matrices use (row, col) tuples
                try:
                    total_scores[3] += blosum62[(wt_aa, mut_aa)]
                    total_scores[4] += pam250[(wt_aa, mut_aa)]
                except KeyError:
                    # Fallback for rare cases or non-standard characters
                    pass
                
                count += 1
        
        # Average across multiple mutation sites
        return total_scores / count if count > 0 else total_scores
        
    except Exception:
        return total_scores

# --- Main Execution ---
input_csv = 'data/processed/multiple_mutations_all.csv'
out_dir = 'data/features/evolutionary'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(input_csv)
cols = ['d_aaindex_hydro', 'd_aaindex_mw', 'd_aaindex_pi', 'blosum62', 'pam250']

print(f"Generating evolutionary features for {len(df)} rows...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pdb_id = str(row['#Pdb']).strip()
    scores = get_evo_scores(row['Mutation(s)_PDB'])
    
    # Save as individual CSV file for modularity
    out_name = f"row_{idx}_{pdb_id}_evo.csv"
    pd.DataFrame([scores], columns=cols).to_csv(os.path.join(out_dir, out_name), index=False)

print(f"\n[Success] Evolutionary features saved in {out_dir}")