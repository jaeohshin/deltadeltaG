import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# [Hydrophobic, Positive, Negative, Aromatic, Sulfur, HB-Donor, HB-Acceptor, Neutral]
AA_PHARMA_MAP = {
    'A': [1,0,0,0,0,0,0,1], 'C': [0,0,0,0,1,0,0,1], 'D': [0,0,1,0,0,0,2,0],
    'E': [0,0,1,0,0,0,2,0], 'F': [1,0,0,1,0,0,0,1], 'G': [0,0,0,0,0,0,0,1],
    'H': [0,1,0,1,0,1,1,0], 'I': [1,0,0,0,0,0,0,1], 'K': [0,1,0,0,0,2,0,0],
    'L': [1,0,0,0,0,0,0,1], 'M': [1,0,0,0,1,0,0,1], 'N': [0,0,0,0,0,1,2,1],
    'P': [0,0,0,0,0,0,0,1], 'Q': [0,0,0,0,0,1,2,1], 'R': [0,1,0,0,0,3,1,0],
    'S': [0,0,0,0,0,1,1,1], 'T': [0,0,0,0,0,1,1,1], 'V': [1,0,0,0,0,0,0,1],
    'W': [1,0,0,1,0,1,0,1], 'Y': [1,0,0,1,0,1,1,1]
}

def get_delta(mut_string):
    total_delta = np.zeros(8)
    try:
        for m in str(mut_string).split(','):
            m = m.strip()
            wt, mut = m[0], m[-1]
            if wt in AA_PHARMA_MAP and mut in AA_PHARMA_MAP:
                total_delta += (np.array(AA_PHARMA_MAP[mut]) - np.array(AA_PHARMA_MAP[wt]))
        return total_delta
    except: return total_delta

# Setup
out_dir = 'data/features/pharma_deltas'
os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv('data/processed/multiple_mutations_all.csv')

print("Generating Pharmacophore Delta files...")
cols = ['d_hyd', 'd_pos', 'd_neg', 'd_aro', 'd_sul', 'd_don', 'd_acc', 'd_neu']

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pdb_id = str(row['#Pdb']).strip()
    delta = get_delta(row['Mutation(s)_PDB'])
    
    # Save as individual file
    out_name = f"row_{idx}_{pdb_id}_delta.csv"
    pd.DataFrame([delta], columns=cols).to_csv(os.path.join(out_dir, out_name), index=False)