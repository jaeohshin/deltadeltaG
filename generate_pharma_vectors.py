import pandas as pd
import numpy as np

# --- The Pharmacophore Table (The 8-D Chemical Identity) ---
# Format: [Hydrophobic, Positive, Negative, Aromatic, Sulfur, HB-Donor, HB-Acceptor, Neutral]
AA_PHARMA_MAP = {
    'A': [1, 0, 0, 0, 0, 0, 0, 1], 'C': [0, 0, 0, 0, 1, 0, 0, 1],
    'D': [0, 0, 1, 0, 0, 0, 2, 0], 'E': [0, 0, 1, 0, 0, 0, 2, 0],
    'F': [1, 0, 0, 1, 0, 0, 0, 1], 'G': [0, 0, 0, 0, 0, 0, 0, 1],
    'H': [0, 1, 0, 1, 0, 1, 1, 0], 'I': [1, 0, 0, 0, 0, 0, 0, 1],
    'K': [0, 1, 0, 0, 0, 2, 0, 0], 'L': [1, 0, 0, 0, 0, 0, 0, 1],
    'M': [1, 0, 0, 0, 1, 0, 0, 1], 'N': [0, 0, 0, 0, 0, 1, 2, 1],
    'P': [0, 0, 0, 0, 0, 0, 0, 1], 'Q': [0, 0, 0, 0, 0, 1, 2, 1],
    'R': [0, 1, 0, 0, 0, 3, 1, 0], 'S': [0, 0, 0, 0, 0, 1, 1, 1],
    'T': [0, 0, 0, 0, 0, 1, 1, 1], 'V': [1, 0, 0, 0, 0, 0, 0, 1],
    'W': [1, 0, 0, 1, 0, 1, 0, 1], 'Y': [1, 0, 0, 1, 0, 1, 1, 1]
}

def get_mutation_delta(mut_string):
    """
    Parses mutation strings and returns the summed chemical change.
    Example: 'RI67A,RI65A' -> calculates (A-R) + (A-R)
    """
    total_delta = np.zeros(8)
    try:
        # Handle multiple mutations separated by commas
        mutations = str(mut_string).split(',')
        for m in mutations:
            m = m.strip()
            if len(m) < 3: continue
            
            wt_aa = m[0]    # e.g., 'R'
            mut_aa = m[-1]  # e.g., 'A'
            
            if wt_aa in AA_PHARMA_MAP and mut_aa in AA_PHARMA_MAP:
                wt_vec = np.array(AA_PHARMA_MAP[wt_aa])
                mut_vec = np.array(AA_PHARMA_MAP[mut_aa])
                total_delta += (mut_vec - wt_vec)
        return total_delta
    except Exception as e:
        return total_delta

# --- Main Execution ---
input_file = 'data/processed/master_features.csv'
output_file = 'data/processed/mmcsm_features.csv'

print(f"Loading {input_file}...")
df = pd.read_csv(input_file)

if 'mutation' not in df.columns:
    print("[Error] No 'mutation' column found. Did you re-run the merger with labels?")
    exit()

print("Generating Pharmacophore Delta Vectors...")
# Calculate deltas for each row
deltas = df['mutation'].apply(get_mutation_delta).tolist()

# Convert list of arrays into a clean DataFrame
delta_cols = ['d_hyd', 'd_pos', 'd_neg', 'd_aro', 'd_sul', 'd_don', 'd_acc', 'd_neu']
delta_df = pd.DataFrame(deltas, columns=delta_cols)

# Concatenate with original features
final_df = pd.concat([df, delta_df], axis=1)

# Save the expanded dataset
final_df.to_csv(output_file, index=False)
print(f"\n[Success] Created {output_file}")
print(f"New Columns Added: {delta_cols}")
print(f"Total Features: {final_df.shape[1] - 3}") # -3 for pdb_id, mutation, ddG_target