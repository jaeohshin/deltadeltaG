import os
import pandas as pd
import glob

DIRS = {
    'sig': 'data/features/signatures',
    'dyn': 'data/features/dynamics',
    'env': 'data/features/residue_env',
    'pharma': 'data/features/pharma_deltas',
    'evo': 'data/features/evolutionary'
}

# 1. Grab just the first signature file
sig_files = sorted(glob.glob(os.path.join(DIRS['sig'], "row_*_sig.csv")))
if not sig_files:
    print("❌ No signature files found.")
    exit()

f_sig = sig_files[0]
fname = os.path.basename(f_sig)
parts = fname.split('_')
# Extract what we *think* are the IDs
row_idx = parts[1]
pdb_id = parts[2].upper()
mut_prefix = "_".join(parts[:-1])

print(f"🕵️  Auditing file: {fname}")
print(f"   Inferred Row Index: {row_idx}")
print(f"   Inferred PDB ID:    {pdb_id}")
print(f"   Inferred Prefix:    {mut_prefix}")
print("-" * 30)

# 2. Check Paths
paths = {
    'evo': os.path.join(DIRS['evo'], f"{mut_prefix}_evo.csv"),
    'delta': os.path.join(DIRS['pharma'], f"{mut_prefix}_delta.csv"),
    'dyn': os.path.join(DIRS['dyn'], f"{pdb_id}_dynamics.csv"),
    'env': os.path.join(DIRS['env'], f"{pdb_id}_env.csv")
}

for key, path in paths.items():
    status = "✅ EXISTS" if os.path.exists(path) else "❌ MISSING"
    print(f"{status:10} | {key:6} | {path}")

# 3. Check Internal Columns
if os.path.exists(f_sig):
    df = pd.read_csv(f_sig)
    print("-" * 30)
    print(f"📄 Columns in {fname}:")
    print(df.columns.tolist()[:10]) # Show first 10 columns