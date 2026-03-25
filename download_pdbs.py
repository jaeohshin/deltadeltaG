import os
import pandas as pd
import requests
from tqdm import tqdm

def download_pdb(pdb_id, output_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_path):
        return False # Already exists
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True
        else:
            print(f"Failed to download {pdb_id}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        return False

# Load the filtered SKEMPI data
csv_path = 'data/processed/multiple_mutations_all.csv'
if not os.path.exists(csv_path):
    print("Error: Run init_data.py first to generate the CSV.")
    exit()

df = pd.read_csv(csv_path)

# SKEMPI 2.0 typically uses '#Pdb' as the header for the first column
pdb_column = '#Pdb' if '#Pdb' in df.columns else 'Pdb'

if pdb_column not in df.columns:
    print(f"Error: Could not find PDB column. Available columns: {df.columns.tolist()}")
    exit()

# Extract unique PDB IDs (first 4 characters)
pdb_ids = df[pdb_column].str[:4].unique()
pdb_dir = 'data/pdb'
os.makedirs(pdb_dir, exist_ok=True)

print(f"Found {len(pdb_ids)} unique PDB structures to download.")

downloaded = 0
for pid in tqdm(pdb_ids, desc="Downloading PDBs"):
    if download_pdb(pid, pdb_dir):
        downloaded += 1

print(f"Finished. Total new PDBs downloaded: {downloaded}")
