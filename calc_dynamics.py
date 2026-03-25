import os
import pandas as pd
from tqdm import tqdm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from Bio.PDB import PDBParser # Added for size checking

# 1. Initialize R-Python bridge
pandas2ri.activate()
bio3d = importr('bio3d')
parser = PDBParser(QUIET=True)

def get_nma_features(pdb_path):
    try:
        pdb = bio3d.read_pdb(pdb_path)
        modes = bio3d.nma(pdb)
        
        fluctuations = bio3d.fluct_nma(modes)
        deformation = bio3d.deformation_nma(modes)
        # Get the 'ei' list from R
        ei_list = deformation.rx2('ei')
        # Aggregate: Sum the energies across modes for each residue 
        # to get a single scalar per residue
        summed_deformation = [sum(list(res_energies)) for res_energies in ei_list]
        return list(fluctuations), summed_deformation
    
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return None, None

# 2. Process your structures
pdb_dir = 'data/pdb'
output_dir = 'data/features/dynamics'
os.makedirs(output_dir, exist_ok=True)

pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])

print(f"Starting dynamics calculation for {len(pdb_files)} structures...")

for pdb_file in pdb_files:
    pdb_id = pdb_file.split('.')[0]
    output_path = os.path.join(output_dir, f"{pdb_id}_dynamics.csv")
    
    if os.path.exists(output_path):
        continue
    
    # Quick check for protein size
    structure = parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_file))
    residue_count = len(list(structure.get_residues()))
    
    print(f"\n" + "="*40)
    print(f"PROCESSING: {pdb_id}")
    print(f"SIZE: {residue_count} residues")
    print("="*40)
    
    flucts, def_energies = get_nma_features(os.path.join(pdb_dir, pdb_file))
    
    if flucts and def_energies:
        res_df = pd.DataFrame({
            'residue_index': range(1, len(flucts) + 1),
            'atomic_fluctuation': flucts,
            'deformation_energy': def_energies
        })
        res_df.to_csv(output_path, index=False)