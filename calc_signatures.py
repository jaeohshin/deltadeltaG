import os
import re
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from tqdm import tqdm
from itertools import combinations_with_replacement
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
RADIUS = 15.0 
BINS = np.arange(0, 16, 2) 
# CRITICAL FIX: Sort types alphabetically so combinations match sorted pair lookups
TYPES = sorted(['Hydrophobic', 'Positive', 'Negative', 'Aromatic', 'Sulfur', 'HB-Donor', 'HB-Acceptor', 'Neutral'])
TYPE_PAIRS = list(combinations_with_replacement(TYPES, 2))

def get_pharmacophore_type(atom):
    res_name = atom.get_parent().get_resname()
    atom_name = atom.get_name()
    element = atom.element
    if element == 'S': return 'Sulfur'
    if res_name in ['ARG', 'LYS', 'HIS'] and element == 'N':
        if res_name == 'ARG' and atom_name in ['NH1', 'NH2', 'NE']: return 'Positive'
        if res_name == 'LYS' and atom_name == 'NZ': return 'Positive'
        if res_name == 'HIS' and atom_name in ['ND1', 'NE2']: return 'Positive'
    if res_name in ['ASP', 'GLU'] and element == 'O': return 'Negative'
    if res_name in ['PHE', 'TYR', 'TRP', 'HIS'] and element == 'C':
        if atom_name in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2']: return 'Aromatic'
    if element == 'C' and res_name in ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO']: return 'Hydrophobic'
    if element == 'N': return 'HB-Donor'
    if element == 'O': return 'HB-Acceptor'
    return 'Neutral'

def parse_mutation_string(mut_str):
    results = []
    # Pattern for SKEMPI: Chain, WT, ResID, Mut (e.g., DA42G)
    parts = str(mut_str).split(',')
    for p in parts:
        match = re.search(r"([A-Za-z0-9])([A-Z])(\d+)([A-Z])", p.strip())
        if match:
            results.append((match.group(1), int(match.group(3))))
    return results

def process_row(args):
    idx, pdb_id_full, mut_str, pdb_dir, output_dir = args
    pdb_code = pdb_id_full.split('_')[0].upper()
    out_path = os.path.join(output_dir, f"row_{idx}_{pdb_id_full}_sig.csv")
    
    if os.path.exists(out_path): return "SUCCESS"

    pdb_path = os.path.join(pdb_dir, f"{pdb_code}.pdb")
    if not os.path.exists(pdb_path):
        return f"MISSING: {pdb_path}"

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        
        mut_list = parse_mutation_string(mut_str)
        centers = []
        for chain_id, res_id in mut_list:
            # TRY FUZZY CHAIN LOOKUP (Case insensitive)
            found_res = None
            for c in [chain_id.upper(), chain_id.lower()]:
                if c in model:
                    if res_id in model[c]:
                        found_res = model[c][res_id]
                        break
            
            if found_res and 'CA' in found_res:
                centers.append(found_res['CA'])
        
        if not centers:
            return f"SKIP: No residues found for {mut_str} in {pdb_id_full}"

        ns = NeighborSearch(list(model.get_atoms()))
        nearby_atoms = []
        seen_ids = set()
        for center in centers:
            for atom in ns.search(center.coord, RADIUS):
                if atom.get_full_id() not in seen_ids:
                    nearby_atoms.append((atom, get_pharmacophore_type(atom)))
                    seen_ids.add(atom.get_full_id())

        # Initialize signature with correct alphabetical sorting
        signature_vector = {f"{p1}_{p2}_bin{b}": 0 for p1, p2 in TYPE_PAIRS for b in BINS[:-1]}
        
        for (a1, t1), (a2, t2) in combinations_with_replacement(nearby_atoms, 2):
            dist = np.linalg.norm(a1.coord - a2.coord)
            if dist > RADIUS or dist == 0: continue
            
            # SORT THE PAIR to match the alphabetical TYPE_PAIRS keys
            pair = tuple(sorted([t1, t2]))
            bin_idx = int(dist // 2) * 2
            
            key = f"{pair[0]}_{pair[1]}_bin{bin_idx}"
            if key in signature_vector:
                signature_vector[key] += 1
        
        pd.DataFrame([signature_vector]).to_csv(out_path, index=False)
        return "SUCCESS"
    except Exception as e:
        return f"ERROR {pdb_id_full}: {str(e)}"

if __name__ == "__main__":
    mutations_df = pd.read_csv('data/processed/multiple_mutations_all.csv')
    pdb_dir = 'data/pdb'
    output_dir = 'data/features/signatures'
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for idx, row in mutations_df.iterrows():
        tasks.append((idx, str(row['#Pdb']).strip(), row['Mutation(s)_PDB'], pdb_dir, output_dir))

    print(f"Starting parallel signatures with 14 workers...")
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(process_row, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calculating Signatures"):
            res = future.result()
            # if res != "SUCCESS": print(f"\n[Warning] {res}") # Uncomment to see skips

    print(f"\nProcessing complete. Files in {output_dir}: {len(os.listdir(output_dir))}")