import os
import pandas as pd
import shutil
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder, ResidueDepth, SASA

def get_residue_features(pdb_id, pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
    except Exception as e:
        print(f"\n[Error] Could not parse {pdb_id}: {e}")
        return None

    # 1. Torsion Angles (Phi/Psi)
    phi_psi_data = []
    ppb = PPBuilder()
    for pp in ppb.build_peptides(model):
        phi_psi = pp.get_phi_psi_list()
        for i, residue in enumerate(pp):
            phi_psi_data.append({
                'res_id': residue.get_id()[1],
                'chain': residue.get_parent().get_id(),
                'phi': phi_psi[i][0] if phi_psi[i][0] is not None else 0.0,
                'psi': phi_psi[i][1] if phi_psi[i][1] is not None else 0.0
            })
    
    # 2. INTERNAL SASA (No DSSP binary needed!)
    # This calculates the Solvent Accessible Surface Area using Biopython's Shrake-Rupley algorithm
    sasa_calc = SASA.ShrakeRupley()
    sasa_calc.compute(model, level="R") # R for Residue level
    
    # 3. Residue Depth (Optional fallback)
    msms_bin = shutil.which("msms")
    depth_data = []
    
    # We try ResidueDepth, but if it fails, we still collect SASA
    rd = None
    if msms_bin:
        try:
            rd = ResidueDepth(model, msms_exec=msms_bin)
        except:
            pass

    for res in model.get_residues():
        res_id = res.get_id()[1]
        chain = res.get_parent().get_id()
        
        # Get SASA (Accessibility)
        sasa_val = getattr(res, "sasa", 0.0)
        
        # Get Depth if available
        r_depth = 0.0
        ca_depth = 0.0
        if rd:
            try:
                d = rd[chain, res_id]
                r_depth, ca_depth = d[0], d[1]
            except:
                pass

        depth_data.append({
            'res_id': res_id,
            'chain': chain,
            'rsa': sasa_val, # Using SASA as our accessibility feature
            'residue_depth': r_depth,
            'ca_depth': ca_depth
        })

    # Merge and Cleanup
    df_phi = pd.DataFrame(phi_psi_data)
    df_depth = pd.DataFrame(depth_data)
    
    if df_phi.empty: return df_depth
    return pd.merge(df_phi, df_depth, on=['res_id', 'chain'], how='outer').fillna(0.0)

# --- Main Execution ---
pdb_dir = 'data/pdb'
output_dir = 'data/features/residue_env'
os.makedirs(output_dir, exist_ok=True)

pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])

print(f"Processing {len(pdb_files)} structures with internal Biopython tools...")
for pdb_file in tqdm(pdb_files):
    pdb_id = pdb_file.split('.')[0]
    out_path = os.path.join(output_dir, f"{pdb_id}_env.csv")
    if os.path.exists(out_path): continue
        
    res_df = get_residue_features(pdb_id, os.path.join(pdb_dir, pdb_file))
    if res_df is not None:
        res_df.to_csv(out_path, index=False)