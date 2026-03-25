import os
import pandas as pd
import shutil
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder, ResidueDepth, SASA
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to process a single PDB (moved out for parallelization)
def process_single_pdb(args):
    pdb_file, pdb_dir, output_dir = args
    pdb_id = pdb_file.split('.')[0]
    out_path = os.path.join(output_dir, f"{pdb_id}_env.csv")
    
    if os.path.exists(out_path):
        return f"Skipped {pdb_id}"

    pdb_path = os.path.join(pdb_dir, pdb_file)
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        model = structure[0]
        
        # 1. Torsion Angles
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
        
        # 2. Internal SASA
        sasa_calc = SASA.ShrakeRupley()
        sasa_calc.compute(model, level="R")
        
        # 3. Residue Depth
        msms_bin = shutil.which("msms")
        depth_data = []
        rd = None
        if msms_bin:
            try:
                rd = ResidueDepth(model, msms_exec=msms_bin)
            except:
                pass

        for res in model.get_residues():
            res_id = res.get_id()[1]
            chain = res.get_parent().get_id()
            sasa_val = getattr(res, "sasa", 0.0)
            
            r_depth, ca_depth = 0.0, 0.0
            if rd:
                try:
                    d = rd[chain, res_id]
                    r_depth, ca_depth = d[0], d[1]
                except:
                    pass

            depth_data.append({
                'res_id': res_id, 'chain': chain,
                'rsa': sasa_val, 'residue_depth': r_depth, 'ca_depth': ca_depth
            })

        # Merge and Save
        df_phi = pd.DataFrame(phi_psi_data)
        df_depth = pd.DataFrame(depth_data)
        
        if df_phi.empty:
            final_df = df_depth
        else:
            final_df = pd.merge(df_phi, df_depth, on=['res_id', 'chain'], how='outer').fillna(0.0)
            
        final_df.to_csv(out_path, index=False)
        return f"Completed {pdb_id}"
        
    except Exception as e:
        return f"Error {pdb_id}: {str(e)}"

# --- Main Execution ---
if __name__ == "__main__":
    pdb_dir = 'data/pdb'
    output_dir = 'data/features/residue_env'
    os.makedirs(output_dir, exist_ok=True)

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    # Prepare arguments for the pool
    tasks = [(f, pdb_dir, output_dir) for f in pdb_files]

    # Use max available cores minus 2 (to keep system responsive)
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_pdb, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Total Progress"):
            pass # The tqdm update happens as tasks complete

    print(f"All features saved to {output_dir}")