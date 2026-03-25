import os
import pandas as pd
import requests

# 1. Setup Directory Structure
dirs = ['data/raw', 'data/processed', 'data/pdb', 'src']
for d in dirs:
    os.makedirs(d, exist_ok=True)

# 2. Download SKEMPI 2.0 (Experimental Thermodynamics)
skempi_url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
raw_path = 'data/raw/skempi_v2.csv'

if not os.path.exists(raw_path):
    print(f"Downloading SKEMPI 2.0 from {skempi_url}...")
    response = requests.get(skempi_url)
    with open(raw_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

# 3. Filter for Multiple Mutations (mmCSM-PPI criteria)
# mmCSM-PPI focuses on 2-27 point mutations.
df = pd.read_csv(raw_path, sep=';')

def count_mutations(mut_string):
    # Format is usually 'Chain WT Position Mutant, ...'
    return len(mut_string.split(','))

df['mut_count'] = df['Mutation(s)_PDB'].apply(count_mutations)

# Filter for multiple mutations (2 to 27) 
multi_df = df[(df['mut_count'] >= 2) & (df['mut_count'] <= 27)].copy()

# Note: The paper used double/triple mutants (1126 entries) for training [cite: 51]
# and 4-27 mutants (595 entries) as a blind test[cite: 52].
training_set = multi_df[multi_df['mut_count'] <= 3]
blind_test = multi_df[multi_df['mut_count'] > 3]

print(f"Total multiple mutation entries: {len(multi_df)}")
print(f"Training set candidates (2-3 mutations): {len(training_set)}")
print(f"Blind test candidates (4-27 mutations): {len(blind_test)}")

# Save processed lists
multi_df.to_csv('data/processed/multiple_mutations_all.csv', index=False)
print("Saved filtered lists to data/processed/")
