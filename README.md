# mmCSM-PPI Reimplementation: ΔΔG Prediction Pipeline

This project implements a high-performance feature engineering pipeline to predict the change in binding free energy ($\Delta\Delta G$) of protein-protein complexes upon mutation. By integrating structural dynamics, geometric orientation, chemical identity, and evolutionary history, it creates a unified machine learning matrix for supervised regression.

---

## 🧬 The Five Pillars of Feature Engineering

The model represents each mutation through **270+ distinct features** categorized into five specialized domains:

### 1. Physical Dynamics (Normal Mode Analysis)
Treats the protein as an **Elastic Network Model (ENM)** to calculate vibrational entropy and local stiffness.
* **Atomic Fluctuation:** Derived from the NMA Hessian matrix, representing the "flexibility" of the mutation site.
* **Deformation Energy:** Measures the local mechanical strain required to deform the structure at the mutation residue.

### 2. Geometric Environment
Describes the spatial orientation and solvent exposure of the residue at the interface.
* **RSA (Relative Solvent Accessibility):** Determines if a mutation is "Buried" in the core or "Exposed" on the surface.
* **Residue Depth:** The distance (in Å) from the residue to the nearest bulk water molecule.
* **Torsion Angles ($\phi, \psi$):** Local backbone geometry defining the secondary structure context.

### 3. Chemical Pharmacophore Signatures (CSM)
A 288-dimensional spatial fingerprint of the mutation neighborhood (15Å radius).
* **Atom Typing:** Atoms are classified into 8 types (Hydrophobic, Positive, Negative, Aromatic, Sulfur, HB-Donor, HB-Acceptor, and Neutral).
* **Distance Binning:** Atom-pairs are counted in 2Å bins (0.2Å to 16Å) to capture specific interactions like salt bridges or long-range packing.

### 4. Pharmacophore Deltas (Chemical Identity)
Captures the net change in chemical properties specifically at the mutation site.
* **Property Vector:** An 8-dimensional vector representing the gain or loss of specific pharmacophore counts (e.g., $d\_pos$, $d\_hyd$) calculated as $\text{Mutant} - \text{WildType}$.

### 5. Evolutionary & Contact Potential
Leverages historical conservation and substitution tolerance data.
* **Substitution Matrices:** Scores from **BLOSUM62** and **PAM250** indicating how frequently a mutation is tolerated by nature.
* **AAINDEX(3):** Physical deltas in Hydrophobicity, Molecular Weight, and Isoelectric Point (pI) to capture the energetic cost of the swap.

---

## ⚙️ Methodology

### The Training Target
Experimental binding affinities ($K_d$) from the **SKEMPI 2.0** dataset are converted into energy units using the thermodynamic relationship:

$$\Delta\Delta G = R \cdot T \cdot \ln\left(\frac{K_{d, \text{mut}}}{K_{d, \text{wt}}}\right)$$

* $R = 1.987 \times 10^{-3}$ kcal/(mol·K)
* $T = 298.15$ K (unless otherwise specified in metadata)

### Multiple Mutations (mmCSM)
For entries containing multiple mutations (e.g., `RI67A,RI65A`), the pipeline applies **Feature Averaging** for structural pillars (Dynamics/Env) and **Feature Summation** for chemical deltas, ensuring the model captures the collective impact of all changes.

### Symmetry Augmentation
To enforce thermodynamic consistency and double the training data, the model utilizes **Reverse Mutation Augmentation**:
$$\Delta\Delta G_{WT \to Mut} = -\Delta\Delta G_{Mut \to WT}$$

---

## 📂 Directory Structure

```plaintext
deltadeltaG/
├── data/
│   ├── pdb/                 # Raw .pdb files
│   ├── processed/           # Master feature CSVs and SKEMPI labels
│   └── features/            # Modular feature directories
│       ├── dynamics/        # NMA-based physical features
│       ├── residue_env/     # RSA, Depth, Torsion
│       ├── signatures/      # CSM pharmacophore bins
│       ├── pharma_deltas/   # Chemical identity deltas
│       └── evolutionary/    # BLOSUM, PAM, and AAINDEX scores
├── merge_features.py        # Assembly line for feature integration
├── calc_pharma_deltas.py    # Chemical feature generator
├── calc_evolutionary.py     # Evolutionary feature generator
└── train_mmcsm.py           # Model training with Symmetry Augmentation