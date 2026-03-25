# ΔΔG Predictor: Structural Feature Engineering Pipeline

This project implements a high-dimensional feature engineering pipeline to predict the change in binding free energy ($\Delta\Delta G$) of protein-protein complexes upon mutation. It integrates physical dynamics, geometric orientation, and chemical environment into a unified machine learning matrix.

---

## The Four Pillars of Feature Engineering

The model represents each mutation site through **260 distinct features** categorized into four structural domains:

### 1. Physical Dynamics (Normal Mode Analysis)
We treat the protein as an **Elastic Network Model (ENM)** to calculate vibrational entropy and local stiffness.
* **Atomic Fluctuation:** Derived from the NMA Hessian matrix, representing the "flexibility" of the mutation site.
* **Deformation Energy:** Measures the local mechanical strain required to deform the structure at the mutation residue.

### 2. Geometric Environment
Describes the spatial orientation and solvent exposure of the residue.
* **RSA (Relative Solvent Accessibility):** Determines if a mutation is "Buried" in the core or "Exposed" on the surface.
* **Residue Depth:** The distance (in Å) from the residue to the nearest bulk water molecule.
* **Torsion Angles ($\phi, \psi$):** Local backbone geometry defining the secondary structure context.

### 3. Chemical Pharmacophore Signatures
A 288-dimensional spatial fingerprint of the mutation neighborhood (15Å radius).
* **Atom Typing:** Atoms are classified into 8 types: *Hydrophobic, Positive, Negative, Aromatic, Sulfur, HB-Donor, HB-Acceptor, and Neutral*.
* **Distance Binning:** Atom-pairs are counted in 2Å bins (0Å to 16Å) to capture specific interactions like salt bridges (bin 2-4Å) or long-range packing.

### 4. Thermodynamics (The Target)
Experimental binding affinities ($K_d$) from the **SKEMPI 2.0** dataset are converted into energy units using the formula:

$$\Delta\Delta G = R \cdot T \cdot \ln\left(\frac{K_{d, \text{mut}}}{K_{d, \text{wt}}}\right)$$

---

## Directory Structure

```plaintext
/data/work/deltadeltaG
├── data/
│   ├── pdb/                # Raw .pdb files from RCSB
│   ├── features/
│   │   ├── dynamics/       # NMA output CSVs
│   │   ├── residue_env/    # RSA and Torsion CSVs
│   │   └── signatures/     # 288-dim Pharmacophore CSVs
│   └── processed/          # Final training matrix (master_features.csv)
├── scripts/                # Python processing pipeline
└── model_performance.png   # Evaluation plots
