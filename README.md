# Mitochondrial Modulators

This repository contains the code and data accompanying the paper:

**"Discovery of Mitochondrial Modulators Using Target-Aggregated Machine Learning"**

Published in *Journal of Pharmaceutical Analysis*, 2026.
DOI: [10.1016/j.jpha.2026.101579](https://doi.org/10.1016/j.jpha.2026.101579)

---

## Quick Start: Use the Pre-trained Model

If you only need to use the pre-trained XGBoost model for prediction (without retraining), you can directly load `XGboost.pkl` and apply it to your own molecules:

```python
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the pre-trained model
model = joblib.load("XGboost.pkl")

# Generate Morgan fingerprints for your molecules (radius=2, 1024 bits)
smiles_list = ["CCO", "c1ccccc1"]  # Replace with your SMILES
mols = [Chem.MolFromSmiles(s) for s in smiles_list]
fingerprints = np.array([AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols])

# Predict activity probabilities
probabilities = model.predict_proba(fingerprints)[:, 1]
for smi, prob in zip(smiles_list, probabilities):
    print(f"{smi}: {prob:.4f}")
```

**Note:** The pre-trained model file (`XGboost.pkl`) is included in this repository. You can use it directly without running the training scripts. To retrain the model from scratch, run Steps 1–3 below.

---

## Overview

This project uses a target-aggregated machine learning approach to discover mitochondrial modulators. Bioactivity data for mitochondrial targets were collected from ChEMBL, preprocessed with Gaussian-based borderline sample removal, converted to Morgan molecular fingerprints, and used to train five classification models (Decision Tree, Neural Network, Random Forest, SVM, and XGBoost). The best-performing XGBoost model was then applied to a subset of the ZINC in-stock database to identify potential mitochondrial modulators.

## Repository Contents

### Data Files

| File | Description |
|------|-------------|
| `ChEMBLMolsRaw.csv` | Preprocessed ChEMBL bioactivity data with activity class labels (input to `a2`) |
| `MCEandLabMolsCanoSmiles.csv` | Canonical SMILES of active molecules from MCE and in-house lab (input to `a3`) |
| `ZincDatabase_in-vitro.csv` | ZINC in-stock molecules for virtual screening (input to `a5`) |
| `ZincDatabase_in-vitro_PredictedScores.xlsx` | ZINC molecules with predicted activity probabilities (output of `a5`) |
| `XGboost.pkl` | Pre-trained XGBoost model (output of `a4xgboost_hyerprot.py`, input to `a5`) |
| `TargetListInChEMBL.xlsx` | Summary table of mitochondrial targets found in ChEMBL (supplementary, Table 1 in the paper) |

### Scripts

Scripts are named with an `a2`–`a5` prefix indicating execution order. The `a1` script (ChEMBL data acquisition via API) is not included; users should start from `a2` using the provided `ChEMBLMolsRaw.csv`.

| Script | Description |
|--------|-------------|
| `a2gaussiandelete.py` | Gaussian-based deletion of borderline activity samples (IC50 500–2000 nM range) |
| `a3morgan.py` | Morgan fingerprint generation, similarity-based deduplication, and merging with external active molecules |
| `a4dt_hyerprot.py` | Decision Tree model with hyperparameter optimization (10-fold CV) |
| `a4nn_hyerprot.py` | Neural Network (MLP) model with hyperparameter optimization (10-fold CV) |
| `a4rf_hyerprot.py` | Random Forest model with hyperparameter optimization (10-fold CV) |
| `a4svm_hyerprot.py` | SVM model with hyperparameter optimization (10-fold CV) |
| `a4xgboost_hyerprot.py` | XGBoost model with hyperparameter optimization (10-fold CV), model training, saving, and full evaluation (ROC/PR curves, confusion matrix, metrics export) |
| `a5XGB_Prediction.py` | Virtual screening of ZINC database using the trained XGBoost model |

## Requirements

- Python 3.x
- numpy
- pandas
- rdkit
- scikit-learn
- xgboost (GPU support required for `a4xgboost_hyerprot.py`)
- hyperopt
- joblib
- matplotlib
- seaborn
- openpyxl (for Excel file read/write)

Install dependencies:

```bash
pip install numpy pandas rdkit scikit-learn xgboost hyperopt joblib matplotlib seaborn openpyxl
```

**Note:** The XGBoost script (`a4xgboost_hyerprot.py`) uses `tree_method='gpu_hist'` and requires an NVIDIA GPU. If no GPU is available, change `tree_method` to `'hist'` in the script (this may produce slightly different results).

## How to Reproduce

Run the scripts in order from the repository root directory:

### Step 1: Gaussian-based sample deletion (`a2`)

```bash
python a2gaussiandelete.py
```

- **Input:** `ChEMBLMolsRaw.csv`
- **Output:** `ChemblMolsGS.csv`
- Removes borderline samples (IC50 500–2000 nM) using a Gaussian probability function with a hard cutoff at 750–1500 nM. Random seed: 37.

### Step 2: Fingerprint generation and data merging (`a3`)

```bash
python a3morgan.py
```

- **Input:** `ChemblMolsGS.csv`, `MCEandLabMolsCanoSmiles.csv`
- **Output:** `traindata_gaussian.npy`
- Generates Morgan fingerprints (radius=2, 1024 bits), removes duplicates with Dice similarity > 0.95, and merges with external active molecules.

### Step 3: Model training (`a4`)

Train one or more models. Each script performs 10-fold cross-validation hyperparameter optimization using Hyperopt (TPE algorithm, 50 evaluations), then trains the final model on the full training set and saves it as a `.pkl` file.

```bash
python a4dt_hyerprot.py          # Output: dt.pkl
python a4nn_hyerprot.py          # Output: nn.pkl
python a4rf_hyerprot.py          # Output: rf.pkl
python a4svm_hyerprot.py         # Output: svm.pkl
python a4xgboost_hyerprot.py     # Output: XGboost.pkl (+ evaluation files)
```

- **Input:** `traindata_gaussian.npy`
- Data split: 90% train / 10% test (`random_state=0`)
- The XGBoost script additionally generates evaluation outputs (ROC/PR curves, confusion matrix, metrics spreadsheets).

### Step 4: Virtual screening (`a5`)

```bash
python a5XGB_Prediction.py
```

- **Input:** `XGboost.pkl`, `traindata_gaussian.npy`, `ZincDatabase_in-vitro.csv`
- **Output:** `ZincDatabase_in-vitro_PredictedScores.xlsx`, `dupdeleted_morgan_with_scores.npy`
- Removes molecules present in the training set, then predicts activity probabilities for the remaining ZINC molecules.

## Notes

- The `a1` script (ChEMBL API data acquisition and activity classification) is not included in this repository. The preprocessed data (`ChEMBLMolsRaw.csv`) is provided directly. The classification criteria are: IC50 ≤ 1000 nM = active (class 1), IC50 ≥ 50000 nM = inactive (class 0), intermediate values excluded. See the paper for details.
- Intermediate files (`ChemblMolsGS.csv`, `traindata_gaussian.npy`, `.pkl` model files) are generated by the scripts and not pre-uploaded. Run the scripts in order to generate them.
- `traindata_gaussian.npy` is ~105 MB and exceeds GitHub's file size limit. It is generated by `a3morgan.py`.
- All scripts use relative paths. Run them from the repository root directory.

## Citation

If you use this code or data, please cite:

```bibtex
@article{Wang2026Mitochondrial,
  title = {Discovery of Mitochondrial Modulators Using Target-Aggregated Machine Learning},
  author = {Wang, Lizhuo and Yang, Peng and others},
  journal = {Journal of Pharmaceutical Analysis},
  year = {2026},
  doi = {10.1016/j.jpha.2026.101579}
}
```

## License

This project is intended for academic research. Please refer to the journal publication for licensing terms.
