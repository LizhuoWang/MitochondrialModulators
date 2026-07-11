# -*- coding: utf-8 -*-
"""

"""
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import joblib



xgb = joblib.load("XGboost.pkl")

c = np.load("traindata_gaussian.npy")
c = np.delete(c, 1024, axis=1)

df = pd.read_csv("ZincDatabase_in-vitro.csv")

df = df.rename(columns={'0':'canonical_smiles'})

mols = [Chem.MolFromSmiles(i) for i in df['canonical_smiles']]


des=[]
for mol in mols:
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    des.append(fp1_morgan_hashed)
pred_morgan = np.array(des)



# Convert c to a set for duplicate checking
c_set = {tuple(row) for row in c}
indices_to_keep = np.array([tuple(row) not in c_set for row in pred_morgan])
# Count the number of False in the boolean index array
num_duplicates = np.sum(~indices_to_keep)

# Remove duplicate rows
x_predict_filtered = pred_morgan[indices_to_keep]
df_filtered = df.loc[indices_to_keep]

# Reset index
df_filtered.reset_index(drop=True, inplace=True)

# Predict
y_predict_filtered = xgb.predict_proba(x_predict_filtered)

# Add prediction scores as the 1025th dimension to the fingerprint array
x_predict_with_scores = np.hstack((x_predict_filtered, y_predict_filtered[:,1:2].reshape(-1, 1)))

# Save npy file containing fingerprints and prediction scores
np.save("dupdeleted_morgan_with_scores.npy", x_predict_with_scores)

# Save to Excel file
df_predict_filtered = pd.concat([df_filtered, pd.DataFrame(y_predict_filtered[:,1:2],columns=['probability'])],axis=1)
df_predict_filtered.to_excel("ZincDatabase_in-vitro_PredictedScores.xlsx")
