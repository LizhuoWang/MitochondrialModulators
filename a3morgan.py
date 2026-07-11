# -*- coding: utf-8 -*-
"""

"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import pandas as pd
import numpy as np


df = pd.read_csv("ChemblMolsGS.csv")

mols = [Chem.MolFromSmiles(i) for i in df['canonical_smiles']]

value = df['class'].values
tmp = []
for i in range(len(value)):
    tmp.append(value[i])

des=[]
for mol in mols:
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    des.append(fp1_morgan_hashed)

similar = 0.95

for i in range(len(des)):
    j = i + 1
    aaa = []
    while i < len(des) and j < len(des):
        if DataStructs.DiceSimilarity(des[i],des[j]) > 0.95 and tmp[i]==tmp[j]:
            del(des[j])
            del(tmp[j])
            continue
        else:

            j += 1

a = np.array(des)
b = np.array(tmp)
b= b.reshape(-1, 1)
c = np.concatenate((a,b),axis=1)

# we have deleted the ones with similarity, but not the ones duplicated with different "tmp" classes.

# Convert NumPy array to DataFrame
df_c = pd.DataFrame(c)
df_c.rename(columns={1024: "class"}, inplace=True)


def dup_deletion_dataframe(df):
    # Group rows by the first 1024 columns (identical fingerprints)
    groups = df.groupby(list(range(1024)))

    # Create an empty list to store DataFrames
    frames = []

    # Iterate over each group
    for _, group in groups:
        # If the group has only one row, add it directly to frames
        if len(group) == 1:
            frames.append(group)
        else:
            # If class is the same, keep only the first row
            if len(group['class'].unique()) == 1:
                frames.append(group.iloc[[0]])
            else:
                # If classes differ, keep only the row with class = 1
                class_1_rows = group[group['class'] == 1]
                if not class_1_rows.empty:
                    frames.append(class_1_rows.iloc[[0]])

    # Concatenate all DataFrames in the frames list
    filtered_df = pd.concat(frames, ignore_index=True)

    # Reset index
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

# Call the function and save the result to a new DataFrame
filtered_df = dup_deletion_dataframe(df_c)



'''
Combine with any other datasets needed.
'''

# Read company and in-house molecules
df_cg = pd.read_csv("MCEandLabMolsCanoSmiles.csv")


mols_cg = [Chem.MolFromSmiles(i) for i in df_cg['canonical_smiles']]

des_cg=[]
for mol in mols_cg:
    fp2_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    des_cg.append(fp2_morgan_hashed)
cg_morgan = np.array(des_cg)

# Convert Morgan fingerprint array to DataFrame
df_cg_morgan = pd.DataFrame(cg_morgan)

# Add a column named "class" and fill with value 1
df_cg_morgan['class'] = 1

# Merge
df_combined = pd.concat([filtered_df, df_cg_morgan], axis=0, ignore_index=True)

# Call the function again to check for duplicates

filtered_df_all = dup_deletion_dataframe(df_combined)


# Convert DataFrame back to NumPy array
comb_dup = filtered_df_all.values

np.save("traindata_gaussian.npy", comb_dup)
