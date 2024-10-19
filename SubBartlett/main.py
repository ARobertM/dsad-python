import numpy as np
import pandas as pd

# Cerinta 1
df_vot = pd.read_csv("dateIN/VotBUN.csv")
df_cod = pd.read_csv("dateIN/Coduri_Localitati.csv")
valori_numerice = list(df_vot)[2:]

voturi_minime = df_vot[valori_numerice].min().min()
categorie_minima = df_vot[valori_numerice].min().idxmin()
index_minim = df_vot[df_vot[categorie_minima] == voturi_minime].index[0]
minimul = df_vot.loc[index_minim]

cerinta1 = pd.DataFrame({
    "Siruta": [minimul['Siruta']],
    "Localitate": [minimul['Localitate']],
    "Categorie": [categorie_minima],
})
print(cerinta1)
# cerinta1.to_csv("dateOUT/Cerinta_1.csv",index=False)
#Cerinta 2
# df_merge_continent = df_merge[v_num+['Continent']].groupby(by='Continent').mean()
df_merge = df_vot.merge(right=df_cod,left_index=True,right_index=True)
df_merge_cod = df_merge[valori_numerice+['Judet']].groupby(by='Judet').mean()






