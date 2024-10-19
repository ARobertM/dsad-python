import numpy as np
import pandas as pd
from functii import *
from scipy.cluster.hierarchy import linkage
from pandas.api.types import is_numeric_dtype

# Partea 1- Media valorilor pe 5 ani
print("Cerinta1")

df_alcohol = pd.read_csv("date_IN/alcohol.csv", index_col=0)
df_alcohol1 = pd.read_csv("date_IN/alcohol.csv", index_col=0)
df_coduri = pd.read_csv("date_IN/CoduriTariExtins.csv", index_col=0)

v_num = list(df_alcohol)[1:]
valori_numerice = df_alcohol.iloc[:,1:]
# print(valori_numerice)
media = valori_numerice.mean(axis=1)
df_alcohol['Media'] = media
# print(df_alcohol)
df_cerinta1 = df_alcohol[['Code', 'Media']]
df_cerinta1.to_csv("date_OUT/Cerinta_1.csv",index=False)
# print(df_cerinta1)

# - Maximul in tarile componentelor respective
df_merge = df_alcohol.merge(right=df_coduri, left_index=True,right_index=True)
# etnicitate_judete = etnicitate_[etnii+["County"]].groupby(by="County").sum()
# print(v_num)
df_merge_continent = df_merge[v_num+['Continent']].groupby(by='Continent').mean()
# print(df_merge_continent)

cerinta2 = pd.DataFrame({
    'Continent_name': df_merge_continent.index,
    'Anul': df_merge_continent.idxmax(axis=1)
})
# print(cerinta2)
cerinta2.to_csv("date_OUT/Cerinta_2.csv",index=False)
# Partea B

def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    for col in t.columns:
        if t[col].isna().any():
            if is_numeric_dtype(t[col]):
                t.fillna({col: t[col].mean()}, inplace=True)
            else:
                t[col].fillna(t[col].mode()[0], inplace=True)

print(df_alcohol1)
valori_lipsa_inainte = df_alcohol1.isna().any().any()
print(valori_lipsa_inainte)

nan_replace(df_alcohol1)

valori_lipsa_dupa = df_alcohol1.isna().any().any()
print(valori_lipsa_dupa)

x = df_alcohol1[v_num].values #asa e bine
metoda = 'ward'
h = linkage(x,method=metoda)
# print(h)
matrice_ierarhie = pd.DataFrame(data=h, columns=['Cluster1','Cluster2','Distanta','Numar de instante'], index=range(1, len(h)+1))
print(matrice_ierarhie)