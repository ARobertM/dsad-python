import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
# Standardizare
from sklearn.preprocessing import StandardScaler
# PCA
from sklearn.decomposition import PCA
# AF - Analiza Factoriala
from factor_analyzer import FactorAnalyzer,calculate_bartlett_sphericity,calculate_kmo
# Cluster
import scipy.cluster.hierarchy as hclust
# Grafice
import matplotlib.pyplot as plt
import seaborn as sb

# # ------- Analiza Componentelor Principale
# df_mortalitate = pd.read_csv("Mortalitate.csv",index_col=0)
#
# def nan_replace(t):
#     assert isinstance(t,pd.DataFrame)
#     for i in t.columns:
#         if any(t[i].isna()):
#             t[i].fillna(t[i].mean(),inplace=True)
#         else:
#             t[i].fillna(t[i].mode()[0],inplace=True)
# nan_replace(df_mortalitate)
#
# scalar = StandardScaler()
# valori_standardizate = scalar.fit_transform(df_mortalitate)
# print(valori_standardizate)
#
# # Initializam modelul
# modelACP = PCA()
# C = modelACP.fit_transform(valori_standardizate)
#
# # Varianta componentelor principale
# varianta = modelACP.explained_variance_
# print("Varianta: ",varianta)
#
# # Scoruri
# scoruri = C / np.sqrt(varianta)
# etichetari = ["C"+str(i+1) for i in range(len(varianta))]
# df_scoruri = pd.DataFrame(data=scoruri, index=df_mortalitate.index,columns=etichetari)
# print(df_scoruri)
#
# #Plot scoruri grafic
# plt.figure("Plot Scoruri, Fig1",figsize=(9,9))
# plt.subplot(1,1,1)
# plt.xlabel('C1')
# plt.ylabel('C2')
# plt.scatter(df_scoruri['C1'], df_scoruri['C2'],edgecolors='y')
# for index,(x,y) in df_scoruri[['C1','C2']].iterrows():
#     plt.text(x,y,index)
#
#
# # Corelatii factoriale
# r_x_c = np.corrcoef(valori_standardizate,C,rowvar=False)[:len(varianta),len(varianta):]
# df_corelatii = pd.DataFrame(data=r_x_c,index=df_mortalitate.columns,columns=[etichetari])
# print(df_corelatii)
# # Corelograma
# plt.figure("Corelograma",figsize=(9,9))
# plt.subplot(1,1,1)
# plt.title("Corelograma corelatiilor factoriale",color="b")
# sb.heatmap(data=r_x_c,vmin=-1,vmax=1,annot=True)
# # Calcul Contributii
# C_patrat = C * C
# contributii = C_patrat / np.sum(C_patrat,axis=0)
# df_contributii = pd.DataFrame(data=contributii,index=df_mortalitate.index,columns=etichetari)
# print(df_contributii)
#
# # Calcul Cosinusuri
# cosinusuri = np.transpose(C_patrat.T / np.sum(C_patrat,axis=1))
# df_cosinusuri = pd.DataFrame(data=cosinusuri,index=df_mortalitate.index,columns=etichetari)
# print(df_cosinusuri)
#
# # Comunalitati
# comunalitati = np.cumsum(r_x_c*r_x_c,axis=1)
# df_comunalitati = pd.DataFrame(data=comunalitati,index=df_mortalitate.columns,columns=etichetari)
# print(df_comunalitati)
# # Corelograma comunalitati
# plt.figure("Corelograma comunalitati",figsize=(9,9))
# plt.subplot(1,1,1)
# plt.title("Corelograma comunalitati",color='r')
# sb.heatmap(data=df_comunalitati,vmax=1,vmin=-1,annot=True)
#
# plt.show()

# #  ---------- Analiza Factoriala
# df_voturi = pd.read_csv('VotBUN.csv',index_col=0)
# print(df_voturi)
#
# def nan_replace(t):
#     assert isinstance(t,pd.DataFrame)
#     for i in t.columns:
#         if any(t[i].isna()):
#             if is_numeric_dtype(t[i]):
#                 t[i].fillna(t[i].mean(),inplace=True)
#             else:
#                 t[i].fillna(t[i].mode()[0],inplace=True)
# nan_replace(df_voturi)
# set_date = list(df_voturi.columns[1:])
# df_voturiFaraLoc = df_voturi[set_date]
#
# scaler = StandardScaler()
# date_standardizate = scaler.fit_transform(df_voturiFaraLoc)
# df_standardizat = pd.DataFrame(data=date_standardizate,index=df_voturiFaraLoc.index,columns=df_voturiFaraLoc.columns)
# print(date_standardizate)
# #  implemntare AF
# x = df_standardizat[set_date]
# nr_var = len(x.columns)
#
# modelAF = FactorAnalyzer(n_factors=nr_var,rotation=None)
# F = modelAF.fit(x)
# # Scoruri AF
# scoruri = modelAF.transform(x)
# etichete = ['F'+str(i+1) for i in range(len(set_date))]
# df_scoruri = pd.DataFrame(data=scoruri,index=df_voturiFaraLoc.index,columns=etichete)
# print(df_scoruri)
#
# # Plot de scoruri
# plt.figure("Fig1-Plot scoruri",figsize=(9,9))
# plt.subplot(1,1,1)
# plt.xlabel('F1')
# plt.ylabel('F2')
# plt.scatter(df_scoruri['F1'],df_scoruri['F2'],edgecolors='y')
# plt.title("Ploturi pentru scorurile analizei factoriale", color='b')
# for index,(X,Y) in df_scoruri[['F1','F2']].iterrows():
#     plt.text(X,Y,index)
#
# # Testul Bartlett
# bartlett_test = calculate_bartlett_sphericity(x)
# print("P-Value:", bartlett_test[1])
# print("Chi-Squared2: ",bartlett_test[0])
# # Testul KMO
# kmo_test= calculate_kmo(x)
# print("KMO Value: ",kmo_test[1])
#
# # Varianta factori
# varianta = modelAF.get_factor_variance()[0]
# print('Varianta:',varianta)
# # Corelatii factoriale
# corelatii = modelAF.loadings_
# df_corelatii = pd.DataFrame(data=corelatii,index=df_voturiFaraLoc.columns,columns=etichete)
# print(df_corelatii)
# # Corelograma Factoriala (PLOT)
# plt.figure("Corelograma Factoriala, Fig2", figsize=(9,9))
# plt.subplot(1,1,1)
# plt.title("Corelograma Factoriala")
# sb.heatmap(data=df_corelatii,vmin=-1,vmax=1,annot=True)
#
# # Comunalitati
# comunalitati = modelAF.get_communalities()
# df_comunalitati = pd.DataFrame(data=comunalitati,index=set_date,columns=['Comunalitati'])
# print(df_comunalitati)
#
# # Figura Comunalitati
# plt.figure("Comunalitati,Fig3",figsize=(9,9))
# plt.subplot(1,1,1)
# plt.title("Comunalitati",color='y')
# sb.heatmap(data=df_comunalitati,vmin=0,annot=True)
#
# plt.show()

# ----------- Analiza de clusteri
df_alcohol = pd.read_csv("alcohol.csv",index_col=0)

def nan_replace(t):
    assert isinstance(t,pd.DataFrame)
    for i in t.columns:
        if any(t[i].isna()):
            t[i].fillna(t[i].mean(),inplace=True)
        else:
            t[i].fillna(t[i].mode()[0],inplace=True)
nan_replace(df_alcohol)
print(df_alcohol)
set_date = list(df_alcohol.columns[1:])
df_alcoholFaraCod = df_alcohol[set_date]

#Standardizare date
scalar = StandardScaler()
date_standardizate = scalar.fit_transform(df_alcoholFaraCod)
df_standardizat = pd.DataFrame(data=date_standardizate,index=df_alcoholFaraCod.index,columns=df_alcoholFaraCod.columns)
print(df_standardizat)

# Initializare Model Cluster - prin metoda WARD
x = df_standardizat.values
metoda = 'ward'
h = hclust.linkage(x, method=metoda)
print("Type h: ",type(h))
# print("h",h)
