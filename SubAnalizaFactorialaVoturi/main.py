import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import factor_analyzer as fact
import seaborn as sb

df_voturi = pd.read_csv("dateIN/Vot_Categorii_Sex.csv",index_col=0)
df_coduri = pd.read_csv("dateIN/Coduri_Localitati.csv",index_col=0)

#Cerinta 1 - inregistrarea celui mai mic procent de prezenta la vot
valori_numerice = list(df_voturi)[1:]
categorii = list(df_voturi.columns[1:].values)
df1 = df_voturi.merge(right=df_coduri, right_index=True, left_index=True)
#val minima din fiecare rand in categorii
min_values = df1[categorii].min(axis=1)
min_categorie = df1[categorii].idxmin(axis=1)

df2 = pd.DataFrame({'Localitate' : df1['Localitate'], 'Categorie' : min_categorie})
# print(df2)
#Cerinta 2
df_merge_judet = df1[valori_numerice+['Judet']].groupby(by='Judet').mean()
# print(df_merge_judet)

#B- Analiza Factoriala
# 1. Aplicarea testului Bartlett de relevanta

x = df_voturi[categorii].values
bartlett_test = fact.calculate_bartlett_sphericity(x)
print("P-Value : ",bartlett_test[1])
if bartlett_test[1] < 0.001:
    print("Nu exista fact comuni")

# Testul KMO - calculeaza index-ul (Kaiser, Meyer, Olkin)
# - arata care variabile sunt mai putin corelate in rap cu celelalte

kmo = fact.calculate_kmo(x)
if kmo[1] < 0.5:
    print("Nu exista factori comuni relevanti")

df_kmo = pd.DataFrame(data={'IndexKMO': np.append(kmo[0],kmo[1])}, index=valori_numerice + ['Total'])
print(df_kmo)
# Corelograma
def corelograma(x, valMin = -1, valMax = 1, titlu="Corelatii factoriale"):
    fig = plt.figure(titlu,figsize=(9,9))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu, fontdict={
        "fontsize":14,
        "color":"b"
    })
    ax_ = sb.heatmap(data=x,vmin=valMin, vmax=valMax, cmap='bwr', annot=True, ax=ax)
    # ax_.set_xticklabels(labels=x.columns, ha='right', rotation=30)

corelograma(df_kmo,valMin=0,titlu='Index KMO');
# plt.show()

# 2. Scoruri factoriale + Construire Model
n, m = np.shape(x)
modelAF = fact.FactorAnalyzer(n_factors=m,rotation=None)

etichete_factori = ["F"+str(i+1) for i in range(m)]
scoruri_factoriale = modelAF.transform(x)
print(scoruri_factoriale)




#m = len(categorii)
# model_AF = fact.FactorAnalyzer(n_factors=m,rotation=None)
# model_AF.fit(x)
# f = model_AF.transform(x)
# t_f = pd.DataFrame(f,df_voturi.index, ["f" + str(i) for i in range(1, m + 1)])
# t_f.to_csv("f.csv")

# 3. ScatterPlot
def scatter_plot(t, v1, v2, titlu="Plot scoruri", corelatii=False):
    fig = plt.figure(figsize=(8,6))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1,aspect=1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(titlu)
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)
    ax.axvline(0)
    ax.axhline(0)
    ax.scatter(t[v1], t[v2],c="r")
    # for i in range(len(t)):
    #      ax.text(t[v1].iloc[i],t[v2].iloc[i],t.index[i])

# scatter_plot(t_f,"f1","f2")
# plt.show()






