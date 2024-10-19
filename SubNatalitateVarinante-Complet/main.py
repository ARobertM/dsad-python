
#ACP - varianta
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#A
#Cerinta 1
df_sporuri = pd.read_csv("dateIN/Mortalitate.csv")
df_coduri = pd.read_csv("dateIN/CoduriTariExtins.csv")
valori_numerice = list(df_sporuri)[1:]

df_rs_negativ = df_sporuri[df_sporuri['RS'] < 0]
df_cerinta1 = pd.DataFrame({
    "Tara" : df_rs_negativ['Tara'],
    'Rata Sporului Natural': df_rs_negativ['RS']
})
df_cerinta1.to_csv("dateOUT/Cerinta1.csv",index=False)

#Cerinta 2
df_merge = df_sporuri.merge(right=df_coduri,left_index=True,right_index=True)
df_continent = df_merge[valori_numerice+['Continent']].groupby(by='Continent').mean()

df_cerinta2 = pd.DataFrame(df_continent)
df_cerinta2.to_csv("dateOUT/Cerinta2.csv",index=True)

#B
#Cerinta 3 - Variantele componentelor principale
indicatori = list(df_sporuri.columns[1:])
x = ((df_sporuri[indicatori].values - np.mean(
    df_sporuri[indicatori].values,axis=0))/
     np.std(df_sporuri[indicatori].values,axis=0))
model_acp = PCA()
model_acp.fit(x)
v = model_acp.explained_variance_
scor = model_acp.transform(x)
df_scor = pd.DataFrame(scor, columns=['C'+str(scor+1) for scor in range(len(valori_numerice))],index=df_sporuri['Tara'])
print(df_scor)

#Cerinta desenare Grafice
def scatter_plot(t, v1, v2, titlu="Plot scoruri"):
    fig = plt.figure(titlu,figsize=(9,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu)
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)
    ax.scatter(t[v1], t[v2])
    for i in range(len(t)):
        ax.text(t[v1].iloc[i], t[v2].iloc[i], t.index[i])

scatter_plot(df_scor,"C1","C2")
plt.show()




