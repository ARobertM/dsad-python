import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

# Cerinta 1
df_industrie = pd.read_csv("Industrie.csv",index_col=0)
df_pop = pd.read_csv("PopulatieLocalitati.csv",index_col=0)

df_merge = df_industrie.merge(right=df_pop,left_index=True,right_index=True)
print(df_merge)
val_num = list(df_merge)[1:-3]
df_media_cifra_afaceri = df_merge[val_num].apply(lambda x: x / df_merge['Populatie'])
df_media_cifra_afaceri['Localitate'] = df_pop['Localitate']
df_media_cifra_afaceri.to_csv("Cerinta1.csv")
# Cerinta 2
df_merge_judet = df_merge[val_num+['Judet']].groupby(by='Judet')[val_num].sum()
print(df_merge_judet)

categorie_maxima = df_merge_judet[val_num].idxmax(axis=1)
valoare_maxima = df_merge_judet[val_num].max(axis=1)
df_cerinta2 = pd.DataFrame({
    'Judet':df_merge_judet.index,
    'Activitate':categorie_maxima,
    'Cifra de afaceri':valoare_maxima
})
print(df_cerinta2)
df_cerinta2.to_csv("Cerinta2.csv",index=False)

# Train la date
variabile = list(df_merge)
predictori = df_merge.columns[1:-3]
tinta = 'Judet'

X = df_merge[predictori]
y = df_merge[tinta]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Analiza Discriminanta
# z -> modelul liniar
model_LDA = LinearDiscriminantAnalysis()
model_LDA.fit(X_train,y_train)

#Calcul de scoruri
z = model_LDA.transform(X_test)
etichete_z = ["z"+ str(i+1) for i in range(z.shape[1])]
df_z = pd.DataFrame(z, index=X_test.index, columns=etichete_z)
print(df_z)

#Plot distributie z
def plot_distributie(z,y,k=0):
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Distributie in axa discriminanta "+str(k+1))
    sb.kdeplot(x=z[:, k], hue=y, fill=True)

for i in range(z.shape[1]):
    plot_distributie(z,y_test,i)

plt.show()





