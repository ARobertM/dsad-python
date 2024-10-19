import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_netflix = pd.read_csv("Netflix.csv")
df_coduri = pd.read_csv("CoduriTari.csv")

df_merged = pd.merge(df_netflix, df_coduri, on='Cod')

indicators = df_merged.columns[2:-2]
scaler = StandardScaler()
df_merged[indicators] = scaler.fit_transform(df_merged[indicators])
df_merged[indicators] = df_merged[indicators].round(3)

df_sorted = df_merged.sort_values(by='Internet', ascending=False)

df_output = df_sorted[['Cod', 'Tara_x'] + indicators.tolist()]

df_output.to_csv('Cerinta1.csv', index=False)
print(df_output.head())
# ex2
def coef_variatie(x):
    return x.std() / x.mean()

indicators = ['Librarie', 'CostLunarBasic', 'CostLunarStandard', 'CostLunarPremium', 'Internet', 'HDI', 'Venit', 'IndiceFericire', 'IndiceEducatie']
df_numeric = df_merged[indicators + ['Continent']].apply(pd.to_numeric, errors='coerce')
df_numeric['Continent'] = df_merged['Continent']

df_cv = df_numeric.groupby('Continent').agg(coef_variatie)
df_cv_sorted = df_cv.sort_values(by='Librarie', ascending=False)

df_cv_sorted = df_cv_sorted.round(3)

df_cv_sorted.to_csv('Cerinta2.csv', index=True)

# B
indicators = ['Librarie', 'CostLunarBasic', 'CostLunarStandard', 'CostLunarPremium', 'Internet', 'HDI', 'Venit', 'IndiceFericire', 'IndiceEducatie']
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_merged[indicators])

acp = PCA()
acp.fit(df_standardized)
scores = acp.transform(df_standardized)

variante = acp.explained_variance_ratio_
print("Variantele comp principale:", variante)

df_scores = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])])
df_scores['Cod'] = df_merged['Cod']
df_scores.to_csv('scoruri.csv', index=False)

plt.figure(figsize=(10, 7))
plt.scatter(df_scores['PC1'], df_scores['PC2'])
for i, txt in enumerate(df_scores['Cod']):
    plt.annotate(txt, (df_scores['PC1'][i], df_scores['PC2'][i]))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scorurile in primele doua axe principale')
plt.grid()
plt.savefig('pca_scores.png')
plt.show()


