import pandas as pd
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('speeddating.csv', low_memory=False)

df.loc[df['gender'] == 'female', 'gender'] = 1
df.loc[df['gender'] == 'male', 'gender'] = 0

df_idades = df[(df["age"] != "?") & (df["age_o"] != "?")] 
df_idades["diferenca"] = df_idades['age_o'].astype(int) - df_idades['age'].astype(int)

df_idades = df_idades[(df_idades["gender"] == 1)]
df_final = df_idades[['gender','age','age_o','decision']]
df_preferemHomensMaisVelhos = df_idades.loc[(df_idades['diferenca'] >= 5) & ((df_idades["decision"].astype(int)) == 1)]

df_preferemHomensMaisNovos = df_idades.drop(df_preferemHomensMaisVelhos.index)

#print(df_preferemHomensMaisVelhos[['gender','age','age_o','decision']])
#print(df_preferemHomensMaisNovos[['gender','age','age_o','decision']])

df_final.to_csv('idades.csv', index = False)

df_dados = pd.concat([df_preferemHomensMaisVelhos[['gender','age','age_o','decision']].head(100) , df_preferemHomensMaisNovos[['gender','age','age_o','decision']].head(150)])
marcacoes = [1] * 100 + [-1] * 150

modelo = MultinomialNB()
modelo.fit(df_dados,marcacoes)
resultado = list(modelo.predict(df_final))
print(resultado.count(1))