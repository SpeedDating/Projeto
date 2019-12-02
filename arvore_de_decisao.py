from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


df = pd.read_csv('speeddating.csv', low_memory=False, usecols=['pref_o_attractive', 'pref_o_sincere', 
'pref_o_intelligence', 'pref_o_ambitious', 'pref_o_funny', 'attractive', 'sincere', 
'intelligence', 'ambition', 'funny', 'decision_o'])

df = df[(df['pref_o_attractive'] != "?") & (df['pref_o_sincere'] != "?") & (df['pref_o_intelligence'] != "?") & 
(df['pref_o_ambitious'] != "?") & (df['pref_o_funny'] != "?") & (df['attractive'] != "?") & (df['sincere'] != "?") & 
(df['intelligence'] != "?") & (df['ambition'] != "?") & (df['funny'] != "?") & (df['decision_o'] != "?")]

def convertFloat(column):
  return column.astype('float64')

df = df.apply(convertFloat, axis=0)

# df['pref_o_attractive'] = df['pref_o_attractive'].astype('float64') 
# df['pref_o_sincere'] = df['pref_o_sincere'].astype('float64') 
# df['pref_o_intelligence'] = df['pref_o_intelligence'].astype('float64') 
# df['pref_o_ambitious'] = df['pref_o_ambitious'].astype('float64') 
# df['pref_o_funny'] = df['pref_o_funny'].astype('float64') 
# df['attractive'] = df['attractive'].astype('float64') 
# df['sincere'] = df['sincere'].astype('float64') 
# df['intelligence'] = df['intelligence'].astype('float64') 
# df['ambition'] = df['ambition'].astype('float64') 
# df['funny'] = df['funny'].astype('float64') 
# df['decision_o'] = df['decision_o'].astype('int32') 

def verifyHundred(row):
  soma = row['pref_o_attractive'] + row['pref_o_sincere'] + row['pref_o_intelligence'] + row['pref_o_ambitious'] + row['pref_o_funny']
  if soma != 100:
    fator =  100 / soma
    row['pref_o_attractive'] *= fator
    row['pref_o_sincere'] *= fator
    row['pref_o_intelligence'] *= fator
    row['pref_o_ambitious'] *= fator
    row['pref_o_funny'] *= fator
  return row


df_pessoasdecision_o = df[(df["decision_o"] == 1)]
df_pessoasNotdecision_o = df[(df["decision_o"] == 0)]

df_dados = pd.concat([df_pessoasdecision_o , df_pessoasNotdecision_o])

df_dados = df_dados.apply(verifyHundred, axis=1)

df_test = pd.concat([df_pessoasdecision_o.head(1500) , df_pessoasNotdecision_o.head(1500)])
# 2739 3793
yTeste = df_test[['decision_o']]

df_dados = df_dados.drop(df_test.index)
df_comp = df_dados.copy()

del df_test['decision_o']
del df_dados['decision_o']

# print(df_test.head())


classifier_dt = DecisionTreeClassifier(random_state=1984, criterion='gini', max_depth=11)
classifier_dt.fit(df_test, yTeste)

result = classifier_dt.predict(df_dados)

# print(result)
df_comp['result'] = result
# print(df_comp.head(25))

correto = df_comp[(df_comp['decision_o']) == (df_comp['result'])]['result'].count()
df_correto = df_comp[(df_comp['decision_o']) == (df_comp['result'])].copy()
incorreto = df_comp[(df_comp['decision_o']) != (df_comp['result'])]['result'].count()
df_incorreto = df_comp[(df_comp['decision_o']) != (df_comp['result'])].copy()

print(correto)
print(incorreto)
print('%.2f%%' % (correto / (correto + incorreto) * 100))

del df_incorreto['decision_o']
del df_incorreto['result']

while True:
  atratividade = int(input('\nAtratividade: '))
  sinceridade = int(input('Sinceridade: '))
  inteligencia = int(input('Inteligência: '))
  ambicao = int(input('Ambição: '))
  graca = int(input('Graça: '))
  lista = [atratividade, sinceridade, inteligencia, ambicao, graca]
  listas = [lista + [8,4,7,8,6], lista + [5,6,1,3,4], lista + [4,5,9,6,3], lista + [1,4,3,6,5], lista + [9,5,2,1,5]]
  new_df = pd.DataFrame(np.array(listas), columns=['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_ambitious', 'pref_o_funny', 'attractive', 'sincere', 'intelligence', 'ambition', 'funny'])
  print(classifier_dt.predict(new_df))
# [[20,20,20,20,20,7,6,7,5,6]]
